# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, NamedTuple, Optional, Sequence
from .replay_buffer import ReplayBuffer
from numpy.typing import NDArray
from .agent import TimeStep, Agent
from .ensemble_linear_layer import EnsembleLinear
from collections import deque
golden_ratio = (1+np.sqrt(5))/2
golden_ratio_sq = golden_ratio ** 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Values(NamedTuple):
    q_values: torch.Tensor
    m_values: torch.Tensor

class TransitionWithMaskAndNoise(NamedTuple):
    o_tm1: NDArray[np.float64]
    a_tm1: int
    r_t: float
    d_t: float
    o_t: NDArray[np.float64]
    m_t: NDArray[np.int64]
    z_t: NDArray[np.float64]
    
      
def make_single_network(input_size: int, output_size: int, hidden_size: int, ensemble_size: int, final_activation = nn.ReLU) -> nn.Module:
    net = [
        EnsembleLinear(input_size, hidden_size, ensemble_size) if ensemble_size > 1 else nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        EnsembleLinear(hidden_size, hidden_size, ensemble_size) if ensemble_size > 1 else nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        EnsembleLinear(hidden_size, output_size, ensemble_size) if ensemble_size > 1 else nn.Linear(hidden_size, output_size)]
    if final_activation is not None:
        net.append(final_activation())
    
    return nn.Sequential(*net)

class EnsembleWithPrior(nn.Module):
    def __init__(self, input_size: int, output_size: int, prior_scale: float, ensemble_size: int, hidden_size: int = 32, final_activation = nn.ReLU()):
        super().__init__()
        
        self.ensemble_size = ensemble_size
        self._network = make_single_network(input_size, output_size, hidden_size, ensemble_size, final_activation=final_activation)
        self._prior_network = make_single_network(input_size, output_size, hidden_size, ensemble_size, final_activation=final_activation)
        self._prior_scale = prior_scale

        def init_weights(m):
            if isinstance(m, EnsembleLinear):
                stddev = 1 / np.sqrt(m.weight.shape[1])
                torch.nn.init.trunc_normal_(m.weight, mean=0, std=stddev, a=-2*stddev, b=2*stddev)
                torch.nn.init.zeros_(m.bias.data)
        
        self._prior_network.apply(init_weights)
        self._network.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[None, ...].repeat(self.ensemble_size, 1, 1)
        values = self._network.forward(x).swapaxes(0,1)
        prior_values = self._prior_network(x).swapaxes(0,1)
        
        return values + self._prior_scale * prior_values.detach()

class ValueEnsembleWithPrior(nn.Module):
    def __init__(self, input_size: int, output_size: int, prior_scale: float, ensemble_size: int, hidden_size: int = 32):
        super().__init__()

    
        self.ensemble_size = ensemble_size
        self._q_network = EnsembleWithPrior(input_size, output_size=output_size, prior_scale=prior_scale, ensemble_size=ensemble_size,
                                            hidden_size=hidden_size, final_activation=None)
        self._m_network = EnsembleWithPrior(input_size, output_size=output_size, prior_scale=prior_scale, ensemble_size=ensemble_size,
                                            hidden_size=hidden_size, final_activation=nn.ReLU)

    def forward(self, x: torch.Tensor) -> Values:
        q = self._q_network.forward(x)
        m = self._m_network.forward(x)
        return Values(q, m) 

class DBMFBPI(Agent):
    """Deep-Bootstrapped Model-Free BPI"""
    def __init__(
            self,
            state_dim: int,
            num_actions: int,
            ensemble: ValueEnsembleWithPrior,
            greedy_network: nn.Module,
            batch_size: int,
            discount: float,
            replay_capacity: int,
            min_replay_size: int,
            sgd_period: int,
            target_update_period: int,
            optimizer: torch.optim.Optimizer,
            optimizer_greedy: torch.optim.Optimizer,
            mask_prob: float,
            noise_scale: float,
            delta_min: float,
            kbar: float = 1.,
            epsilon_fn: Callable[[int], float] = lambda _: 0.,
            prior_scale: float = 3,
            seed: Optional[int] = None):
        # Agent components.
        self._state_dim = state_dim
        self._ensemble = ensemble        
        self._target_ensemble = copy.deepcopy(ensemble)
        self._greedy_network = greedy_network
        self._target_greedy_network = copy.deepcopy(greedy_network)
        self._kbar = kbar

        self._num_ensemble = ensemble.ensemble_size
        self._optimizer = optimizer
        self._optimizer_greedy = optimizer_greedy
        self._replay = ReplayBuffer(capacity=replay_capacity)
        self._prior_scale = prior_scale
        self._delta_min = delta_min

        # Agent hyperparameters.
        self._num_actions = num_actions
        self._batch_size = batch_size
        self._sgd_period = sgd_period
        self._target_update_period = target_update_period
        self._min_replay_size = min_replay_size
        self._epsilon_fn = epsilon_fn
        self._mask_prob = mask_prob
        self._noise_scale = noise_scale
        self._rng = np.random.RandomState(seed)
        self._discount = discount

        # Agent state.
        self._total_steps = 1
        self._active_head = 0
        torch.random.manual_seed(seed)

        self._current_return = 0
        
        self._minimums = deque([], maxlen=200)
        self.uniform_number = np.random.uniform()
        self._overall_steps = 0
        self._pool_states = []

    def get_models(self):
        return [('QNetwork', self._ensemble)]

    def _step(self, transitions: Sequence[torch.Tensor]):
        """Does a step of SGD for the whole ensemble over `transitions`."""
        o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t = transitions
        a_tm1 = torch.tensor(a_tm1, dtype=torch.int64, requires_grad=False, device=device)
        r_t = torch.tensor(r_t, dtype=torch.float32, requires_grad=False, device=device)
        d_t = torch.tensor(d_t, dtype=torch.float32, requires_grad=False, device=device)
        o_tm1 = torch.tensor(o_tm1, dtype=torch.float32, requires_grad=False, device=device)
        o_t = torch.tensor(o_t, dtype=torch.float32, requires_grad=False, device=device)
        
        m_t = self._rng.binomial(1, self._mask_prob, (self._batch_size, self._num_ensemble)).astype(np.float32)
        
        m_t = torch.tensor(np.array(m_t), dtype=torch.float32, requires_grad=False, device=device)
        z_t = torch.tensor(z_t, dtype=torch.float32, requires_grad=False, device=device)
        

        with torch.no_grad():
            q_values_target = self._target_ensemble.forward(o_t).q_values
            next_actions = self._ensemble.forward(o_t).q_values.max(-1)[1]
            q_target = q_values_target.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)

            target_y = r_t.unsqueeze(-1) + z_t + self._discount * (1-d_t.unsqueeze(-1)) * q_target
            
            values_tgt = self._target_ensemble.forward(o_tm1).q_values
            q_values_tgt = values_tgt.gather(-1, a_tm1[:, None, None].repeat(1, self._ensemble.ensemble_size, 1)).squeeze(-1)
            M = (r_t.unsqueeze(-1) + z_t + (1-d_t.unsqueeze(-1)) * self._discount * q_target - q_values_tgt.detach()) / (self._discount)
            target_M = (M ** (2 * self._kbar)).detach()
    
        values = self._ensemble.forward(o_tm1)
        q_values = values.q_values.gather(-1, a_tm1[:, None, None].repeat(1, self._ensemble.ensemble_size, 1)).squeeze(-1)
        m_values = values.m_values.gather(-1, a_tm1[:, None, None].repeat(1, self._ensemble.ensemble_size, 1)).squeeze(-1)

        self._optimizer.zero_grad()
        loss1 = torch.mul(torch.square(q_values - target_y.detach()), m_t).mean()
        loss2 = torch.mul(torch.square(m_values - target_M.detach()), m_t).mean()
        loss = loss1 + loss2
        loss.backward()
        self._optimizer.step()

        # Periodically update the target network.
        if self._total_steps % self._target_update_period == 0:
            self._target_ensemble.load_state_dict(self._ensemble.state_dict())
        
        
        # Estimate deltamin
        with torch.no_grad():
            q_target = self._ensemble.forward(o_t).q_values
            estim_delta_min = (-(q_target.topk(2)[0].diff()))
            estim_delta_min = estim_delta_min[estim_delta_min > 0].min().cpu()
            self._minimums.append(estim_delta_min)
        H = 1 / (1-self._discount)
        alpha_t = (H + 1) / (H + self._total_steps)

        self._delta_min = alpha_t * self._delta_min + (1-alpha_t) * np.min(self._minimums)
            
        self._total_steps += 1

        
        return loss.item()

    @torch.no_grad()
    def _select_action(self, observation: NDArray[np.float32], greedy: bool=False) -> int:
        if greedy is False and self._rng.rand() < self._epsilon_fn(self._total_steps):
            return self._rng.randint(self._num_actions)
        
        observation = torch.tensor(observation, dtype=torch.float32, device=device)
        values  = self._ensemble.forward(observation)
        q_values = values.q_values[0].cpu().numpy().astype(np.float64)

        if greedy:
            return self.mode_vector(q_values.argmax(1))

        m_values = values.m_values[0].cpu().numpy().astype(np.float64)
        q_values = np.quantile(q_values, self.uniform_number, axis=0)
        m_values = np.quantile(m_values, self.uniform_number, axis=0)** (2 ** (1- self._kbar))

        mask = q_values == q_values.max()

        if len(q_values[~mask]) == 0:
            return np.random.choice(self._num_actions)
        delta = q_values.max() - q_values
        delta[mask] = self._delta_min * ((1 - self._discount)) / (1 + self._discount)


        Hsa = (2 + 8 * golden_ratio_sq * m_values) / np.clip((delta ** 2), 1e-16, np.inf)
        if np.any(np.isnan(Hsa)):
            return np.random.choice(self._num_actions)

        C = np.max(np.maximum(4, 16 * (self._discount ** 2) * golden_ratio_sq * m_values[mask]))
        Hopt = C / (delta[mask] ** 2)

        Hsa[mask] = np.sqrt(  Hopt * Hsa[~mask].sum(-1)* 2 / (self._state_dim * (1 - self._discount)))
        H = Hsa * 1e-14
        p = (H/H.sum(-1, keepdims=True))
        
        if np.any(np.isnan(p)):
            return np.random.choice(self._num_actions)

        return np.random.choice(self._num_actions, p=p)

    def select_action(self, observation: NDArray[np.float32], step: int) -> int:
        return self._select_action(observation)

    def select_greedy_action(self, observation: NDArray[np.float32]) -> int:
        return self._select_action(observation, greedy=True)
    
    def update(self, timestep: TimeStep) -> None:
        return self._update(
            np.float32(timestep.observation).flatten(), timestep.action, np.float32(timestep.reward),
            np.float32(timestep.next_observation).flatten(), timestep.done)
    
    def _update(
            self,
            observation: NDArray[np.float64],
            action: int,
            reward: float,
            new_observation: NDArray[np.float64],
            done: bool) -> Optional[float]:
        """Update the agent: add transition to replay and periodically do SGD."""

        self._current_return += reward
        if done:
            self._active_head = self._rng.randint(self._num_ensemble)
            self._current_return  = 0
            self.uniform_number = np.random.uniform()

        # Periodically sample a new uniform number
        if self._overall_steps % 2 * int(1/(1-self._discount)) == 0:
            self.uniform_number = np.random.uniform()

        self._overall_steps += 1
        self._replay.add(
            TransitionWithMaskAndNoise(
                o_tm1=observation,
                a_tm1=action,
                r_t=np.float32(reward),
                d_t=np.float32(done),
                o_t=new_observation,
                m_t=self._rng.binomial(1, self._mask_prob,
                                    self._num_ensemble).astype(np.float32),
                z_t=self._rng.randn(self._num_ensemble).astype(np.float32) *
                self._noise_scale,
            ))

        if self._replay.size < self._min_replay_size:
            self._pool_states.append(observation.flatten())
            return None

        if self._total_steps % self._sgd_period != 0:
            return None
        
        if self._total_steps == 1:
            self._pool_states = torch.tensor(np.vstack(self._pool_states), dtype=torch.float32, requires_grad=False).to(device)

        minibatch = self._replay.sample(self._batch_size)
        return self._step(minibatch)


def default_agent(
        obs_spec: NDArray,
        num_actions: int,
        num_ensemble: int = 20,
        prior_scale: float = 3,
        seed: int = 0) -> DBMFBPI:
    """Initialize DBMFBPI"""

    state_dim = np.prod(obs_spec.shape)
    ensemble = ValueEnsembleWithPrior(state_dim, num_actions, prior_scale, num_ensemble, 50).to(device)
    greedy_network = make_single_network(state_dim, num_actions, 32, 1, final_activation=None).to(device)
    
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=5e-4)
    optimizer_greedy = torch.optim.Adam(greedy_network.parameters(), lr=1e-4)

    return DBMFBPI(
        state_dim=state_dim,
        num_actions=num_actions,
        ensemble=ensemble,
        greedy_network=greedy_network,
        prior_scale=prior_scale,
        batch_size=128,
        discount=.99,
        replay_capacity=100000,
        min_replay_size=128,
        sgd_period=1,
        target_update_period=4,
        optimizer=optimizer,
        optimizer_greedy=optimizer_greedy,
        mask_prob=.7,
        noise_scale=0.0,
        delta_min=1e-2,
        kbar=2,
        epsilon_fn=lambda t:  10 / (10 + t),
        seed=seed,
    )