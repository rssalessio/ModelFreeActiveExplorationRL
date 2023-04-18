import copy
from typing import Callable, NamedTuple, Optional, Sequence
from .replay_buffer import ReplayBuffer
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import bisect
from .agent import TimeStep, Agent
from .ensemble_linear_layer import EnsembleLinear
import scipy.io as sio
from scipy.stats import weibull_min, kstest
import scipy.optimize
from collections import deque

golden_ratio = (1+np.sqrt(5))/2
golden_ratio_sq = golden_ratio ** 2


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
        nn.Flatten(start_dim=-2),
        EnsembleLinear(input_size, hidden_size, ensemble_size),
        nn.ReLU(),
        EnsembleLinear(hidden_size, hidden_size, ensemble_size),
        nn.ReLU(),
        EnsembleLinear(hidden_size, output_size, ensemble_size)]
    if final_activation is not None:
        net.append(final_activation())
    
    return nn.Sequential(*net)

class EnsembleWithPrior(nn.Module):
    def __init__(self, input_size: int, output_size: int, prior_scale: float, ensemble_size: int, hidden_size: int = 32, final_activation = nn.ReLU()):
        super().__init__()
        
        self.ensemble_size = ensemble_size
        self._network = make_single_network(input_size, output_size, hidden_size, ensemble_size, final_activation=final_activation)
        self._prior_network = make_single_network(input_size, output_size, hidden_size, ensemble_size, final_activation=final_activation)
    
        def init_weights(m):
            if isinstance(m, EnsembleLinear):
                gain = torch.nn.init.calculate_gain('relu')
                torch.nn.init.xavier_normal_(m.weight, gain)
                #torch.nn.init.zeros_(m.bias.data)
                m.bias.data.fill_(1e-3)
        self._prior_scale = prior_scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[None, ...].repeat(self.ensemble_size, 1, 1, 1)
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
    
class PolicyNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 32):
        super().__init__()
        self._policy_network = nn.Sequential(*[
            nn.Flatten(start_dim=-2),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=-1)]
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._policy_network(x)



class ExplorativeAgent2(Agent):
    """Bootstrapped DQN with additive prior functions."""
    def __init__(
            self,
            state_dim: int,
            num_actions: int,
            ensemble: ValueEnsembleWithPrior,
            policy: PolicyNetwork,
            batch_size: int,
            discount: float,
            replay_capacity: int,
            min_replay_size: int,
            sgd_period: int,
            target_update_period: int,
            optimizer: torch.optim.Optimizer,
            policy_optimizer: torch.optim.Optimizer,
            mask_prob: float,
            noise_scale: float,
            delta_min: float,
            epsilon_fn: Callable[[int], float] = lambda _: 0.,
            prior_scale: float = 3,
            seed: Optional[int] = None):
        """Bootstrapped DQN with additive prior functions."""
        # Agent components.
        self._state_dim = state_dim
        self._ensemble = ensemble        
        self._target_ensemble = copy.deepcopy(ensemble)

        self._num_ensemble = ensemble.ensemble_size
        self._optimizer = optimizer
        self._policy = policy
        self._policy_optimizer = policy_optimizer
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
        
        self._minimums = deque([], maxlen=200)

    def _step(self, transitions: Sequence[torch.Tensor]):
        """Does a step of SGD for the whole ensemble over `transitions`."""
        o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t = transitions
        a_tm1 = torch.tensor(a_tm1, dtype=torch.int64, requires_grad=False)
        r_t = torch.tensor(r_t, dtype=torch.float32, requires_grad=False)
        d_t = torch.tensor(d_t, dtype=torch.float32, requires_grad=False)
        o_tm1 = torch.tensor(o_tm1, dtype=torch.float32, requires_grad=False)
        o_t = torch.tensor(o_t, dtype=torch.float32, requires_grad=False)
        m_t = torch.tensor(m_t, dtype=torch.float32, requires_grad=False)
        z_t = torch.tensor(z_t, dtype=torch.float32, requires_grad=False)

        with torch.no_grad():
            q_target = self._target_ensemble.forward(o_t).q_values.max(-1)[0]
            target_y = r_t.unsqueeze(-1) + z_t + self._discount * (1-d_t.unsqueeze(-1)) * q_target
                    
            
        values = self._ensemble.forward(o_tm1)
        q_values = values.q_values.gather(-1, a_tm1[:, None, None].repeat(1, self._ensemble.ensemble_size, 1)).squeeze(-1)
        m_values = values.m_values.gather(-1, a_tm1[:, None, None].repeat(1, self._ensemble.ensemble_size, 1)).squeeze(-1)
        
        with torch.no_grad():
            values_tgt = self._ensemble.forward(o_tm1).q_values
            q_values_tgt = values_tgt.gather(-1, a_tm1[:, None, None].repeat(1, self._ensemble.ensemble_size, 1)).squeeze(-1)
            M = (r_t.unsqueeze(-1) + z_t + (1-d_t.unsqueeze(-1)) * self._discount * q_target - q_values_tgt.detach()) / (self._discount)
            target_M = (M ** 2).detach()
        
        
        q_values = torch.mul(q_values, m_t)
        target_y = torch.mul(target_y, m_t)
        
          
        m_values = torch.mul(m_values, m_t)
        target_M = torch.mul(target_M, m_t)
        
        
        self._optimizer.zero_grad()
        loss = nn.HuberLoss()(q_values, target_y.detach()) + nn.HuberLoss()(m_values, target_M.detach())
        loss.backward()
        self._optimizer.step()
            
        self._total_steps += 1

        # Periodically update the target network.
        if self._total_steps % self._target_update_period == 0:
                self._target_ensemble.load_state_dict(self._ensemble.state_dict())
          
        # Update network
        with torch.no_grad():
            values = self._ensemble.forward(o_t)
            qvalues = values.q_values
            mvalues = values.m_values
            
            mmu, mstd = mvalues.max(1)[0], mvalues.std(1) 
            qmu, qstd = qvalues.max(1)[0], qvalues.std(1)
            qvalues = qmu# + 2 * qstd
            mvalues = mmu# + 2 * mstd
            
            qmax, qmax_idxs = qvalues.max(1)
            mask = qmax_idxs.unsqueeze(-1).int()
            mask = torch.arange(0, self._num_actions).repeat(qmu.shape[0], 1) == mask
            
            delta = torch.clip(qmax.unsqueeze(-1) - qvalues, 1e-5, torch.inf)
            delta_subopt = delta[~mask]
            delta_min = delta_subopt.min()
            
            H_subopt = (2 + 8* golden_ratio_sq * mvalues[~mask]) / (delta_subopt.pow(2))
            H = ((2 + 8 * golden_ratio_sq * mvalues[mask]) / (delta_min.pow(2) * ((1 - self._discount) ** 2)))
            
            
        self._policy_optimizer.zero_grad()
        omega = self._policy(o_t)
        omega = omega / omega.sum()
        policy_loss =  (H_subopt / omega[~mask]).mean() + (H / omega[mask]).mean()
        policy_loss.backward()
        self._policy_optimizer.step()       
            
        return loss.item() + policy_loss.item()
    

    @torch.no_grad()
    def _select_action(self, observation: NDArray[np.float32], greedy: bool=False) -> int:
        observation = torch.tensor(observation[None, ...], dtype=torch.float32)
        probabilities  = self._policy.forward(observation)[0].numpy()
        if greedy:
            return probabilities.argmax()
        probabilities = probabilities / probabilities.sum()
        if np.random.uniform() < 1e-2:
            print(probabilities)
        return np.random.choice(self._num_actions, p=probabilities)

    def select_action(self, observation: NDArray[np.float32], step: int) -> int:
        return self._select_action(observation)

    def select_greedy_action(self, observation: NDArray[np.float32]) -> int:
        return self._select_action(observation, greedy=True)
    
    def update(self, timestep: TimeStep) -> None:
        return self._update(
            np.float32(timestep.observation), timestep.action, np.float32(timestep.reward),
            np.float32(timestep.next_observation), timestep.done)
    
    def _update(
            self,
            observation: NDArray[np.float64],
            action: int,
            reward: float,
            new_observation: NDArray[np.float64],
            done: bool) -> Optional[float]:
        """Update the agent: add transition to replay and periodically do SGD."""
        if done:
            self._active_head = self._rng.randint(self._num_ensemble)

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
            return None

        if self._total_steps % self._sgd_period != 0:
            return None
        minibatch = self._replay.sample(self._batch_size)
        return self._step(minibatch)



  
def default_agent(
        obs_spec: NDArray,
        num_actions: int,
        num_ensemble: int = 20,
        prior_scale: float = 3,
        seed: int = 0) -> ExplorativeAgent2:
    """Initialize a Bootstrapped DQN agent with default parameters."""

    state_dim = np.prod(obs_spec.shape)
    ensemble = ValueEnsembleWithPrior(state_dim, num_actions, prior_scale, num_ensemble, 32)
    
    policy = PolicyNetwork(state_dim, num_actions, hidden_size=32)
    
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=8e-3)
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=1e-3)

    return ExplorativeAgent2(
        state_dim=state_dim,
        num_actions=num_actions,
        ensemble=ensemble,
        policy=policy,
        prior_scale=prior_scale,
        batch_size=128,
        discount=.99,
        replay_capacity=100000,
        min_replay_size=128,
        sgd_period=1,
        target_update_period=4,
        optimizer=optimizer,
        policy_optimizer=policy_optimizer,
        mask_prob=.5,
        noise_scale=0.0,
        delta_min=1e-3,
        epsilon_fn=lambda t: 10 / (10 + t),
        seed=seed,
    )