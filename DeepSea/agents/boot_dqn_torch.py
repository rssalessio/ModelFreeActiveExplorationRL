import copy
from typing import Callable, NamedTuple, Optional, Sequence
from .replay_buffer import ReplayBuffer
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from .agent import TimeStep, Agent
from .ensemble_linear_layer import EnsembleLinear
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BootstrappedDqn(Agent):
    """Bootstrapped DQN with additive prior functions."""
    def __init__(
            self,
            state_dim: int,
            num_actions: int,
            ensemble: nn.Module,
            batch_size: int,
            discount: float,
            replay_capacity: int,
            min_replay_size: int,
            sgd_period: int,
            target_update_period: int,
            optimizer: torch.optim.Optimizer,
            mask_prob: float,
            noise_scale: float,
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
        self._replay = ReplayBuffer(capacity=replay_capacity)
        self._prior_scale = prior_scale

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

    def _step(self, transitions: Sequence[torch.Tensor]):
        """Does a step of SGD for the whole ensemble over `transitions`."""
        o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t = transitions
        a_tm1 = torch.tensor(a_tm1, dtype=torch.int64, requires_grad=False, device=device)
        r_t = torch.tensor(r_t, dtype=torch.float32, requires_grad=False, device=device)
        d_t = torch.tensor(d_t, dtype=torch.float32, requires_grad=False, device=device)
        o_tm1 = torch.tensor(o_tm1, dtype=torch.float32, requires_grad=False, device=device)
        o_t = torch.tensor(o_t, dtype=torch.float32, requires_grad=False, device=device)
        m_t = torch.tensor(m_t, dtype=torch.float32, requires_grad=False, device=device)
        z_t = torch.tensor(z_t, dtype=torch.float32, requires_grad=False, device=device)

        with torch.no_grad():
            q_target = self._target_ensemble(o_t).max(-1)[0]
            target_y = r_t.unsqueeze(-1) + z_t + self._discount * (1-d_t.unsqueeze(-1)) * q_target
            
        q_values = self._ensemble(o_tm1).gather(-1, a_tm1[:, None, None].repeat(1, self._ensemble.ensemble_size, 1)).squeeze(-1)
        q_values = torch.mul(q_values, m_t)
        target_y = torch.mul(target_y, m_t)
        
        self._optimizer.zero_grad()
        loss = nn.HuberLoss()(q_values, target_y.detach())
        loss.backward()
        self._optimizer.step()
    
        self._total_steps += 1

        # Periodically update the target network.
        if self._total_steps % self._target_update_period == 0:
            self._target_ensemble.load_state_dict(self._ensemble.state_dict())
        return loss.item()#np.mean(losses)

    @torch.no_grad()
    def _select_action(self, observation: NDArray[np.float32], greedy: bool=False) -> int:
        if greedy is False and self._rng.rand() < self._epsilon_fn(self._total_steps):
            return self._rng.randint(self._num_actions)
        
        observation = torch.tensor(observation[None, ...], dtype=torch.float32, device=device)
        # Greedy policy, breaking ties uniformly at random.
        q_values = self._ensemble(observation)[0, self._active_head].cpu().numpy()
        return int(q_values.argmax())
        
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


class TransitionWithMaskAndNoise(NamedTuple):
    o_tm1: NDArray[np.float64]
    a_tm1: int
    r_t: float
    d_t: float
    o_t: NDArray[np.float64]
    m_t: NDArray[np.int64]
    z_t: NDArray[np.float64]
    
      
        

def make_single_network(input_size: int, output_size: int, hidden_size: int, ensemble_size: int) -> nn.Module:
    return nn.Sequential(*[
        nn.Flatten(start_dim=-2),
        EnsembleLinear(input_size, hidden_size, ensemble_size),
        nn.ReLU(),
        EnsembleLinear(hidden_size, hidden_size, ensemble_size),
        nn.ReLU(),
        EnsembleLinear(hidden_size, output_size, ensemble_size)])

class EnsembleWithPrior(nn.Module):
    def __init__(self, input_size: int, output_size: int, prior_scale: float, ensemble_size: int, hidden_size: int = 32):
        super().__init__()
        
        self.ensemble_size = ensemble_size
        self._network = make_single_network(input_size, output_size, hidden_size, ensemble_size)
        self._prior_network = make_single_network(input_size, output_size, hidden_size, ensemble_size)
        self._prior_scale = prior_scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[None, ...].repeat(self.ensemble_size, 1, 1, 1)
        q_values = self._network.forward(x).swapaxes(0,1)
        prior_q_values = self._prior_network(x).swapaxes(0,1)
        return q_values + self._prior_scale * prior_q_values.detach()
  
def default_agent(
        obs_spec: NDArray,
        num_actions: int,
        num_ensemble: int = 20,
        prior_scale: float = 3,
        seed: int = 0) -> BootstrappedDqn:
    """Initialize a Bootstrapped DQN agent with default parameters."""

    state_dim = np.prod(obs_spec.shape)
    ensemble = EnsembleWithPrior(state_dim, num_actions, prior_scale, num_ensemble, 32).to(device)
    
    
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=1e-3)

    return BootstrappedDqn(
        state_dim=state_dim,
        num_actions=num_actions,
        ensemble=ensemble,
        prior_scale=prior_scale,
        batch_size=128,
        discount=.99,
        replay_capacity=100000,
        min_replay_size=128,
        sgd_period=1,
        target_update_period=4,
        optimizer=optimizer,
        mask_prob=.5,
        noise_scale=0.0,
        epsilon_fn=lambda t: 10 / (10 + t),
        seed=seed,
    )