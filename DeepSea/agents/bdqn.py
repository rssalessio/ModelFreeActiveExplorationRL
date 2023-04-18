from __future__ import annotations
import copy
from typing import Callable, NamedTuple, Optional, Sequence
from .replay_buffer import ReplayBuffer
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from .agent import TimeStep, Agent
from .ensemble_linear_layer import EnsembleLinear

class QNetworkWithSampling(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 32):
        super().__init__()

        self._network = nn.Sequential(*[
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()])
        
        self._out_layer = torch.normal(0, 1, size=(hidden_size, output_size), requires_grad=False)
        self._normal_distributions: Sequence[torch.distributions.MultivariateNormal] | None = None
        
    def set_gaussian(self, dists: Sequence[torch.distributions.MultivariateNormal]):
        self._normal_distribution = dists
    
    def sample(self):
        if self._normal_distributions is not None:
            self._out_layer = torch.hstack([dist.sample() for dist in self._normal_distributions])
        else:
            self._out_layer = torch.normal(0, 1, size=self._out_layer.shape, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phi = self._network.forward(x)
        return phi @ self._out_layer
    
    def copy_from(self, src: QNetworkWithSampling):
        self._network.load_state_dict(src._network.state_dict())
        self._normal_distributions = src._normal_distributions
        self._out_layer = src._out_layer

class BQDN(Agent):
    """Bootstrapped DQN with additive prior functions."""
    def __init__(
            self,
            state_dim: int,
            num_actions: int,
            network: QNetworkWithSampling,
            batch_size: int,
            discount: float,
            replay_capacity: int,
            min_replay_size: int,
            sgd_period: int,
            target_update_period: int,
            posterior_sampling_period: int,
            posterior_update_period: int,
            noise_scale_prior: float,
            noise_scale_likelihood: float,
            optimizer: torch.optim.Optimizer,
            seed: Optional[int] = None):
        """Bootstrapped DQN with additive prior functions."""
        # Agent components.
        self._state_dim = state_dim
        self._network = network        
        self._target_network = copy.deepcopy(network)

        self._optimizer = optimizer
        self._replay = ReplayBuffer(capacity=replay_capacity)
        self._posterior_sampling_period = posterior_sampling_period
        self._posterior_update_period = posterior_update_period
        self._noise_scale_prior = noise_scale_prior
        self._noise_scale_likelihood = noise_scale_likelihood

        # Agent hyperparameters.
        self._num_actions = num_actions
        self._batch_size = batch_size
        self._sgd_period = sgd_period
        self._target_update_period = target_update_period
        self._min_replay_size = min_replay_size
        self._rng = np.random.RandomState(seed)
        self._discount = discount

        # Agent state.
        self._total_steps = 1
        self._active_head = 0
        torch.random.manual_seed(seed)

    def _step(self, transitions: Sequence[torch.Tensor]):
        """Does a step of SGD for the whole ensemble over `transitions`."""
        o_tm1, a_tm1, r_t, d_t, o_t = transitions
        a_tm1 = torch.tensor(a_tm1, dtype=torch.int64, requires_grad=False)
        r_t = torch.tensor(r_t, dtype=torch.float32, requires_grad=False)
        d_t = torch.tensor(d_t, dtype=torch.float32, requires_grad=False)
        o_tm1 = torch.tensor(o_tm1, dtype=torch.float32, requires_grad=False)
        o_t = torch.tensor(o_t, dtype=torch.float32, requires_grad=False)

        with torch.no_grad():
            q_target = self._target_network(o_t).max(-1)[0]
            target_y = r_t + self._discount * (1-d_t) * q_target
            
        q_values = self._network(o_tm1).gather(-1, a_tm1.unsqueeze(-1)).squeeze(-1)
        self._optimizer.zero_grad()
        loss = nn.HuberLoss()(q_values, target_y.detach())
        loss.backward()
        self._optimizer.step()
    
        self._total_steps += 1

        # Periodically update the target network.
        if self._total_steps % self._target_update_period == 0:
                self._target_network.copy_from(self._network)
        return loss.item()#np.mean(losses)
    
    @torch.no_grad()
    def _update_posterior(self, transitions: Sequence[torch.Tensor]):
        """Does a step of SGD for the whole ensemble over `transitions`."""
        o_tm1, a_tm1, r_t, d_t, o_t = transitions
        a_tm1 = torch.tensor(a_tm1, dtype=torch.int64, requires_grad=False)
        r_t = torch.tensor(r_t, dtype=torch.float32, requires_grad=False)
        d_t = torch.tensor(d_t, dtype=torch.float32, requires_grad=False)
        o_tm1 = torch.tensor(o_tm1, dtype=torch.float32, requires_grad=False)
        o_t = torch.tensor(o_t, dtype=torch.float32, requires_grad=False)
        
        features = self._network._network(o_tm1)  # [B] x [d]

        dists = []
        for a in range(self._num_actions):
            idxs_a = torch.argwhere(a_tm1== a).flatten()
            phi_a = features[idxs_a]   # [B_a] x [d]
            
            cov_a = (phi_a.T @ phi_a) / (self._noise_scale_likelihood ** 2) + torch.eye(phi_a.shape[1]) / (self._noise_scale_prior ** 2)
            cov_a = torch.inverse(cov_a)
            mu_a = cov_a @ phi_a.T @ r_t[idxs_a] / (self._noise_scale_likelihood ** 2)
            dists.append(torch.distributions.MultivariateNormal(mu_a, cov_a))
        
        self._network.set_gaussian(dists)

    @torch.no_grad()
    def _select_action(self, observation: NDArray[np.float32]) -> int:        
        observation = torch.tensor(observation[None, ...], dtype=torch.float32)
        # Greedy policy, breaking ties uniformly at random.
        q_values = self._network(observation).numpy()[0]
        return int(q_values.argmax())
        
    def select_action(self, observation: NDArray[np.float32], step: int) -> int:
        return self._select_action(observation)

    def select_greedy_action(self, observation: NDArray[np.float32]) -> int:
        return self._select_action(observation)     
    
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

        self._replay.add(
            Transition(
                o_tm1=observation,
                a_tm1=action,
                r_t=np.float32(reward),
                d_t=np.float32(done),
                o_t=new_observation,
            ))

        if self._replay.size < self._min_replay_size:
            return None
            
        if self._total_steps % self._posterior_update_period == 0:
            minibatch = self._replay.sample(self._batch_size)
            self._update_posterior(minibatch)
            
        if self._total_steps % self._posterior_sampling_period == 0:
            self._network.sample()
            
        if self._total_steps % self._sgd_period == 0:
            minibatch = self._replay.sample(self._batch_size)
            return self._step(minibatch)


class Transition(NamedTuple):
    o_tm1: NDArray[np.float64]
    a_tm1: int
    r_t: float
    d_t: float
    o_t: NDArray[np.float64]   


  
def default_agent(
        obs_spec: NDArray,
        num_actions: int,
        seed: int = 0) -> BQDN:
    """Initialize a Bootstrapped DQN agent with default parameters."""

    state_dim = np.prod(obs_spec.shape)
    network = QNetworkWithSampling(state_dim, num_actions, 32)
    optimizer = torch.optim.Adam(network._network.parameters(), lr=1e-3)

    return BQDN(
        state_dim=state_dim,
        num_actions=num_actions,
        network=network,
        batch_size=128,
        discount=.99,
        replay_capacity=100000,
        min_replay_size=128,
        sgd_period=1,
        target_update_period=4,
        optimizer=optimizer,
        posterior_sampling_period=1,
        posterior_update_period=16,
        noise_scale_prior=1e-3,
        noise_scale_likelihood=1,
        seed=seed,
    )