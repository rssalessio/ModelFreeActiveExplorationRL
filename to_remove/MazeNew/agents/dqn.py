from typing import Any, Callable, NamedTuple, Sequence, Optional
from numpy.typing import NDArray
import torch
import torch.nn as nn
import numpy as np
import copy
from .replay_buffer import ReplayBuffer

class DQN(object):
    """A simple DQN agent using TF2."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        network: nn.Module,
        batch_size: int,
        discount: float,
        replay_capacity: int,
        min_replay_size: int,
        sgd_period: int,
        target_update_period: int,
        optimizer: torch.optim.Optimizer,
        epsilon_fn: Callable[[int], float] = lambda _: 0.05,
        seed: Optional[int] = None,
    ):

        # Internalise hyperparameters.
        self._num_actions = num_actions
        self._state_dim = state_dim
        self._discount = discount
        self._batch_size = batch_size
        self._sgd_period = sgd_period
        self._target_update_period = target_update_period
        self._epsilon_fn = epsilon_fn
        self._min_replay_size = min_replay_size

        # Seed the RNG.
        torch.random.manual_seed(seed)
        self._rng = np.random.RandomState(seed)

        # Internalise the components (networks, optimizer, replay buffer).
        self._optimizer = optimizer
        self._replay = ReplayBuffer(capacity=replay_capacity)
        self._online_network = network
        self._target_network = copy.deepcopy(network)
        self._total_steps = 1

    @torch.no_grad()
    def select_action(self, observation: NDArray[np.float64]) -> int:
        # Epsilon-greedy policy.
        if self._rng.rand() < self._epsilon_fn(self._total_steps):
            return self._rng.randint(self._num_actions)

        
        observation = torch.tensor(observation[None, ...], dtype=torch.float32)
        # Greedy policy, breaking ties uniformly at random.
        q_values = self._online_network(observation).numpy()[0]
        idxs = np.isclose(q_values.max() - q_values, 0)
        action = self._rng.choice(idxs)
        return int(action)

    def update(
        self,
        observation: NDArray[np.float64],
        action: int,
        reward: float,
        new_observation: NDArray[np.float64],
        done: bool) -> Optional[float]:
        # Add this transition to replay.
        self._replay.add([
            observation,
            action,
            reward,
            new_observation,
            done
        ])

        self._total_steps += 1
        if self._total_steps % self._sgd_period != 0:
            return None

        if self._replay.size < self._min_replay_size:
            return None

        # Do a batch of SGD.
        transitions = self._replay.sample(self._batch_size)
        return self._training_step(transitions)

    def _training_step(self, transitions: Sequence[NDArray]) -> float:
        """Does a step of SGD on a batch of transitions."""
        o_tm1, a_tm1, r_t, o_t, d_t = transitions

        
        a_tm1 = torch.tensor(a_tm1, dtype=torch.int64, requires_grad=False)
        r_t = torch.tensor(r_t, dtype=torch.float32, requires_grad=False)
        d_t = torch.tensor(d_t, dtype=torch.float32, requires_grad=False)
        o_tm1 = torch.tensor(o_tm1, dtype=torch.float32, requires_grad=False)
        o_t = torch.tensor(o_t, dtype=torch.float32, requires_grad=False)

        
        
        with torch.no_grad():
            q_t = self._target_network(o_t).max(-1)[0]  # [B]
            target = r_t + (1-d_t) * self._discount * q_t
            
        
        self._optimizer.zero_grad()
        q_tm1 = self._online_network(o_tm1).gather(-1, a_tm1.unsqueeze(-1)).flatten()
        loss = nn.HuberLoss()(q_tm1, target.detach())
        loss.backward()
        self._optimizer.step()

        # Periodically copy online -> target network variables.
        if self._total_steps % self._target_update_period == 0:
            self._target_network.load_state_dict(self._online_network.state_dict())
        return loss.item()


def default_agent(state_dim: int,
                  num_actions: int,
                  seed: int = 0) -> DQN:
    """Initialize a DQN agent with default parameters."""

    def make_network(input_size: int, output_size: int) -> nn.Module:
        return nn.Sequential(*[
            nn.Flatten(),
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)])
    

    network = make_network(state_dim, num_actions)
    optim = torch.optim.Adam(network.parameters(), lr=1e-3)
    return DQN(
        state_dim=state_dim,
        num_actions=num_actions,
        network=network,
        optimizer=optim,
        batch_size=32,
        discount=0.99,
        replay_capacity=10000,
        min_replay_size=100,
        sgd_period=1,
        target_update_period=4,
        epsilon_fn=lambda t: 10 / (10 + t),
        seed=42,
    )