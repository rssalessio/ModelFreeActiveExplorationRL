"""A simple windowed buffer for accumulating sequences."""

from typing import NamedTuple
import numpy as np
from numpy.typing import NDArray
import torch

class Trajectory(NamedTuple):
    """A trajectory is a sequence of observations, actions, rewards, discounts.

    Note: `observations` should be of length T+1 to make up the final transition.
    """
    # TODO(b/152889430): Make this generic once it is supported by Pytype.
    observations: torch.Tensor  # [T + 1, ...]
    actions: torch.Tensor  # [T]
    rewards: torch.Tensor  # [T]
    discounts: torch.Tensor  # [T]
    cumulative_reward: torch.Tensor # [T]
    length: int

class SequenceBuffer:
    """A simple buffer for accumulating trajectories."""

    _observations: NDArray[np.float64]
    _actions: NDArray
    _rewards: NDArray[np.float64]
    _dones: NDArray[np.float64]

    _max_sequence_length: int
    _needs_reset: bool = True
    _t: int = 0

    def __init__(
        self,
        obs_spec: NDArray,
        max_sequence_length: int,
        discount_factor: float,
    ):
        """Pre-allocates buffers of numpy arrays to hold the sequences."""
        self._observations = np.zeros(
            shape=(max_sequence_length + 1, *obs_spec.shape), dtype=obs_spec.dtype)
        self._actions = np.zeros(
            shape=(max_sequence_length, 1),
            dtype=np.float64)
        self._rewards = np.zeros(max_sequence_length, dtype=np.float64)
        self._dones = np.zeros(max_sequence_length, dtype=np.float64)

        self._max_sequence_length = max_sequence_length
        self._discount = discount_factor

    def append(
      self,
      observation: NDArray[np.float64],
        action: int,
        reward: float,
        new_observation: NDArray[np.float64],
        done: bool):
        """Appends an observation, action, reward, and discount to the buffer."""
        if self.full():
            raise ValueError('Cannot append; sequence buffer is full.')

        # Start a new sequence with an initial observation, if required.
        if self._needs_reset:
            self._t = 0
            self._observations[self._t] = observation
            self._needs_reset = False

        # Append (o, a, r, d) to the sequence buffer.
        self._observations[self._t + 1] = new_observation
        self._actions[self._t] = action
        self._rewards[self._t] = reward
        self._dones[self._t] = done
        self._t += 1

        # Don't accumulate sequences that cross episode boundaries.
        # It is up to the caller to drain the buffer in this case.
        if done:
            self._needs_reset = True

    def drain(self) -> Trajectory:
        """Empties the buffer and returns the (possibly partial) trajectory."""
        if self.empty():
            raise ValueError('Cannot drain; sequence buffer is empty.')
        cum_rewards = torch.zeros(self._t, requires_grad=False, dtype=torch.float32)
        for i in range(self._t-1, -1, -1):
            next_cumrew = self._discount * cum_rewards[i+1] if i < self._t - 1 else 0
            cum_rewards[i] = self._rewards[i] + next_cumrew
        
        trajectory = Trajectory(
            torch.tensor(self._observations[:self._t + 1], requires_grad=False, dtype=torch.float32),
            torch.tensor(self._actions[:self._t], requires_grad=False, dtype=torch.int64),
            torch.tensor(self._rewards[:self._t], requires_grad=False, dtype=torch.float32),
            torch.tensor(self._dones[:self._t], requires_grad=False, dtype=torch.float32),
            cum_rewards,
            self._t
        )
        self._t = 0  # Mark sequences as consumed.
        self._needs_reset = True
        return trajectory

    def empty(self) -> bool:
        """Returns whether or not the trajectory buffer is empty."""
        return self._t == 0

    def full(self) -> bool:
        """Returns whether or not the trajectory buffer is full."""
        return self._t == self._max_sequence_length
