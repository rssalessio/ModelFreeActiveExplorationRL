import numpy as np
import random
from numpy.typing import NDArray
from typing import NamedTuple, Tuple
from agents.agent import TimeStep

class MultiRewardsDeepSea(object):
    def __init__(self,
                 size: int = 10,
                 terminal_reward_parameter: float = 1.0,
                 enable_multi_rewards: bool = True,
                 seed: int = 0,
                 slipping_probability: float = 0.2,
                 randomize: bool = True):
        self._size = size
        self._move_cost = 0.01 / size
        
        assert terminal_reward_parameter > 0, 'Terminal reward should be strictly positive'
        self._rewards = np.linspace(0, terminal_reward_parameter, size)
        self._rewards
        self._terminal_reward_parameter = terminal_reward_parameter
        self._slipping_probability = slipping_probability

        if enable_multi_rewards is False:
            self._rewards[:-1] = 0
        
        self._enable_multi_rewards = enable_multi_rewards

        self._column = 0
        self._row = 0

        if randomize:
            rng = np.random.RandomState(seed)
            self._action_mapping = rng.binomial(1, 0.5, size)
            np.random.shuffle(self._rewards[:-1])
        else:
            self._action_mapping = np.ones(size)

        self._done = False
        self._compute_optimal_values()
    
    def _compute_optimal_values(self):
        Q = np.zeros((self._size - 1, self._size, 2))
        p = self._slipping_probability
        for row in reversed(range(self._size - 1)):
            for column in range(self._size):
                if column >= row + 1: break
                
                if row == self._size - 2:
                    Q[row, column, 0] = (1-p) * self._rewards[column] + p * self._rewards[column + 1] - self._move_cost
                    Q[row, column, 1] = (1-p) * self._rewards[column + 1] + p * self._rewards[column] - self._move_cost
                else:
                    Q[row, column, 0] = p * Q[row + 1, column + 1].max() + (1-p) * Q[row + 1, column].max() - self._move_cost 
                    Q[row, column, 1] = p * Q[row + 1, column].max() + (1-p) * Q[row + 1, column + 1].max() - self._move_cost 
        self._Q_values = Q

    def step(self, action: int) -> TimeStep:
        _current_observation = self._get_observation(self._row, self._column)
        
        if np.random.uniform() < self._slipping_probability:
            action = int(not action)

        if self._done:
            observation = self._get_observation(self._row, self._column)
            return TimeStep(_current_observation, action, 0, True, observation)

        # Remap actions according to column (action_right = go right)
        action_right = action == self._action_mapping[self._column]

        # Compute the reward
        reward = 0.0
        if self._row == self._size - 1:
            idx_col = self._column + 1 if action_right else self._column - 1
            idx_col = np.clip(idx_col, 0, self._size - 1)
            reward += 1 if np.random.uniform() < self._rewards[idx_col] else 0

        # State dynamics
        if action_right:  # right
            self._column = np.clip(self._column + 1, 0, self._size - 1)
            reward -= self._move_cost
        else:  # left
            self._column = np.clip(self._column - 1, 0, self._size - 1)

        # Compute the observation
        self._row += 1
        if self._row == self._size:
            observation = self._get_observation(self._row - 1, self._column)
            self._done = True
            return TimeStep(_current_observation, action, reward, True, observation)
        else:
            observation = self._get_observation(self._row, self._column)
            return TimeStep(_current_observation, action, reward, False, observation)

    def reset(self) -> NDArray[np.float32]:
        self._done = False
        self._column = 0
        self._row = 0
        observation = self._get_observation(self._row, self._column)

        return observation

    def _get_observation(self, row: int, column: int) -> NDArray[np.float32]:
        observation = np.zeros(shape=(self._size, self._size), dtype=np.float32)
        observation[row, column] = 1

        return observation

    @property
    def obs_shape(self) -> Tuple[int, int]:
        return self._size, self._size

    @property
    def num_actions(self) -> int:
        return 2

    @property
    def optimal_return(self) -> float:
        return self._Q_values[0,0].max() #self._terminal_reward_parameter - self._move_cost * self._size
    
if __name__ == '__main__':
    env = MultiRewardsDeepSea(10)
    print(env.reset())