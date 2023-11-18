# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.


import numpy as np
import abc
from typing import NamedTuple
from numpy.typing import NDArray


class TimeStep(NamedTuple):
    observation: NDArray[np.float32]
    action: int
    reward: float
    done: bool
    next_observation: NDArray[np.float32]
    
class Agent(abc.ABC):
    @abc.abstractmethod
    def select_action(self, observation: NDArray[np.float32], step: int) -> int:
        pass
    
    @abc.abstractmethod
    def select_greedy_action(self, observation: NDArray[np.float32]) -> int:
        pass

    @abc.abstractmethod
    def update(self, timestep: TimeStep) -> None:
        pass
    
    
