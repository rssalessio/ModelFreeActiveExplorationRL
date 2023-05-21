import numpy as np
import abc
from typing import NamedTuple
from numpy.typing import NDArray
import dm_env
from dm_env import specs
    
class Agent(abc.ABC):
    @abc.abstractmethod
    def select_action(self, timestep: dm_env.TimeStep):
        pass
    
    @abc.abstractmethod
    def select_greedy_action(self, observation: NDArray[np.float32]) -> int:
        pass

    @abc.abstractmethod
    def update(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep) -> None:
        pass
    
    
