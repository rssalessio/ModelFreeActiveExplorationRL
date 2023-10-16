import numpy as np
import abc
import torch
from typing import NamedTuple, Sequence, Tuple
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
    
    @abc.abstractmethod
    def get_models(self) -> Sequence[Tuple[str, torch.nn.Module]]:
        pass

    def save(self, filename: str) -> None:
        models = self.get_models()
        torch.save({
            model_name: model.state_dict() for (model_name, model) in models
        }, filename)
    
    def mode_vector(self, arr: NDArray) -> int:
        unique, counts = np.unique(arr, return_counts=True)
        return unique[counts.argmax()]