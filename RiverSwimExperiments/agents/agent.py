from __future__ import annotations
import numpy.typing as npt
import numpy as np
from abc import ABC, abstractmethod
from typing import NamedTuple, Callable

class Experience(NamedTuple):
    s_t: int
    a_t: int
    r_t: float
    s_tp1: int

class AgentParameters(NamedTuple):
    dim_state_space: int
    dim_action_space: int
    discount_factor: float

class Agent(ABC):
    dim_state_space: int
    dim_action_space: int
    discount_factor: float
    exp_visits: npt.NDArray[np.float64]
    total_state_visits: npt.NDArray[np.float64]
    last_visit: npt.NDArray[np.float64]
    greedy_policy: npt.NDArray[np.int64]
    omega: npt.NDArray[np.float64]

    def __init__(self, agent_parameters: AgentParameters):
        self.dim_state_space = agent_parameters.dim_state_space
        self.dim_action_space = agent_parameters.dim_action_space
        self.discount_factor = agent_parameters.discount_factor
        self.exp_visits = np.zeros((self.ns, self.na, self.ns), order='C')
        self.total_state_visits = np.zeros((self.ns), order='C')
        self.last_visit = np.zeros((self.ns), order='C')
        self.greedy_policy = np.zeros((self.ns), dtype=np.int64, order='C')
        self.omega = np.ones((self.ns, self.na), order='C')
    
    @property
    def ns(self) -> int:
        return self.dim_state_space
    
    @property
    def na(self) -> int:
        return self.dim_action_space

    def forced_exploration_callable(self, state: int, step: int) -> float:
        #  max(0.1, 1 /((1+t) ** p.alpha)))
        c = 1
        return max(0.1, c / max(1, self.total_state_visits[state]))
    
    @abstractmethod
    def forward(self, state: int, step: int) -> int:
        raise NotImplementedError

    @abstractmethod
    def process_experience(self, experience: Experience, step: int) -> None:
        raise NotImplementedError

    def backward(self, experience: Experience, step: int) -> None:
        self.exp_visits[experience.s_t, experience.a_t, experience.s_tp1] += 1
        self.last_visit[experience.s_tp1] = step + 1
        self.total_state_visits[experience.s_tp1] += 1
        
        if step == 0:
            self.last_visit[experience.s_t] = step
            self.total_state_visits[experience.s_t] += 1

        self.process_experience(experience, step)
    
    