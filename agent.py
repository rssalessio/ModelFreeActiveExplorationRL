import numpy as np
from abc import ABC
from typing import NamedTuple

class Experience(NamedTuple):
    state: int
    action: int
    reward: float
    next_state: int
    done: bool

class Agent(ABC):
    ns: int # Number of states
    na: int # Number of actions
    discount_factor: float # Discuount factor

    def __init__(self, ns: int, na: int, discount_factor: float):
        self.ns = ns
        self.na = na
        self.discount_factor = discount_factor
    
    def forward(self, state: int, step: int) -> int:
        raise NotImplementedError
    
    def backward(self, experience: Experience):
        raise NotImplementedError
    
    def greedy_action(self, state: int) -> int:
        raise NotImplementedError
    
class QlearningAgent(Agent):
    def __init__(self, ns: int, na: int, discount_factor: float, alpha: float):
        super().__init__(ns, na, discount_factor)
        self.num_visits_state = np.zeros(self.ns)
        self.num_visits_actions = np.zeros((self.ns, self.na))
        self.q_function = np.zeros((self.ns, self.na))
        self.alpha = alpha
        
    def forward(self, state: int, step: int) -> int:
        self.num_visits_state[state] += 1
        eps = 1 if self.num_visits_state[state] <= 2 * self.na else max(0.5, 1 / (self.num_visits_state[state] - 2*self.na))

        action = np.random.choice(self.na) if np.random.uniform() < eps else self.q_function[state].argmax()
        self.num_visits_actions[state][action] += 1
        return action

    def greedy_action(self, state: int) -> int:
        return self.q_function[state].argmax()

    def backward(self, experience: Experience):
        state, action, reward, next_state, done = list(experience)
        
        target = reward + (1-done) * self.discount_factor * self.q_function[next_state].max()
        
        lr = 1 / (self.num_visits_actions[state][action] ** self.alpha)
        self.q_function[state][action] += lr * (target - self.q_function[state][action])
