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
    name: str # Algo name

    def __init__(self, ns: int, na: int, discount_factor: float):
        self.ns = ns
        self.na = na
        self.discount_factor = discount_factor
        self.num_visits_state = np.zeros(self.ns)
        self.num_visits_actions = np.zeros((self.ns, self.na))
        self.last_visit_state = np.zeros(self.ns)
    
    def forward(self, state: int, step: int) -> int:
        self.num_visits_state[state] += 1
        self.last_visit_state[state] = step
        action = self._forward_logic(state, step)
        self.num_visits_actions[state][action] += 1
        return action
    
    def backward(self, experience: Experience):
        self._backward_logic(experience)
    
    def _forward_logic(self, state: int, step: int) -> int:
        raise NotImplementedError

    def _backward_logic(self, experience: Experience):
        raise NotImplementedError
    
    def greedy_action(self, state: int) -> int:
        raise NotImplementedErrorna
    
class QlearningAgent(Agent):
    name = 'qlearning'

    def __init__(self, ns: int, na: int, discount_factor: float, alpha: float, epsilon: float):
        super().__init__(ns, na, discount_factor)
        self.q_function = np.zeros((self.ns, self.na))
        self.alpha = alpha
        self.epsilon = epsilon
        
    def _forward_logic(self, state: int, step: int) -> int:
        eps = 1 if self.num_visits_state[state] <= self.na else self.epsilon
        action = np.random.choice(self.na) if np.random.uniform() < eps else self.q_function[state].argmax()
        return action

    def greedy_action(self, state: int) -> int:
        return self.q_function[state].argmax()

    def _backward_logic(self, experience: Experience):
        state, action, reward, next_state, done = list(experience)
        target = reward + (1-done) * self.discount_factor * self.q_function[next_state].max()
        lr = 1 / (self.num_visits_actions[state][action] ** self.alpha)
        self.q_function[state][action] += lr * (target - self.q_function[state][action])

class RandomAgent(Agent):
    name = 'random'

    def __init__(self, ns: int, na: int, discount_factor: float):
        super().__init__(ns, na, discount_factor)
        
    def _forward_logic(self, state: int, step: int) -> int:
        return np.random.choice(self.na)

    def greedy_action(self, state: int) -> int:
        return np.random.choice(self.na)
    def _backward_logic(self, experience: Experience):
        pass