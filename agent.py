import numpy as np
from abc import ABC
from typing import NamedTuple
from BestPolicyIdentificationMDP.characteristic_time import CharacteristicTime, \
    compute_generative_characteristic_time, compute_characteristic_time_fw
from empirical_model import EmpiricalModel
from policy_iteration import policy_iteration

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
        self.num_visits_state = np.zeros(self.ns)
        self.num_visits_actions = np.zeros((self.ns, self.na))
    
    def forward(self, state: int, step: int) -> int:
        self.num_visits_state[state] += 1
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
        raise NotImplementedError
    
class QlearningAgent(Agent):
    def __init__(self, ns: int, na: int, discount_factor: float, alpha: float):
        super().__init__(ns, na, discount_factor)
        self.q_function = np.zeros((self.ns, self.na))
        self.alpha = alpha
        
    def _forward_logic(self, state: int, step: int) -> int:
        eps = 1 if self.num_visits_state[state] <= 2 * self.na else max(0.5, 1 / (self.num_visits_state[state] - 2*self.na))
        action = np.random.choice(self.na) if np.random.uniform() < eps else self.q_function[state].argmax()
        return action

    def greedy_action(self, state: int) -> int:
        return self.q_function[state].argmax()

    def _backward_logic(self, experience: Experience):
        state, action, reward, next_state, done = list(experience)
        target = reward + (1-done) * self.discount_factor * self.q_function[next_state].max()
        lr = 1 / (self.num_visits_actions[state][action] ** self.alpha)
        self.q_function[state][action] += lr * (target - self.q_function[state][action])

class GenerativeExplorativeAgent(Agent):
    def __init__(self, ns: int, na: int, discount_factor: float, min_epsilon: float = 0.05, alpha: float = 0.5, frequency_computation: int = 20, navigation_constraints: bool = False):
        super().__init__(ns, na, discount_factor)
        assert min_epsilon > 0, 'Min epsilon needs to be strictly positive'
        self.min_epsilon = min_epsilon
        self.model =  EmpiricalModel(self.ns, self.na)
        self.greedy_policy = None
        self.alpha = alpha
        self.allocation = None
        self.frequency_computation = frequency_computation
        self.navigation_constraints = navigation_constraints
        
    def _forward_logic(self, state: int, step: int) -> int:
        if self.num_visits_state[state] < self.na + 1:
            return np.random.choice(self.na)

        if self.allocation is None or step % self.frequency_computation == 0:
            if self.navigation_constraints is False:
                self.allocation = compute_generative_characteristic_time(self.discount_factor, self.model.transition_function,
                                                    self.model.reward)
            else:
               self.allocation = compute_characteristic_time_fw(self.discount_factor,
                                                                self.model.transition_function,
                                                                self.model.reward, with_navigation_constraints=True,
                                                                use_pgd=False, max_iter=100)
        allocation = self.allocation
        
        epsilon = max(self.min_epsilon, 1 / ((step + 1) ** self.alpha))
        omega = epsilon * np.ones(self.na)/self.na + (1 - epsilon) * allocation.omega[state] / allocation.omega[state].sum()
        q = step * omega - self.num_visits_actions[state]
        return q.argmax()

    def greedy_action(self, state: int) -> int:
        if self.greedy_policy is None:
            V, pi, Q = policy_iteration(self.discount_factor, self.model.transition_function, self.model.reward)
            self.greedy_policy = pi
        return self.greedy_policy[state]

    def _backward_logic(self, experience: Experience):
        self.model.update_visits(experience.state, experience.action, experience.next_state, experience.reward)
        self.greedy_policy = None