import numpy as np
import torch
import torch.nn as nn
import pickle
from abc import ABC
from typing import NamedTuple
from BestPolicyIdentification import CharacteristicTime, \
    compute_characteristic_time, compute_generative_characteristic_time
from empirical_model import EmpiricalModel
from policy_iteration import policy_iteration
from network import Network
from scipy.special import rel_entr

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
        self.last_visit_state = np.zeros(self.ns)
        
        with open('data/lb_generative.pkl', 'rb') as f:
            self.generative_allocation: CharacteristicTime = pickle.load(f)
        with open('data/lb_with_constraints.pkl', 'rb') as f:
            self.constraints_allocation: CharacteristicTime = pickle.load(f)
        self.policy_diff_generative = []
        self.policy_diff_constraints = []
    
    def forward(self, state: int, step: int) -> int:
        self.num_visits_state[state] += 1
        self.last_visit_state[state] = step
        action = self._forward_logic(state, step)
        self.num_visits_actions[state][action] += 1
        
        if step % 20 == 0:
            p = self.num_visits_actions / self.num_visits_actions.sum()
            self.policy_diff_generative.append((step,rel_entr(p, self.generative_allocation.omega).sum()))
            self.policy_diff_constraints.append((step, rel_entr(p, self.constraints_allocation.omega).sum()))
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
        self.last_computation = 0

    def _forward_logic(self, state: int, step: int) -> int:               
        if self.num_visits_actions[state].min() < 2:
            return np.argmin(self.num_visits_actions[state])
        
        if self.allocation is None or step - self.last_computation > self.frequency_computation:
            self.last_computation = step
            if self.navigation_constraints is False:
                self.allocation = compute_generative_characteristic_time(self.discount_factor, self.model.transition_function,
                                                    self.model.reward)
            else:
                self.frequency_computation  = min(500, self.frequency_computation*1.2)
                omega = np.ones((self.ns * self.na)) if self.allocation is None else self.allocation.omega.flatten()
                omega = omega / omega.sum()
                self.allocation = compute_characteristic_time(self.discount_factor,
                                                                self.model.transition_function,
                                                                self.model.reward, with_navigation_constraints=True,
                                                                use_pgd=False, max_iter=100, x0=omega)

        allocation = self.allocation
        
        epsilon = max(self.min_epsilon, 1 / ((step + 1) ** self.alpha))
        omega = epsilon * (np.ones(self.na)/self.na) + (1 - epsilon) * allocation.omega[state] / allocation.omega[state].sum()

        if self.navigation_constraints is False:
            q = step * omega - self.num_visits_actions[state]
            return q.argmax()
        else:
            return np.random.choice(self.na, p = omega)

    def greedy_action(self, state: int) -> int:
        if self.greedy_policy is None:
            V, pi, Q = policy_iteration(self.discount_factor, self.model.transition_function, self.model.reward)
            self.greedy_policy = pi
        return self.greedy_policy[state]

    def _backward_logic(self, experience: Experience):
        self.model.update_visits(experience.state, experience.action, experience.next_state, experience.reward)
        self.greedy_policy = None


class Eq6Agent(Agent):
    def __init__(self, ns: int, na: int, discount_factor: float, alpha: float, estimate_var: bool = False):
        super().__init__(ns, na, discount_factor)
        self.q_function = np.zeros((self.ns, self.na))
        self.w_function = np.zeros_like(self.q_function)
        self.w_function2 = np.copy(self.w_function)
        self.alpha = alpha
        self.model = EmpiricalModel(self.ns, self.na)
        self.greedy_policy = None
        self.estimate_var = estimate_var
        
    def _forward_logic(self, state: int, step: int) -> int:
        if self.num_visits_state[state] > 2 * self.na:
            exp_policy = self.compute_explorative_policy(state)
        else:
            exp_policy = np.ones(self.na) / self.na
        action = np.random.choice(self.na, p = exp_policy)
        return action

    def greedy_action(self, state: int) -> int:
        if self.greedy_policy is None:
            V, pi, Q = policy_iteration(self.discount_factor, self.model.transition_function, self.model.reward)
            self.greedy_policy = pi
        return self.greedy_policy[state]
        # return self.q_function[state].argmax()

    def _backward_logic(self, experience: Experience):
        self.greedy_policy = None
        state, action, reward, next_state, done = list(experience)
        self.model.update_visits(state, action, next_state, reward)
        target = reward + (1-done) * self.discount_factor * self.q_function[next_state].max()
        lr = 1 / (self.num_visits_actions[state][action] ** self.alpha)
        self.q_function[state][action] += lr * (target - self.q_function[state][action])
          
        #Estimate W as a substitute for the variance
        # expq = np.max(self.q_function[next_state])**2
        # w_target = expq - ((self.q_function[state][action] - reward)/self.discount_factor)**2
        # self.w_function[state][action] += lr * (w_target - self.w_function[state][action])
        
        w_target = reward + (1- done)*self.discount_factor * self.q_function[next_state].max() - self.q_function[state][action]
        self.w_function[state][action] += lr * ((w_target / self.discount_factor) ** 2 - self.w_function[state][action])
        
        
    def compute_explorative_policy(self, state: int, tol = .2):
        if self.estimate_var is False:
            V, pi, Q = policy_iteration(self.discount_factor, self.model.transition_function, self.model.reward)
            avg_V = np.array([self.model.transition_function[state, a] @ V for a in range(self.na)])
            var_V = np.array([self.model.transition_function[state, a] @ ((V - avg_V[a]) ** 2) for a in range(self.na)])
        else:
            var_V = self.w_function[state]
            V = self.q_function.max(1)
            Q = self.q_function
            pi = self.q_function.argmax(1)
        
        delta = np.array([[Q[s, pi[s]] - Q[s, a] for a in range (self.na)] for s in range(self.ns)])
        delta_min = max(tol, np.min(delta))
        
        delta_sq = np.clip(delta[state], a_min=delta_min, a_max=None) ** 2
        rho = (1 + var_V) / delta_sq
        exp_policy = rho
        action_pi = pi[state]
        exp_policy[action_pi] = np.sqrt(rho[action_pi] * (rho.sum() - rho[action_pi]))
        
        exp_policy = exp_policy / exp_policy.sum()
        
        return exp_policy
        

class OnPolicyAgent(Agent):
    def __init__(self, ns: int, na: int, discount_factor: float, lr: float = 1e-2, hidden: int = 16, training_period: int = 64, alpha: float = 0.6):
        super().__init__(ns, na, discount_factor)
        self.q_function = np.zeros((self.ns, self.na))
        self.w_function = np.zeros_like(self.q_function)
        self.network = Network(ns, na, hidden, lr=lr)
        self.buffer = []
        self.training_period = training_period
        self.to_tensor = lambda x: torch.tensor(x, requires_grad=False, dtype=torch.float)
        self.pick_greedy_actions = lambda x: self.q_function[x].argmax()
        self.alpha = alpha
        
    @torch.no_grad()
    def _forward_logic(self, state: int, step: int) -> int:
        epsilon = 0.1
        policy = self.network(self.to_tensor([state])).detach().numpy()
        
        policy = epsilon * np.ones(self.na)/self.na + (1 - epsilon) * policy
        policy = policy / policy.sum()
        # print(policy)
        return np.random.choice(self.na, p=policy)

    def greedy_action(self, state: int) -> int:
        return self.q_function[state].argmax()
    
    
    def train_q_w(self, experience: Experience):
        state, action, reward, next_state, done = list(experience)
        target = reward + (1-done) * self.discount_factor * self.q_function[next_state].max()
        lr = 1 / (self.num_visits_actions[state][action] ** self.alpha)
        self.q_function[state][action] += lr * (target - self.q_function[state][action])
          
        #Estimate W as a substitute for the variance
        # expq = np.max(self.q_function[next_state])**2
        # w_target = expq - ((self.q_function[state][action] - reward)/self.discount_factor)**2
        # self.w_function[state][action] += lr * (w_target - self.w_function[state][action])
        w_target = reward + (1- done)*self.discount_factor * self.q_function[next_state].max() - self.q_function[state][action]
        self.w_function[state][action] += lr * ((w_target / self.discount_factor) ** 2 - self.w_function[state][action])

    def _backward_logic(self, experience: Experience, tol=0.2):
        self.buffer.append(tuple(experience))
        self.train_q_w(experience)

        if len(self.buffer) > self.training_period:
            
            states, actions, rewards, next_states, dones = zip(*self.buffer)
            
            mask = np.zeros((len(states), self.na), dtype=np.bool8)
            pi = self.q_function.argmax(1)
            for idx, s in enumerate(states):
                mask[idx, pi[s]] = True

            
            Q = self.q_function
            
            delta = np.array([[Q[s, pi[s]] - Q[s, a] for a in range (self.na)] for s in states])
            delta_min = max(tol, np.min(delta))
            delta_sq = np.clip(delta, a_min=delta_min, a_max=None) ** 2
            var_V = self.to_tensor([[self.w_function[s, a] for a in range (self.na)] for s in states])
            mask = self.to_tensor(mask).bool()

            with torch.no_grad():
                prold: torch.Tensor = self.network(self.to_tensor(states).unsqueeze(-1)).detach()

            for it in range(10):
                pr: torch.Tensor = self.network(self.to_tensor(states).unsqueeze(-1))
                rho = (1 + var_V) / (pr * self.to_tensor(delta_sq))
                
                H1 = rho[~mask].reshape(rho.shape[0], self.na-1)
                H2 = rho[mask] / ((1 - self.discount_factor)**3)
                
                # print(f'{torch.max(H1).item()} - {torch.max(H2).item()}')
                loss = torch.log(H1.max(-1)[0] + H2).mean() #+ torch.nn.KLDivLoss(reduction='batchmean')(pr, prold)
                # loss_detached= loss.detach()
                # probs = pr.gather(1, self.to_tensor(actions).long().unsqueeze(-1))
                total_loss = loss
                self.network.backward(total_loss)
            self.buffer = []
            
    