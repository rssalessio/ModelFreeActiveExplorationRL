import numpy as np
from .agent import Agent, Experience, AgentParameters
from typing import NamedTuple
import sys
import cvxpy as cp
sys.path.append("../..")

from utils.simplified_new_mdp_description import SimplifiedNewMDPDescription

golden_ratio = (1 + np.sqrt(5)) / 2
golden_ratio_sq = golden_ratio ** 2

class BayesOBPIParameters(NamedTuple):
    frequency_computation: int
    kbar: int
    learning_rate_q: float
    learning_rate_m: float
    ensemble_size: int


class BayesOBPI(Agent):
    """ Bayes-O-BPI Algorithm """

    def __init__(self, parameters: BayesOBPIParameters, agent_parameters: AgentParameters):
        super().__init__(agent_parameters)
        self.parameters = parameters
        self.uniform_policy = np.ones((self.ns, self.na)) / (self.ns * self.na)
        self.Q = np.random.uniform(size=(self.parameters.ensemble_size, self.ns, self.na)) #/ (1 - self.discount_factor)
        self.M = np.random.uniform(size=(self.parameters.ensemble_size, self.ns, self.na)) #/ ((1 - self.discount_factor) ** (2 * self.parameters.kbar))
        
        self.prior_Q = 1e-2*np.random.uniform(size=(self.parameters.ensemble_size, self.ns, self.na)) #/ (1 - self.discount_factor)
        self.prior_M = 1e-2*np.random.uniform(size=(self.parameters.ensemble_size, self.ns, self.na)) #/ ((1 - self.discount_factor) ** (2 * self.parameters.kbar))
        
        
        self._current_head = 0     
        self._compute_Q = lambda s, head: self.Q[head, s] + self.prior_Q[head, s]
        self._compute_M = lambda s, head: self.M[head, s] + self.prior_M[head, s]
        self.alpha = np.ones(self.parameters.ensemble_size)
        self.beta = np.ones(self.parameters.ensemble_size)
        

    # def forward(self, state: int, step: int) -> int:
    #     epsilon = self.forced_exploration_callable(state, step)
    #     omega = (1-epsilon) * self.omega + epsilon * self.uniform_policy
    #     omega = omega[state] / omega[state].sum()
    #     return np.random.choice(self.na, p=omega)
    
    def forward(self, state: int, step: int) -> int:
        q_values = (self.Q + self.prior_Q)[self._current_head]
        m_values = (self.M + self.prior_M)[self._current_head] ** (2 ** (1 - self.parameters.kbar))
        greedy_policy = self.greedy_policy
        
        idxs_subopt_actions = np.array([
            [False if greedy_policy[s] == a else True for a in range(self.na)] for s in range(self.ns)]).astype(np.bool_)

        # Compute Delta
        delta = np.clip((q_values.max(-1, keepdims=True) - q_values) , a_min=1e-8, a_max=None)
        delta_subopt = delta[idxs_subopt_actions]
        delta_min = delta_subopt.min()
    
        delta[~idxs_subopt_actions] = delta_min * (1 - self.discount_factor) 
        H = (2 + 8 * golden_ratio_sq * m_values) / (delta ** 2)
        H[~idxs_subopt_actions] = np.sqrt(H[~idxs_subopt_actions] * H[~idxs_subopt_actions].sum() )
        p = (H/H.sum(-1, keepdims=True))
        
        epsilon = self.forced_exploration_callable(state, step)
        omega = (1-epsilon) * p + epsilon * np.ones((self.ns, self.na)) / ( self.na)
        if np.any(np.isnan(p)):
            import pdb
            pdb.set_trace()

        return np.random.choice(self.na, p=omega[state])
    
    def process_experience(self, experience: Experience, step: int) -> None:
        s, a, r, sp = experience.s_t, experience.a_t, experience.r_t, experience.s_tp1
        
        
        self.alpha[self._current_head] = max(1, self.alpha[self._current_head] + r)
        self.beta[self._current_head] = max(1, self.beta[self._current_head] + 1-r)
        
        T = self.exp_visits[s, a].sum()
        alpha_t = 1 / (1 + (T ** self.parameters.learning_rate_q))
        beta_t = 1 / (1 + (T ** self.parameters.learning_rate_m))

        ## Update Q
        for head in range(self.parameters.ensemble_size):
            if np.random.uniform() < 0.5: continue
            #head = self._current_head
            target = r + self.discount_factor * self._compute_Q(sp, head).max()
            self.Q[head, s,a] = (1 - alpha_t) * self._compute_Q(s,head)[a] + alpha_t * target
            
            ## Update V
            delta = (r + self.discount_factor * self._compute_Q(sp,head).max()- self._compute_Q(s, head)[a]) / self.discount_factor
            self.M[head, s,a] = self._compute_M(s, head)[a] + beta_t * (delta ** (2 * self.parameters.kbar)  - self._compute_M(s, head)[a])
                
        if step % int(1/(1-self.discount_factor)) == 0:            
            # self.omega = SimplifiedNewMDPDescription.compute_mf_allocation(
            #     self.discount_factor,
            #     (self.Q + self.prior_Q)[self._current_head], 
            #     (self.M + self.prior_M)[self._current_head] ** (2 ** (1 - self.parameters.kbar)),
            #     self.exp_visits, navigation_constraints=False
            # )
        
            self._current_head = np.random.beta(self.alpha, self.beta).argmax()

        self.greedy_policy = self.Q[self._current_head].argmax(1)
        
    def _choose_head(self):
        theta = np.random.beta(self.alpha, self.beta)
        I = theta.argmax()
        if np.random.uniform() < 1/2:
            for _ in range(100):
                J = np.random.beta(self.alpha, self.beta).argmax()
                if I != J:
                    I = J
                    break
        self._current_head = I
        return I
        
        
        


    
