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
    prior_q_scale: float
    prior_m_scale: float


class BayesOBPI(Agent):
    """ Bayes-O-BPI Algorithm """

    def __init__(self, parameters: BayesOBPIParameters, agent_parameters: AgentParameters):
        super().__init__(agent_parameters)
        self.parameters = parameters
        self.uniform_policy = np.ones((self.ns, self.na)) / (self.ns * self.na)
        self.Q = np.zeros((self.ns, self.na)) 
        self.M = np.zeros((self.ns, self.na)) 
        
        self.prior_Q = np.random.uniform(size=(self.ns, self.na)) 
        self.prior_M = np.random.uniform(size=(self.ns, self.na)) 

        
        self.combined_Q = lambda T: self.combined_value(self.Q, self.prior_Q, T)# self.Q + self.prior_Q * self.parameters.prior_q_scale *np.sqrt(np.log(self.ns * self.na * (T + 1))/(1+self.state_action_visits)) #/ (1 - self.discount_factor)
        self.combined_M = lambda T: self.M #np.clip(self.combined_value(self.M, self.prior_M, T,2 * self.parameters.kbar),0,np.inf)#self.M + self.prior_M * self.parameters.prior_m_scale*np.sqrt(np.log(self.ns * self.na * (T + 1))/(1+self.state_action_visits))#/ ((1 - self.discount_factor) ** (2 * self.parameters.kbar))

    def combined_value(self, X, priorX, T: int, power=1.):
        #H = 1 / (1-self.discount_factor) ** power
        confidence = 1e-3
        iota = np.log(self.ns * self.na * (T+1)/confidence )
        scale = 1e3* np.sqrt(iota / (1  + self.state_action_visits))
        return X + scale * priorX

    def forward(self, state: int, step: int) -> int:
        q_values = self.combined_Q(step)
        m_values = self.combined_M(step) ** (2 ** (1 - self.parameters.kbar))
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
        
       
        T = self.exp_visits[s, a].sum()
        alpha_t = 1 / (1 + (T ** self.parameters.learning_rate_q))
        beta_t = 1 / (1 + (T ** self.parameters.learning_rate_m))


        target = r + self.discount_factor * self.combined_Q(step)[sp].max()
        self.Q[s,a] = (1 - alpha_t) * self.Q[s,a] + alpha_t * target
        
        ## Update V
        delta = (r + self.discount_factor * self.Q[sp].max()- self.Q[s,a]) / self.discount_factor
        self.M[s,a] = (1 - beta_t) * self.M[s,a] + beta_t * (delta ** (2 * self.parameters.kbar))
        
        #if step % int(1/(1-self.discount_factor)):
        self.prior_Q = np.random.uniform(size=(self.ns, self.na)) 
            #self.prior_M = np.random.uniform(size=(self.ns, self.na)) 

        self.greedy_policy = self.combined_Q(step).argmax(1)
