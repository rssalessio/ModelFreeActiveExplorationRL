import numpy as np
from .agent import Agent, Experience, AgentParameters
from typing import NamedTuple
import sys
import cvxpy as cp
sys.path.append("../..")
from scipy.stats import t
from utils.simplified_new_mdp_description import SimplifiedNewMDPDescription

golden_ratio = (1 + np.sqrt(5)) / 2
golden_ratio_sq = golden_ratio ** 2

class BayesOBPIParameters(NamedTuple):
    frequency_computation: int
    kbar: int
    confidence: float


class BayesOBPI(Agent):
    """ Bayes-O-BPI Algorithm """

    def __init__(self, parameters: BayesOBPIParameters, agent_parameters: AgentParameters):
        super().__init__(agent_parameters)
        self.parameters = parameters
        self.ensemble_size = 40#+ 2
        self.uniform_policy = np.ones((self.ns, self.na)) / (self.ns * self.na)
        self.Q = np.tile(np.linspace(0, 1, self.ensemble_size)[:,None,None], (1, self.ns, self.na))  / (1 - self.discount_factor)
        self.M = np.tile(np.linspace(0, 1, self.ensemble_size)[:,None,None], (1, self.ns, self.na)) / ((1 - self.discount_factor) ** (2*self.parameters.kbar))
        
        
        self.Q = self.Q.flatten()
        self.M = self.M.flatten()

        np.random.shuffle(self.Q)
        np.random.shuffle(self.M)
        self.Q = self.Q.reshape(self.ensemble_size, self.ns, self.na)
        self.M = self.M.reshape(self.ensemble_size, self.ns, self.na)

        #np.tile(np.linspace(0, 1, 10), (1, 5, 5))
        
        # import pdb
        # pdb.set_trace()
        # H = 1 / (1 - self.discount_factor)
        # self.Q = np.vstack([
        #     np.random.uniform(size=(self.ensemble_size, self.ns, self.na), low=0, high = H),
        #     np.zeros((1, self.ns, self.na)),
        #     np.ones((1, self.ns, self.na)) * H,
        # ])
        # import matplotlib.pyplot as plt
        # plt.hist(self.Q[:,0,0])
        # plt.show()
        # import matplotlib.pyplot as plt
        
        # H = 1 / (1 - self.discount_factor)** (2*self.parameters.kbar)
        # self.M = np.vstack([
        #     np.random.uniform(size=(self.ensemble_size, self.ns, self.na), low=0, high = H),
        #     np.zeros((1, self.ns, self.na)),
        #     np.ones((1, self.ns, self.na)) * H,
        # ])
        # self.ensemble_size = self.ensemble_size + 2
        
        self.omega = np.ones(shape=(self.ns, self.na)) / (self.ns * self.na)
        self.policy = np.ones(shape=(self.ns, self.na)) / (self.na)
        # self.Q = np.vstack((
        #     np.random.uniform(size=(self.ensemble_size-2, self.ns, self.na)),
        #     np.zeros((1, self.ns, self.na)),
        #     np.ones((1, self.ns, self.na))
        #     )) / (1 - self.discount_factor)
        # self.M = np.vstack((
        #     np.random.uniform(size=(self.ensemble_size-2, self.ns, self.na)),
        #     np.zeros((1, self.ns, self.na)),
        #     np.ones((1, self.ns, self.na))
        #     )) / ((1 - self.discount_factor) ** (2*self.parameters.kbar))
        # self.Q = np.ones((self.ensemble_size, self.ns, self.na))  / (1 - self.discount_factor)
        # self.M = np.ones((self.ensemble_size, self.ns, self.na)) / ((1 - self.discount_factor) ** (2*self.parameters.kbar))
    
        
        
        
        
        self._visits = np.zeros((self.ensemble_size, self.ns, self.na, self.ns))
        self.combined_Q = lambda: self.combined_value(self.Q)# self.Q + self.prior_Q * self.parameters.prior_q_scale *np.sqrt(np.log(self.ns * self.na * (T + 1))/(1+self.state_action_visits)) #/ (1 - self.discount_factor)
        self.combined_M = lambda: self.combined_value(self.M, 2 * self.parameters.kbar)#self.M + self.prior_M * self.parameters.prior_m_scale*np.sqrt(np.log(self.ns * self.na * (T + 1))/(1+self.state_action_visits))#/ ((1 - self.discount_factor) ** (2 * self.parameters.kbar))
        self.c = 0.99
        self.t_val = t.ppf(self.c + (1-self.c)/2, self.ensemble_size)
    
    @staticmethod
    def suggested_exploration_parameter(dim_state: int, dim_action: int) -> float:
        return 1#max(1, 1 / (2*dim_state))

    def combined_value(self, X,  power=1.):
        H = 1 / (1-self.discount_factor) ** power
        # if np.random.uniform() < 1e-3:
        #     import pdb
        #     pdb.set_trace()
        # import pdb
        # pdb.set_trace()
        std = X.std(0, ddof=1)
        mu = X.mean(0)
        ce = self.t_val * std / np.sqrt(self.ensemble_size)
        #return np.clip(mu + ce* np.random.normal(size=(self.ns, self.na)), 0, H)
        Y = np.random.uniform(low=mu-ce, high=mu+ce)
    
        #Y = np.quantile(X, np.random.uniform(size=(self.ns*self.na))).reshape(self.ns, self.na)
        #Y = np.random.uniform(low=X.min(0), high=X.max(0))
        #Y = X.mean(0) + (H / np.sqrt())
        return np.clip(Y, 0, H)
        
    # s = np.std(x, axis=0, ddof=1)
    # return x.mean(0), c * s/ np.sqrt(N)
        
    #     return np.clip(np.random.uniform(low=X.min(0), high=X.max(0)), 0, H)
        
        
        # confidence = self.parameters.confidence
        # iota = np.log(self.ns * self.na * (T+1)/confidence )
        # scale = confidence*H* np.sqrt(iota / (1  + self.state_action_visits))

        # scale2 =  1* 0.5 * H / (1  + self.state_action_visits)
        
        #ce = scale2 * np.sqrt(2 * np.log(2/confidence) / (1  + self.state_action_visits))
        #low, high = np.maximum(0, X-ce), np.minimum(H, X+ce)
        #import pdb
        #pdb.set_trace()
        #Y = np.random.uniform(low=low, high=high)

        #print(f'{np.linalg.norm(scale)} - {np.linalg.norm(scale2)}')
        #return np.random.uniform(low=low, high=high) #
        #return np.clip(X + scale2 * np.random.normal(size=(self.ns, self.na)), 0, H)

    def forward(self, state: int, step: int) -> int:
        epsilon = self.forced_exploration_callable(state, step)
        omega = (1-epsilon) * self.policy[state] + epsilon * np.ones(( self.na)) / ( self.na)
        return np.random.choice(self.na, p=omega)
    
    def compute_omega(self):
        q_values = self.Q[self._head]# self.combined_Q()
        m_values = self.M[self._head]#self.combined_M() ** (2 ** (1 - self.parameters.kbar))
        greedy_policy = q_values.argmax(1)
        
        idxs_subopt_actions = np.array([
            [False if greedy_policy[s] == a else True for a in range(self.na)] for s in range(self.ns)]).astype(np.bool_)

        # Compute Delta
        delta = np.clip((q_values.max(-1, keepdims=True) - q_values) , a_min=1e-8, a_max=None)
        delta_subopt = delta[idxs_subopt_actions]
        delta_min = delta_subopt.min()
    
        delta[~idxs_subopt_actions] = delta_min * (1 - self.discount_factor) / (1 + self.discount_factor)
       
        Hsa = (2 + 8 * golden_ratio_sq * m_values) / (delta ** 2)

        C = np.max(np.maximum(4, 16 * (self.discount_factor ** 2) * golden_ratio_sq * m_values[~idxs_subopt_actions]))

        Hopt = C / (delta[~idxs_subopt_actions] ** 2)
        

        Hsa[~idxs_subopt_actions] = np.sqrt(Hopt * Hsa[idxs_subopt_actions].sum() )
  
        self.omega = Hsa/Hsa.sum()
        self.policy = self.omega / self.omega.sum(-1, keepdims=True)

    
    def process_experience(self, experience: Experience, step: int) -> None:
        s, a, r, sp = experience.s_t, experience.a_t, experience.r_t, experience.s_tp1
        
       
        # T = self.exp_visits[s, a].sum()
        # alpha_t = 1 / (1 + (T ** self.parameters.learning_rate_q))
        # beta_t = 1 / (1 + (T ** self.parameters.learning_rate_m))

        
        #current_head = self.current_head
        qmax = self.Q[:,sp].max()
        
        idxs = np.random.choice(self.ensemble_size, size= int(self.ensemble_size), replace=False)
        #for current_head in :
            # if np.random.uniform() < 0.5: continue
        
        self._visits[idxs, s, a, sp] += 1
        k = self._visits[idxs, s, a].sum(axis=1)
        H = 1 / (1-self.discount_factor)
        alpha_t = (H + 1) / (H + k)
        


        beta_t = alpha_t ** 1.1
        
        target = r + self.discount_factor * self.Q[idxs,sp].max(-1)
        self.Q[idxs,s,a] = (1 - alpha_t) * self.Q[idxs,s,a] + alpha_t * target
        
        ## Update V
        delta = (r + self.discount_factor * self.Q[idxs,sp].max(-1) - self.Q[idxs,s,a]) / self.discount_factor
        self.M[idxs,s,a] = (1 - beta_t) * self.M[idxs,s,a] + beta_t * (delta ** (2 * self.parameters.kbar))
    
        Qmu = self.Q.mean(0)    
        self.greedy_policy = (np.random.random(Qmu.shape) * (Qmu==Qmu.max(1, keepdims=True))).argmax(1)

        #if step % int(1/(1-self.discount_factor)) == 0:
        self._head = np.random.choice(self.ensemble_size)
        
        #if step % 10000 == 0:
            # import matplotlib.pyplot as plt
            # plt.hist(self.Q[:,0,0])
            # plt.show()
            # import pdb
            # pdb.set_trace()
        self.compute_omega()