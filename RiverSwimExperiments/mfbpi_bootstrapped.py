import numpy as np
from new_mdp_description import MDPDescription2

golden_ratio = (1 + np.sqrt(5)) / 2
golden_ratio_sq = golden_ratio ** 2

class MFBPIBootstrapped(object):
    def __init__(self, gamma: float, ns: int, na: int, eta1: float, eta2: float, ensemble_size: int = 10):
        self.Q = np.random.normal(size=(ensemble_size, ns, na))
        self.M = np.random.uniform(size=(ensemble_size, ns, na))
        self.gamma = gamma
        self.visits = np.zeros((ns, na, ns))
        self.ns = ns
        self.na = na
        self.step = 0
        self.omega = np.ones((ns, na)) / (ns * na)
        self.greedy_policy = [0 for _ in range(ns)]
        self.eta1 = eta1
        self.eta2 = eta2
        self.ensemble_size = ensemble_size
        self.active_head = np.random.randint(0, self.ensemble_size)
        self.update_visits = np.zeros((self.ensemble_size, ns, na, ns))

    def forward(self, s: int, epsilon: float = 0.):
        # import pdb
        # pdb.set_trace()
        q_values = np.random.normal(self.Q.mean(0).flatten(), self.Q.std(0).flatten()).reshape(self.ns, self.na)
        m_values = np.clip(np.random.normal(self.M.mean(0).flatten(), self.M.std(0).flatten()).reshape(self.ns, self.na), 0, np.inf)
        # q_values = self.Q[self.active_head]
        # m_values = self.M[self.active_head]
        greedy_policy = q_values.argmax(1)
        # import pdb
        # pdb.set_trace()
        idxs_subopt_actions = np.array([
            [False if greedy_policy[s] == a else True for a in range(self.na)] for s in range(self.ns)]).astype(np.bool_)

        # Compute Delta
        delta_sq = np.clip((q_values.max(-1, keepdims=True) - q_values) ** 2, a_min=1e-8, a_max=None)
        delta_sq_subopt = delta_sq[idxs_subopt_actions]
        delta_sq_min = delta_sq_subopt.min()
        
        
        delta_sq[~idxs_subopt_actions] = delta_sq_min * ((1 - self.gamma) ** 2)
        
        # idxs = q_values.argmax(-1)
        # delta = q_values.max(-1, keepdims=True) - q_values
        # delta[idxs] = self._delta_min * (1 - self._discount)
        H = (2 + 8 * golden_ratio_sq * m_values) / delta_sq
        
        # if len(H[~idxs].shape) == 1:
        #     H[idxs] = np.sqrt(H[idxs] * H[~idxs] )
        # else:
        H[~idxs_subopt_actions] = np.sqrt(H[~idxs_subopt_actions] * H[~idxs_subopt_actions].sum() )
        p = (H/H.sum(-1, keepdims=True))#.mean(0)
        
        omega = (1-epsilon) * p + epsilon * np.ones((self.ns, self.na)) / ( self.na)
        if np.any(np.isnan(p)):
            import pdb
            pdb.set_trace()

        return np.random.choice(self.na, p=omega[s])
    
    def backward(self, s: int, a: int, r: int, sp: int):
        self.visits[s, a, sp] += 1
        
        
        ## Update Q
        
        for i in range(self.ensemble_size):
            #if i == self.active_head: continue
            #if  np.random.uniform() < 0.5: continue
            
            self.update_visits[i, s, a, sp] += 1
            T = self.update_visits[i, s, a].sum()
            alpha_t = 1 / (1 + (T ** self.eta1))
            beta_t = 1 / (1 + (T * self.eta2))
        
        
            target = r + self.gamma * self.Q[i,sp].max()
            self.Q[i,s,a] = (1 - alpha_t) * self.Q[i,s,a] + alpha_t * target
            
            ## Update V
            delta = (r + self.gamma * self.Q[i,sp].max()- self.Q[i,s,a]) / self.gamma
            self.M[i,s,a] = self.M[i,s,a] + beta_t * (delta ** 2  - self.M[i,s,a])
            
            self.greedy_policy = self.Q[self.active_head].argmax(1)
            self.step += 1
        
        #if np.random.uniform() < 1 - self.gamma:
        self.active_head = np.random.randint(0, self.ensemble_size)
      