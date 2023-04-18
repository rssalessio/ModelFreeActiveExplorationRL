import numpy as np
from new_mdp_description import MDPDescription2

golden_ratio = (1 + np.sqrt(5)) / 2
golden_ratio_sq = golden_ratio ** 2

class MFBPIUCB(object):
    def __init__(self, gamma: float, ns: int, na: int, eta1: float, eta2: float, num_k:int = 5):
        self.Q = np.zeros((ns, na))
        self.M = np.zeros((num_k, ns, na))
        self.gamma = gamma
        self.visits = np.zeros((ns, na, ns))
        self.ns = ns
        self.na = na
        self.step = 0
        self.omega = np.ones((ns, na)) / (ns * na)
        self.greedy_policy = [0 for _ in range(ns)]
        self.eta1 = eta1
        self.eta2 = eta2
        self.num_k = num_k

    def forward(self, s: int, epsilon: float = 0.):
        omega = (1-epsilon) * self.omega + epsilon * np.ones((self.ns, self.na)) / ( self.na)
        if np.any(np.isnan(omega)):
            import pdb
            pdb.set_trace()

        return np.random.choice(self.na, p=omega[s])
    
    def backward(self, s: int, a: int, r: int, sp: int):
        self.visits[s, a, sp] += 1
        T = self.visits[s, a].sum()
        alpha_t = 1 / (1 + (T ** self.eta1))
        beta_t = 1 / (1 + (T * self.eta2))
            
        H = 1 / (1-self.gamma)
        alpha_t = (H + 1) / (H + T)
        beta_t = alpha_t
        
        alpha_t = alpha_t ** self.eta1
        beta_t = alpha_t ** self.eta2
    
        iota = np.log(self.ns * self.na * self.visits.sum()/ 1e-2)
        b_t = 1e-2 * H * np.sqrt(iota / T)
        
        ## Update Q
        V = np.minimum(1/(1-self.gamma), self.Q.max(1))
        target = r + self.gamma * V[sp] + b_t
        self.Q[s,a] = (1 - alpha_t) * self.Q[s,a] + alpha_t * target
        
        
        ## Update M
        V = np.minimum(1/(1-self.gamma), self.Q.max(1))
        for i in range(self.num_k):
            delta = (r + self.gamma * V[sp] - self.Q[s,a]) / self.gamma
            self.M[i,s,a] = self.M[i,s,a] + beta_t * (delta ** (2 ** (i+1))  - self.M[i,s,a])
            
        
        self.step += 1
        self._update_omega()
    
    def _update_omega(self):
        self.greedy_policy = self.Q.argmax(1)
        idxs_subopt_actions = np.array([
            [False if self.greedy_policy[s] == a else True for a in range(self.na)] for s in range(self.ns)]).astype(np.bool_)

        # Compute Delta
        delta_sq = np.clip((self.Q.max(-1, keepdims=True) - self.Q) ** 2, a_min=1e-16, a_max=None)
        delta_sq_subopt = delta_sq[idxs_subopt_actions]
        delta_sq_min = delta_sq_subopt.min()
        
        delta_sq[~idxs_subopt_actions] = delta_sq_min * ((1 - self.gamma) ** 2)
        
        self.kvalues = [(self.M[k-1] ** ( 2 ** (-k))).max() for k in range(1, self.num_k+1)]
        self.kbar = np.argmax(self.kvalues)
        H = (2 + 8 * golden_ratio_sq * (self.M[self.kbar] ** (2. ** (- self.kbar)))) / delta_sq
        
        H[~idxs_subopt_actions] = np.sqrt(H[~idxs_subopt_actions] * H[~idxs_subopt_actions].sum() )
        self.omega = (H/H.sum(-1, keepdims=True))
        