import numpy as np

golden_ratio = (1 + np.sqrt(5)) / 2
golden_ratio_sq = golden_ratio ** 2

class BPI(object):
    def __init__(self, gamma: float, ns: int, na: int, c: float = 2., p: float = 0.1):
        self.Q = np.ones((ns, na)) / (1 - gamma)
        self.M = np.ones((ns, na)) / ((1 - gamma) ** 2)
        self.gamma = gamma
        self.visits = np.zeros((ns, na))
        self.c = c
        self.ns = ns
        self.na = na
        self.p = p
        
    def forward(self, s: int):
        Q = np.minimum(1/(1-self.gamma), self.Q)
        pi_greedy = Q.argmax(1)
        delta_sq = np.clip((Q.max(1)[:, np.newaxis] - Q) ** 2, a_min=1e-9, a_max=None)
        idxs_subopt_actions = np.array([[False if pi_greedy[s] == a else True for a in range(self.na)] for s in range(self.ns)])
        delta_sq_subopt = delta_sq[idxs_subopt_actions]
        delta_sq_min =  delta_sq_subopt.min()
        
        VarV = self.VarV
        
        H = (2 + 8 * golden_ratio_sq * VarV[idxs_subopt_actions]) / delta_sq_subopt
        Hstar = (2 + 8 * golden_ratio_sq * VarV[~idxs_subopt_actions]) / (delta_sq_min * ((1 - self.gamma) ** 2))

        
        probs = np.ones((self.ns,self.na))
        probs[idxs_subopt_actions] = H
        probs[~idxs_subopt_actions] = np.sqrt(Hstar * np.sum(probs))
        
        probs = probs / probs.sum()

        return np.random.choice(self.na, p=probs[s] / probs[s].sum())
    
    def backward(self, s: int, a: int, r: int, sp: int):
        self.visits[s, a] += 1
        
        T = self.visits.sum()
        t = self.visits[s,a]
        H = 1 / (1-self.gamma)
        alpha_t = (H + 1) / (H + t)
        beta_t = (H + 1) / (H + t * np.log(t + 1))
        
        ## Update Q
        target = r + self.gamma * self.V[sp]
        iota = np.log(self.ns * self.na * T / self.p)
        b_t = self.c * H * np.sqrt(iota / t)
        self.Q[s,a] = (1 - alpha_t) * self.Q[s,a] + alpha_t * (target + b_t)
        
        ## Update V
        b_t = (2 * self.c) * ( H ** 2) * np.sqrt(iota / t)
        delta = (r + self.gamma * self.V[sp]- np.minimum(H, self.Q[s,a])) / self.gamma
        self.M[s,a] = self.M[s,a] + beta_t * (delta ** 2 + b_t - self.M[s,a])
        
    @property
    def V(self):
        return np.minimum(1/(1-self.gamma), self.Q.max(1))
    
    @property
    def VarV(self):
        return np.minimum(1 / ((1-self.gamma)**2), self.M)