import numpy as np
from new_mdp_description import MDPDescription2


class QUCB(object):
    def __init__(self, gamma: float, ns: int, na: int):
        self.Q = np.ones((ns, na)) / (1 - gamma)
        #self.Qhat = np.ones((ns, na)) / (1 - gamma)
        self.gamma = gamma
        self.visits = np.zeros((ns, na, ns))
        self.ns = ns
        self.na = na
        self.step = 0
        self.greedy_policy = [0 for s in range(ns)]
        self.omega = np.ones((ns, na))

    def forward(self, s: int, epsilon: float = 0.):        
        return self.Q[s].argmax()
    
    
    def backward(self, s: int, a: int, r: int, sp: int):
        self.visits[s, a, sp] += 1
        
        k = self.visits[s, a].sum()
        # c2 = 0.1# 4 * np.sqrt(2)
        # eps = 1e-2
        # R = np.ceil(np.log(3 / (eps * (1 - self.gamma))) / (1 - self.gamma))
        # M = 10
        # eps1 = eps / (24 * R * M * np.log(1 / (1 - self.gamma)))
        # H = np.log(1 / ((1 - self.gamma) * eps1)) / np.log(1 / self.gamma)
        # iota = np.log(self.ns * self.na * (k + 1) * (k + 2) / 1e-2)
        
        # alpha_t = (H + 1) / (H + k)
        # b_t = c2 * np.sqrt(H * iota / k) / (1 - self.gamma)
        
        
        H = 1 / (1-self.gamma)
        alpha_t = (H + 1) / (H + k)
        
        T = self.visits.sum()
        iota = np.log(self.ns * self.na * T / 1e-2)
        b_t = 1e-2 * H * np.sqrt(iota / k)
        
        ## Update Q
        target = r + self.gamma * self.V[sp] + b_t
        self.Q[s,a] = (1 - alpha_t) * self.Q[s,a] + alpha_t * target
        
        self.greedy_policy = self.Q.argmax(1)
        self.step += 1
      
    @property
    def V(self):
        return np.minimum(1/(1-self.gamma), self.Q.max(1))