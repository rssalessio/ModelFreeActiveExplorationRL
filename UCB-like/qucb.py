import numpy as np


class QUCB(object):
    def __init__(self, gamma: float, ns: int, na: int, c: float = 2., p: float = 0.1):
        self.Q = np.ones((ns, na)) / (1 - gamma)
        self.gamma = gamma
        self.visits = np.zeros((ns, na))
        self.c = c
        self.ns = ns
        self.na = na
        self.p = p
        
    def forward(self, s: int):
        return self.Q[s].argmax()
    
    def backward(self, s: int, a: int, r: int, sp: int):
        self.visits[s, a] += 1
        
        T = self.visits.sum()
        t = self.visits[s,a]
        H = 1 / (1-self.gamma)
        alpha_t = (H + 1) / (H + t)
        target = r + self.gamma * self.V[sp]
        
        
        iota = np.log(self.ns * self.na * T / self.p)
        b_t = self.c * H * np.sqrt(iota / t)
        self.Q[s,a] = (1 - alpha_t) * self.Q[s,a] + alpha_t * (target + b_t)
        
    @property
    def V(self):
        return np.minimum(1/(1-self.gamma), self.Q.max(1))