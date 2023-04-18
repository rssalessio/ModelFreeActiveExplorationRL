import numpy as np
from new_mdp_description import MDPDescription2


class QLearning(object):
    def __init__(self, gamma: float, ns: int, na: int, eta: float):
        self.Q = np.zeros((ns, na))
        self.gamma = gamma
        self.visits = np.zeros((ns, na, ns))
        self.ns = ns
        self.na = na
        self.step = 0
        self.eta = eta
        self.greedy_policy = [0 for s in range(ns)]
        self.omega = np.ones((ns, na))

    def forward(self, s: int, epsilon: float = 0.):
        if np.random.uniform() < epsilon:
            return np.random.choice(self.na)
        
        return self.Q[s].argmax()
    
    
    def backward(self, s: int, a: int, r: int, sp: int):
        self.visits[s, a, sp] += 1
        
        T = self.visits[s, a].sum() ** self.eta
        alpha_t = 1 / (1 + T)
        ## Update Q
        target = r + self.gamma * self.Q[sp].max()
        self.Q[s,a] = (1 - alpha_t) * self.Q[s,a] + alpha_t * target
        
        self.greedy_policy = self.Q.argmax(1)
        self.step += 1
      