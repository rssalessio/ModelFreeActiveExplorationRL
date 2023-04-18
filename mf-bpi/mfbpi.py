import numpy as np
from new_mdp_description import MDPDescription2

golden_ratio = (1 + np.sqrt(5)) / 2
golden_ratio_sq = golden_ratio ** 2

class MFBPI(object):
    def __init__(self, gamma: float, ns: int, na: int, eta1: float, eta2: float, frequency_computation: int, nav_constr: bool = True):
        self.Q = np.zeros((ns, na))
        self.M = np.zeros((ns, na))
        self.gamma = gamma
        self.visits = np.zeros((ns, na, ns))
        self.ns = ns
        self.na = na
        self.frequency_computation = frequency_computation
        self.step = 0
        self.omega = np.ones((ns, na)) / (ns * na)
        self.nav_constr = nav_constr
        self.greedy_policy = [0 for s in range(ns)]
        self.eta1 = eta1
        self.eta2 = eta2

    def forward(self, s: int, epsilon: float = 0.):
        omega = (1-epsilon) * self.omega + epsilon * np.ones((self.ns, self.na)) / (self.ns * self.na)
        omega = omega[s] / omega[s].sum()
        return np.random.choice(self.na, p=omega)
    
    def backward(self, s: int, a: int, r: int, sp: int):
        self.visits[s, a, sp] += 1
        
        T = self.visits[s, a].sum()
        alpha_t = 1 / (1 + (T ** self.eta1))
        beta_t = 1 / (1 + (T * self.eta2))
        
        ## Update Q
        target = r + self.gamma * self.Q[sp].max()
        self.Q[s,a] = (1 - alpha_t) * self.Q[s,a] + alpha_t * target
        
        ## Update V
        delta = (r + self.gamma * self.Q[sp].max()- self.Q[s,a]) / self.gamma
        self.M[s,a] = self.M[s,a] + beta_t * (delta ** 2  - self.M[s,a])
        
        if self.step % self.frequency_computation == 0:
            self.omega = MDPDescription2.compute_mf_allocation(
                self.gamma, self.Q, self.M, self.visits, navigation_constraints=self.nav_constr
            )
        self.greedy_policy = self.Q.argmax(1)
        self.step += 1
      