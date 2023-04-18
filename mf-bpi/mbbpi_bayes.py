import numpy as np
from new_mdp_description import MDPDescription2
from posterior import PosteriorProbabilisties

golden_ratio = (1 + np.sqrt(5)) / 2
golden_ratio_sq = golden_ratio ** 2

class MBBPIBayes(object):
    def __init__(self, gamma: float, ns: int, na: int, frequency_computation: int, nav_constr: bool = True):
        self.gamma = gamma
        self.visits = np.zeros((ns, na, ns))
        self.rewards = np.zeros((ns, na))
        
        self.ns = ns
        self.na = na
        self.frequency_computation = frequency_computation
        self.step = 0
        self.omega = np.ones((ns, na)) / (ns * na)
        self.nav_constr = nav_constr
        self.greedy_policy = [0 for s in range(ns)]
        self.posterior = PosteriorProbabilisties(self.ns, self.na)

    def forward(self, s: int, epsilon: float = 0.):
        omega = self.omega
        omega = omega[s] / omega[s].sum()
        return np.random.choice(self.na, p=omega)
    
    def backward(self, s: int, a: int, r: int, sp: int):
        self.posterior.update(s, a, sp, r)
        self.visits[s, a, sp] += 1
        n_sa = self.visits[s, a].sum()
        r_sa = self.rewards[s, a]
        self.rewards[s, a] = (n_sa - 1) * r_sa / n_sa + r / n_sa
        
        if self.step % self.frequency_computation == 0:
            P, R = self.posterior.sample_posterior()
            mdp = MDPDescription2(P, R, self.gamma, 1)
            self.omega = mdp.compute_allocation(navigation_constraints=self.nav_constr)[0]
            self.greedy_policy = mdp.pi_greedy
        self.step += 1
        
        
    