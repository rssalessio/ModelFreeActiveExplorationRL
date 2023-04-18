import numpy as np
from new_mdp_description import MDPDescription2
import cvxpy as cp

golden_ratio = (1 + np.sqrt(5)) / 2
golden_ratio_sq = golden_ratio ** 2

class MFBPIProjected(object):
    def __init__(self, gamma: float, ns: int, na: int, eta1: float, eta2: float, frequency_computation: int):
        self.Q = np.zeros((ns, na))
        self.M = np.zeros((ns, na))
        self.gamma = gamma
        self.visits = np.zeros((ns, na, ns))
        self.ns = ns
        self.na = na
        self.frequency_computation = frequency_computation
        self.step = 0
        self.omega = np.ones((ns, na)) / (ns * na)
        self.greedy_policy = [0 for s in range(ns)]
        self.eta1 = eta1
        self.eta2 = eta2
        self._flag_retry = False
        self._build_projection_problem()
    
    def _build_projection_problem(self):
        self._omega = cp.Variable((self.ns, self.na), nonneg=True)
        self._omega_gen = cp.Parameter((self.ns, self.na))
        self._transitions = [cp.Parameter((self.ns, self.na)) for s in range(self.ns)]
        self._sigma = cp.Variable(nonneg=True)
        
        
        self._objective = cp.sum(cp.rel_entr(self._omega, self._omega_gen))  #+ self._sigma #+ 20*cp.norm(self._omega) #cp.sum(cp.abs(cp.cumsum(cp.vec(self._omega - self._omega_gen)))) 
        self._constraints = [cp.sum(self._omega) == 1]
        for s in range(self.ns):
            self._constraints.append(
                cp.sum(self._omega[s]) == cp.sum(cp.multiply(self._omega, self._transitions[s]))
            )
        
        # kl_constraint = 0
        # omega_stack = cp.vstack([cp.sum(self._omega[s]) for s in range(self.ns)])
        
        # for s in range(self.ns):
        #     for a in range(self.na):
        #         Psa = cp.vstack([self._transitions[i][s,a] for i in range(self.ns)])
        #         kl_constraint += cp.sum(cp.rel_entr(omega_stack, Psa))
  
        # self._constraints.append(kl_constraint <= self._sigma)
                
        self._problem = cp.Problem(cp.Minimize(self._objective), self._constraints)
        self.solve_projection = lambda: self._problem.solve(warm_start=True, solver=cp.MOSEK, verbose=False)        
        

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
        
        if self.step % self.frequency_computation == 0 or self._flag_retry is True:
            self.omega_gen = MDPDescription2.compute_mf_allocation(
                self.gamma, self.Q, self.M, self.visits, navigation_constraints=False
            )
            
            eps = 0.15
            self._omega_gen.value = (1-eps) * self.omega_gen + eps * np.ones((self.ns, self.na)) / (self.ns * self.na)
            p_transitions = np.ones((self.ns, self.na, self.ns)) + self.visits
            P = p_transitions / p_transitions.sum(-1, keepdims=True)
        
            for s in range(self.ns):
                self._transitions[s].value = P[:, :, s]
                
            self._omega.value = np.ones((self.ns, self.na)) / (self.ns * self.na)
            #print(f'{self._omega_gen.value} - {self.omega_gen}')
            res = self.solve_projection()
            
            # if self.step % 500 == 0:
            #     print(self._omega.value)
            #     import pdb
            #     pdb.set_trace()
            omega = self._omega.value
            
            #print(f'{self.omega_gen} - {omega}')
            
            if omega is not None:
                self.omega = omega        
                self._flag_retry = False
            else:
                self._flag_retry = True
            
            
        self.greedy_policy = self.Q.argmax(1)
        self.step += 1
      