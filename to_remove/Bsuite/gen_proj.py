
import copy
from typing import Optional, Sequence

from bsuite.baselines import base
from bsuite.baselines.utils import replay
import numpy as np
from new_mdp_description import MDPDescription2
import cvxpy as cp
import dm_env
from dm_env import specs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


golden_ratio = (1+np.sqrt(5))/2
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
        self._step = 0
    
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
                
        self._problem = cp.Problem(cp.Minimize(self._objective), self._constraints)
        self.solve_projection = lambda: self._problem.solve(warm_start=True, solver=cp.MOSEK, verbose=False)        
        

    def select_action(self, timestep: dm_env.TimeStep):
        epsilon = max(0.1, 1/((1+self._step) **  0.2))
        #print(epsilon)
        s = timestep.observation.argmax()

        omega = (1-epsilon) * self.omega + epsilon * np.ones((self.ns, self.na)) / (self.ns * self.na)

        omega = omega[s] / omega[s].sum()
        self._step += 1
        
        return np.random.choice(self.na, p=omega)

    def update(self, timestep: dm_env.TimeStep,
      action: base.Action,
      new_timestep: dm_env.TimeStep):
        s=timestep.observation.argmax()
        a = action
        r = new_timestep.reward
        d = new_timestep.discount
        sp = new_timestep.observation.argmax()
        
        self.visits[s, a, sp] += 1
        
        T = self.visits[s, a].sum()
        alpha_t = 1 / (1 + (T ** self.eta1))
        beta_t = 1 / (1 + (T * self.eta2))
        
        ## Update Q
        target = r + self.gamma * self.Q[sp].max() * (1-d)
        self.Q[s,a] = (1 - alpha_t) * self.Q[s,a] + alpha_t * target
        
        ## Update V
        delta = (r + self.gamma * self.Q[sp].max()- self.Q[s,a]) / self.gamma
        self.M[s,a] = self.M[s,a] + beta_t * (delta ** 2  - self.M[s,a])
        
        if self.step % self.frequency_computation == 0 or self._flag_retry is True:
            # self.omega_gen = MDPDescription2.compute_mf_allocation(
            #     self.gamma, self.Q, self.M, self.visits, navigation_constraints=False
            # )
            
            pi_greedy = self.Q.argmax(1)
            Delta_sq = np.clip((self.Q.max(1)[:, np.newaxis] - self.Q) ** 2, a_min=1e-9, a_max=None)
            idxs_subopt_actions = np.array([[False if pi_greedy[s] == a else True for a in range(self.na)] for s in range(self.ns)])
            Delta_sq_subopt = Delta_sq[idxs_subopt_actions]
            Delta_sq_min =  Delta_sq_subopt.min()
            
            rho = np.zeros((self.ns, self.na))
            rho[idxs_subopt_actions] = (2+8 * golden_ratio * self.M[idxs_subopt_actions]) / Delta_sq_subopt
            Hstar = (2+8 * golden_ratio * self.M[~idxs_subopt_actions].max()) / (Delta_sq_min * ((1-self.gamma) ** 2))
            rho[~idxs_subopt_actions] = np.sqrt(Hstar * np.sum(rho[idxs_subopt_actions]))
            self.omega_gen = rho / rho.sum(-1, keepdims=True)
            
            eps = 0.3
            self._omega_gen.value = (1-eps) * self.omega_gen + eps * np.ones((self.ns, self.na)) / (self.ns * self.na)
            p_transitions = np.ones((self.ns, self.na, self.ns)) + self.visits
            P = p_transitions / p_transitions.sum(-1, keepdims=True)
        
            #print(self.visits.sum(-1))
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
            #print((self.omega / self.omega.sum(-1, keepdims=True))[:,-1].reshape(10, 10))
            #print(f'{self.omega_gen} - {omega}')
            
            if omega is not None:
                self.omega = omega        
                self._flag_retry = False
            else:
                self._flag_retry = True
            
            
        self.greedy_policy = self.Q.argmax(1)
        self.step += 1
      
      
      

def default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray):
  """Initialize a DQN agent with default parameters."""
  #del obs_spec  # Unused.
  
  size_s = np.prod(obs_spec.shape)
  import pdb
  pdb.set_trace()
  return MFBPIProjected(0.99, size_s, action_spec.num_values, 0.5, 0.6, 25)