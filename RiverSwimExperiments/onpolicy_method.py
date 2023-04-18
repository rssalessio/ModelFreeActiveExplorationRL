import numpy as np
from new_mdp_description import MDPDescription2
import cvxpy as cp
import torch


golden_ratio = (1 + np.sqrt(5)) / 2
golden_ratio_sq = golden_ratio ** 2

class OnPolicyAgent(object):
    def __init__(self, gamma: float, ns: int, na: int, eta1: float, eta2: float, batch_size: int, lr: float = 1e-2):
        self.Q = np.ones((ns, na)) / (1-gamma)
        self.Qgreedy= np.zeros((ns, na)) / (1-gamma)
        self.M = np.ones((ns, na)) / ((1-gamma)**2)
        self.gamma = gamma
        self.visits = np.zeros((ns, na, ns))
        self.ns = ns
        self.na = na
        self.batch_size = batch_size
        self.step = 0
        self.omega = torch.tensor(np.ones((ns, na)) / na, dtype=torch.float32, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.omega], lr=lr)
        
        self.greedy_policy = [0 for s in range(ns)]
        self.eta1 = eta1
        self.eta2 = eta2
        
        self.batch = []

    def forward(self, s: int, epsilon: float = 0.):    
        omega = self.omega.clone().detach().numpy()
        omega = omega[s] / omega[s].sum()
        return np.random.choice(self.na, p=omega)
    
    def backward(self, s: int, a: int, r: int, sp: int):
        self.batch.append((s, a, r, sp))
        self.visits[s, a, sp] += 1
        
        T = self.visits[s, a].sum()
        alpha_t = 1 / (1 + (T ** self.eta1))
        beta_t = 1 / (1 + (T * self.eta2))
        
        ## Update Q
        target = r + self.gamma * self.Q[sp].max()
        target2 = r + self.gamma * self.Qgreedy[sp].max()
        self.Q[s,a] = (1 - alpha_t) * self.Q[s,a] + alpha_t * target
        self.Qgreedy[s,a] = (1-alpha_t) * self.Qgreedy[s,a] + alpha_t * target2
         
        ## Update V
        delta = (r + self.gamma * self.Q[sp].max()- self.Q[s,a]) / self.gamma
        self.M[s,a] = self.M[s,a] + beta_t * (delta ** 2  - self.M[s,a])
        
                 
            
        self.greedy_policy = self.Qgreedy.argmax(1)
        self.step += 1
        
        self.train_policy()  
        
    def train_policy(self):
        if len(self.batch) < self.batch_size:
            return
        
        pi_greedy = self.Q.argmax(1)
        Delta_sq = np.clip((self.Q.max(1)[:, np.newaxis] - self.Q) ** 2, a_min=1e-9, a_max=None)
        idxs_subopt_actions = np.array([[False if pi_greedy[s] == a else True for a in range(self.na)] for s in range(self.ns)])
        Delta_sq_subopt = Delta_sq[idxs_subopt_actions]
        Delta_sq_min =  Delta_sq_subopt.min()
        
        Hstar = (2 + 8 * golden_ratio_sq * self.M[~idxs_subopt_actions].max()) / (((1 - self.gamma) ** 2) * Delta_sq_min )
        Hsa = (2 + 8 * golden_ratio_sq * self.M[idxs_subopt_actions]) / (((1 - self.gamma) ** 2) * Delta_sq_subopt )
        Hsa = Hsa.reshape(self.ns, self.na-1)
        
        Hpolicy = np.zeros((self.ns, self.na))
        normalizer = Delta_sq_min * ((1 - self.gamma) ** 3) / (self.ns * self.na)
        for s in range(self.ns):
            for a in range(self.na):
                if np.isclose(self.Q[s].max(), self.Q[s,a]):
                    Hpolicy[s,a] = Hstar * normalizer
                else:
                    Hpolicy[s,a] = normalizer * (2 + 8 * golden_ratio_sq * self.M[s,a]) / ( Delta_sq[s,a] )
        
 
        Hpolicy = np.clip(Hpolicy, Hpolicy.max() / 1e2, Hpolicy.max())

        Hpolicy =  np.exp((Hpolicy - Hpolicy.max()))
        Hpolicy = Hpolicy / Hpolicy.sum(-1, keepdims=True)
        
        Hpolicy = 0.01 * np.ones((self.ns, self.na)) / self.na  + 0.99 * Hpolicy
                
        values = [0]
        losses = []
        
        
        self.omega.requires_grad_()
        for t in reversed(range(len(self.batch))):
            s, a, _, _ = self.batch[t]
            # if a == pi_greedy[s]:
            #     rt = Hstar * normalizer
            # else:
            #     rt =  normalizer * (2 + 8 * golden_ratio_sq * self.M[s,a]) / ( Delta_sq[s,a] )
            r_t = (1-self.gamma) * torch.log(torch.clamp(self.omega[s,a] / Hpolicy[s,a], 1e-6, 1e15))
            Vt = self.gamma * values[-1]  + r_t
            values.append(Vt)
            
            Lt = torch.log(self.omega[s,a]) * Vt + r_t
            losses.append(Lt)
        loss = torch.mean(torch.stack(losses))
        # print(loss)
        # import pdb
        # pdb.set_trace()
        # print(self.omega)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self.omega, 1.)
        self.optimizer.step()
        
        with torch.no_grad():
            omega = (self.omega.clone() / 0.3).exp()
            #print(self.omega)
            omega /= omega.sum(-1, keepdims=True)
            self.omega[:] = omega
            
            
        # print(self.visits.sum(-1).sum(-1))
        # print(self.omega)
        # print(self.Qgreedy.argmax(1))
        # print('---------------------------')
            
        self.batch = []
        
        
        
      