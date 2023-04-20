import numpy as np
from .agent import Agent, Experience, AgentParameters
from typing import NamedTuple
import sys
sys.path.append("../..")

from utils.simplified_new_mdp_description import SimplifiedNewMDPDescription
from utils.posterior import PosteriorProbabilisties

golden_ratio = (1 + np.sqrt(5)) / 2
golden_ratio_sq = golden_ratio ** 2

class BPIBayesParameters(NamedTuple):
    frequency_computation: int
    kbar: int

class BPIBayes(Agent):
    """ BPI Algorithm with posterior sampling of the MDP """

    def __init__(self, parameters: BPIBayesParameters, agent_parameters: AgentParameters):
        super().__init__(agent_parameters)
        self.parameters = parameters
        self.posterior = PosteriorProbabilisties(self.ns, self.na)
        self.rewards = np.zeros((self.ns, self.na), dtype=np.float64, order='C')

    def forward(self, state: int, step: int) -> int:
        omega = self.omega
        omega = omega[state] / omega[state].sum()
        return np.random.choice(self.na, p=omega)
    
    def process_experience(self, experience: Experience, step: int) -> None:
        s, a, r, sp = experience.s_t, experience.a_t, experience.r_t, experience.s_tp1
        self.posterior.update(s, a, sp, r)

        n_sa = self.exp_visits[s, a].sum()
        r_sa = self.rewards[s, a]
        self.rewards[s, a] = (n_sa - 1) * r_sa / n_sa + r / n_sa
        
        if step % self.parameters.frequency_computation == 0:
            P, R = self.posterior.sample_posterior()
            mdp = SimplifiedNewMDPDescription(P, R, self.discount_factor, self.parameters.kbar)
            self.omega = mdp.compute_allocation(navigation_constraints=True)[0]
            self.greedy_policy = mdp.pi_greedy.astype(np.int64)

        
        



    
