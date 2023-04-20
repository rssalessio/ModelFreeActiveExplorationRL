import numpy as np
from .agent import Agent, Experience, AgentParameters
from typing import NamedTuple
import sys
sys.path.append("../..")

from utils.mdp_description import MDPDescription

golden_ratio = (1 + np.sqrt(5)) / 2
golden_ratio_sq = golden_ratio ** 2

class MDPNaSParameters(NamedTuple):
    frequency_computation: int

class MDPNaS(Agent):
    """ MDPNaS Algorithm """

    def __init__(self, parameters: MDPNaSParameters, agent_parameters: AgentParameters):
        super().__init__(agent_parameters)
        self.parameters = parameters
        self.rewards = np.zeros((self.ns, self.na), dtype=np.float64, order='C')
        self.uniform_policy = np.ones((self.ns, self.na)) / (self.ns * self.na)

    def forward(self, state: int, step: int) -> int:
        epsilon = self.forced_exploration_callable(state, step)
        omega = (1-epsilon) * self.omega + epsilon * self.uniform_policy
        omega = omega[state] / omega[state].sum()
        return np.random.choice(self.na, p=omega)
    
    def process_experience(self, experience: Experience, step: int) -> None:
        s, a, r, sp = experience.s_t, experience.a_t, experience.r_t, experience.s_tp1
        
        n_sa = self.exp_visits[s, a].sum()
        r_sa = self.rewards[s, a]
        self.rewards[s, a] = (n_sa - 1) * r_sa / n_sa + r / n_sa
        
        if step % self.parameters.frequency_computation == 0:
            p_transitions = np.ones((self.ns, self.na, self.ns)) + self.exp_visits
            P = p_transitions / p_transitions.sum(-1, keepdims=True)
            R = self.rewards[..., np.newaxis]
            mdp = MDPDescription(P, R, self.discount_factor, self.parameters)
            self.omega = mdp.compute_allocation(navigation_constraints=True)[0]
            self.greedy_policy = mdp.pi_greedy

        
        



    