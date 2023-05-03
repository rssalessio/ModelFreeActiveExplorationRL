import numpy as np
from .agent import Agent, Experience, AgentParameters
from typing import NamedTuple, Optional
import sys
sys.path.append("../..")

from utils.mdp_description import MDPDescription
from utils.new_mdp_description import NewMDPDescription
from utils.simplified_new_mdp_description import SimplifiedNewMDPDescription
from utils.posterior import PosteriorProbabilisties
from enum import Enum

golden_ratio = (1 + np.sqrt(5)) / 2
golden_ratio_sq = golden_ratio ** 2

class BPIType(Enum):
    ORIGINAL_BOUND = 0
    NEW_BOUND = 1
    NEW_BOUND_SIMPLIFIED = 2

class BPIParameters(NamedTuple):
    frequency_computation_omega: int
    frequency_computation_greedy_policy: int
    kbar: Optional[int]
    enable_posterior_sampling: bool
    bpi_type: BPIType

class BPI(Agent):
    """ BPI Algorithm """

    def __init__(self, parameters: BPIParameters, agent_parameters: AgentParameters):
        super().__init__(agent_parameters)
        self.parameters = parameters
        self.posterior = PosteriorProbabilisties(self.ns, self.na)
        self.uniform_policy = np.ones((self.ns, self.na)) / (self.ns * self.na)
        self.frequency_computation_omega = self.parameters.frequency_computation_omega
        self.frequenty_computation_greedy_policy = self.parameters.frequency_computation_greedy_policy

    @staticmethod
    def suggested_exploration_parameter(dim_state: int, dim_action: int) -> float:
        return max(1, 1 / (2*dim_state))

    def forward(self, state: int, step: int) -> int:
        epsilon = self.forced_exploration_callable(state, step)
        omega = (1-epsilon) * self.omega + epsilon * self.uniform_policy
        omega = omega[state] / omega[state].sum()
        try:
            act=  np.random.choice(self.na, p=omega)
        except:
            import pdb
            pdb.set_trace()

        return act
    
    def get_mdp(self, force_mle: bool = False) -> MDPDescription:
        if force_mle or not self.parameters.enable_posterior_sampling:
            P, R = self.posterior.mle()
        else:
            P, R = self.posterior.sample_posterior()

        match self.parameters.bpi_type:
            case BPIType.ORIGINAL_BOUND:
                mdp = MDPDescription(P, R, self.discount_factor)
            case BPIType.NEW_BOUND:
                mdp = NewMDPDescription(P, R, self.discount_factor)
            case BPIType.NEW_BOUND_SIMPLIFIED:
                mdp = SimplifiedNewMDPDescription(P, R, self.discount_factor, self.parameters.kbar)
            case _:
                raise Exception(f'Type {self.parameters.bpi_type} not found.')
        return mdp
    
    def process_experience(self, experience: Experience, step: int) -> None:
        s, a, r, sp = experience.s_t, experience.a_t, experience.r_t, experience.s_tp1
        self.posterior.update(s, a, sp, r)
        mdp = None

        if step % self.frequenty_computation_greedy_policy == 0:
            mdp = self.get_mdp(force_mle=True)
            self.greedy_policy = mdp.pi_greedy.astype(np.int64)

        if step > self.horizon / 50 and step % self.frequency_computation_omega == 0:
            mdp = self.get_mdp()
            self.prev_omega = self.omega.copy()
            self.omega = mdp.compute_allocation(navigation_constraints=True)[0]

            # slope = max(self.parameters.frequency_computation_omega, 2000 * (step) / (self.horizon * 0.7))
            # self.frequency_computation_omega = min(2000, int(slope))
            slope = max(self.parameters.frequency_computation_omega, 5000 * (step) / (self.horizon * 0.5))
            self.frequency_computation_omega = min(5000, int(slope))


    
