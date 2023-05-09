import numpy as np
from numpy.typing import NDArray
from .agent import Agent, Experience, AgentParameters
from typing import NamedTuple

class QUCBParameters(NamedTuple):
    confidence: float
class QUCB(Agent):
    def __init__(self, parameters: QUCBParameters, agent_parameters: AgentParameters):
        super().__init__(agent_parameters)
        self.Q = np.ones((self.ns, self.na)) / (1 - self.discount_factor)
        self.parameters = parameters

    @staticmethod
    def suggested_exploration_parameter(dim_state: int, dim_action: int) -> float:
        return  1

    def forward(self, state: int, step: int) -> int:
        if np.random.uniform() < self.forced_exploration_callable(state, step, minimum_exploration=1e-3):
            return np.random.choice(self.na)
        
        return self.Q[state].argmax()

    def process_experience(self, experience: Experience, step: int) -> None:
        s, a, r, sp = experience.s_t, experience.a_t, experience.r_t, experience.s_tp1
        k = self.exp_visits[s, a].sum()
        
        H = 1 / (1-self.discount_factor)
        alpha_t = (H + 1) / (H + k)
        
        T = step + 1
        iota = np.log(self.ns * self.na * T / self.parameters.confidence)
        b_t =   1e-3* H * np.sqrt( iota / k)
        
        # if np.random.uniform() < 1e-3 and step >  20000:
        #     import pdb
        #     pdb.set_trace()
        ## Update Q
        target = r + self.discount_factor * self.V[sp] + b_t
        self.Q[s,a] = (1 - alpha_t) * self.Q[s,a] + alpha_t * target
        
        self.greedy_policy = (np.random.random(self.Q.shape) * (self.Q==self.Q.max(1, keepdims=True))).argmax(1)
      
    @property
    def V(self) -> NDArray[np.float64]:
        return np.minimum(1/(1-self.discount_factor), self.Q.max(1))