import numpy as np
from .agent import Agent, Experience, AgentParameters
from typing import NamedTuple

class QLearningParameters(NamedTuple):
    pass

class QLearning(Agent):
    """ Classical Qlearning agent """

    def __init__(self, parameters: QLearningParameters, agent_parameters: AgentParameters):
        super().__init__(agent_parameters)
        self.Q = np.zeros((self.ns, self.na))
        self.parameters = parameters

    @staticmethod
    def suggested_exploration_parameter(dim_state: int, dim_action: int) -> float:
        return 10 / dim_state

    def forward(self, state: int, step: int) -> int:
        if np.random.uniform() < self.forced_exploration_callable(state, step):
            return np.random.choice(self.na)
        
        return self.Q[state].argmax()

    def process_experience(self, experience: Experience, step: int) -> None:
        #T = self.exp_visits[experience.s_t, experience.a_t].sum() ** self.parameters.learning_rate
        k = self.exp_visits[experience.s_t, experience.a_t].sum()
        
        H = 1 / (1-self.discount_factor)
        alpha_t = (H + 1) / (H + k)

        ## Update Q
        target = experience.r_t + self.discount_factor * self.Q[experience.s_tp1].max()
        self.Q[experience.s_t, experience.a_t] = (1 - alpha_t) * self.Q[experience.s_t, experience.a_t] + alpha_t * target
        
        self.greedy_policy = self.Q.argmax(1)
      