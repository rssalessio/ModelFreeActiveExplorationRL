from agents.agent import Agent, AgentParameters
from agents.qlearning import QLearning, QLearningParameters
from agents.qucb import QUCB, QUCBParameters
from agents.mfbpi import MFBPIParameters, MFBPI
from agents.bpi import BPIParameters, BPI, BPIType
from agents.psrl import PSRL
from enum import Enum

class AgentType(Enum):
    Q_LEARNING = 'Q-Learning'
    Q_UCB = 'Q-UCB'
    BAYES_MFBPI = 'Bayes-MFBPI'
    FORCED_MFBPI = 'Forced-MFBPI'
    MDP_NAS = 'MDP-NaS'
    PSRL = 'PSRL'

QUCB_PARAMETERS = QUCBParameters(confidence=1e-3)
QLEARNING_PARAMETERS = QLearningParameters()
BAYES_MFBPI_PARAMETERS = MFBPIParameters(kbar=1, ensemble_size=50)
FORCED_MFBPI_PARAMETERS = MFBPIParameters(kbar=1, ensemble_size=1)
MDP_NAS_PARAMETERS = BPIParameters(
    frequency_computation_greedy_policy=200, frequency_computation_omega=200, kbar=None, enable_posterior_sampling=False, bpi_type=BPIType.ORIGINAL_BOUND)

def make_agent(agent_name: AgentType, agent_parameters: AgentParameters) -> Agent:
    match agent_name:
        case AgentType.Q_LEARNING:
            return QLearning(QLEARNING_PARAMETERS, agent_parameters)
        case AgentType.Q_UCB:
            return QUCB(QUCB_PARAMETERS, agent_parameters)
        case AgentType.BAYES_MFBPI:
            return MFBPI(BAYES_MFBPI_PARAMETERS, agent_parameters)
        case AgentType.FORCED_MFBPI:
            return MFBPI(FORCED_MFBPI_PARAMETERS, agent_parameters)
        case AgentType.PSRL:
            return PSRL(agent_parameters)
        case AgentType.MDP_NAS:
            return BPI(MDP_NAS_PARAMETERS, agent_parameters)
        case _:
            raise NotImplementedError(f'Type {agent_name.value} not found.')