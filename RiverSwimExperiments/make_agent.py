from agents.agent import Agent, AgentParameters
from agents.qlearning import QLearning, QLearningParameters
from agents.qucb import QUCB, QUCBParameters
from agents.bpi_bayes import BPIBayes, BPIBayesParameters
from agents.obpi import OBPI, OBPIParameters
from agents.mdp_nas import MDPNaS, MDPNaSParameters
from enum import Enum

class AgentType(Enum):
    Q_LEARNING = 'Q-Learning'
    Q_UCB = 'Q-UCB'
    BPI_BAYES = 'BPI-Bayes'
    OBPI = 'OBPI'
    MDP_NAS = 'MDP-NaS'

QUCB_PARAMETERS = QUCBParameters(confidence=1e-2)
QLEARNING_PARAMETERS = QLearningParameters(learning_rate=0.6)
BPI_BAYES_PARAMETERS = BPIBayesParameters(frequency_computation=100, kbar=1)
OBPI_PARAMETERS = OBPIParameters(frequency_computation=100, kbar=1)
MDP_NAS_PARAMETERS = MDPNaSParameters(frequency_computation=100)

def make_agent(agent_name: AgentType, agent_parameters: AgentParameters) -> Agent:
    match agent_name:
        case AgentType.Q_LEARNING:
            return QLearning(QLEARNING_PARAMETERS, agent_parameters)
        case AgentType.Q_UCB:
            return QUCB(QUCB_PARAMETERS, agent_parameters)
        case AgentType.BPI_BAYES:
            return BPIBayes(BPI_BAYES_PARAMETERS, agent_parameters)
        case AgentType.OBPI:
            return OBPI(OBPI_PARAMETERS, agent_parameters)
        case AgentType.MDP_NAS:
            return MDPNaS(MDP_NAS_PARAMETERS, agent_parameters)
        case _:
            raise NotImplementedError(f'Type {agent_name.value} not found.')


    # if agent_name == 'MFBPI':
    #     agent = MFBPI(p.gamma, env.ns, env.na, p.eta1, p.eta2, p.frequency_computation, True)
    # elif agent_name == 'MFBPI-GEN':
    #     agent = MFBPI(p.gamma, env.ns, env.na, p.eta1, p.eta2, p.frequency_computation, False)
    # elif agent_name == 'MBBPI':
    #     agent = MBBPI(p.gamma, env.ns, env.na, p.frequency_computation, True)
    # elif agent_name == 'QLEARNING':
    #     agent = QLearning(p.gamma, env.ns, env.na, p.eta1)
    # elif agent_name == 'QUCB':
    #     agent = QUCB(p.gamma, env.ns, env.na)
    # elif agent_name == 'MBBPIBayes':
    #     agent = MBBPIBayes(p.gamma, env.ns, env.na, p.frequency_computation, True)
    # elif agent_name == 'MFBPIProjected':
    #     agent = MFBPIProjected(p.gamma, env.ns, env.na, p.eta1, p.eta2, p.frequency_computation)
    # elif agent_name == 'OnPolicy':
    #     agent = OnPolicyAgent(p.gamma, env.ns, env.na, p.eta1, p.eta2, 16, lr=1e-2)
    # elif agent_name == 'MFBPIBootstrapped':
    #     agent = MFBPIBootstrapped(p.gamma, env.ns, env.na, p.eta1, p.eta2)
    # elif agent_name == 'MFBPIUCB':
    #     agent = MFBPIUCB(p.gamma, env.ns, env.na, p.eta1, p.eta2)
        