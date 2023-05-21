from agents.agent import Agent, AgentParameters
from agents.qlearning import QLearning, QLearningParameters
from agents.qucb import QUCB, QUCBParameters
from agents.bpi import BPIParameters, BPI, BPIType
# from agents.pgobpi import PGOBPIParameters, PGOOBPI
from agents.obpi import OBPI, OBPIParameters
from agents.bayesobpi import BayesOBPI, BayesOBPIParameters
from agents.psrl import PSRL
from enum import Enum

class AgentType(Enum):
    Q_LEARNING = 'Q-Learning'
    Q_UCB = 'Q-UCB'
    BPI_NEW_BOUND_BAYES = 'BPI-Bayes'
    BPI_NEW_BOUND = 'BPI'
    MDP_NAS = 'MDP-NaS'
    BPI_NEW_BOUND_SIMPLIFIED_1 = 'BPISimplified - 1'
    OBPI = 'O-BPI'
    #PGOBPI = 'PGO-BPI'
    BAYESOBPI = 'Bayes-O-BPI'
    PSRL = 'PSRL'
    GENEARATIVE_OBPI = 'Generative-O-BPI'

FREQUENCY = 200
QUCB_PARAMETERS = QUCBParameters(confidence=1e-3)
QLEARNING_PARAMETERS = QLearningParameters()
BPI_NEW_BOUND_BAYES_PARAMETERS = BPIParameters(
    frequency_computation_greedy_policy=FREQUENCY, frequency_computation_omega=FREQUENCY, kbar=None, enable_posterior_sampling=True, bpi_type=BPIType.NEW_BOUND)
BPI_NEW_BOUND_PARAMETERS = BPIParameters(
    frequency_computation_greedy_policy=FREQUENCY, frequency_computation_omega=FREQUENCY, kbar=None, enable_posterior_sampling=False, bpi_type=BPIType.NEW_BOUND)
MDP_NAS_PARAMETERS = BPIParameters(
    frequency_computation_greedy_policy=FREQUENCY, frequency_computation_omega=FREQUENCY, kbar=None, enable_posterior_sampling=False, bpi_type=BPIType.ORIGINAL_BOUND)
BPI_NEW_BOUND_SIMPLIFIED_1_PARAMETERS = BPIParameters(
    frequency_computation_greedy_policy=FREQUENCY, frequency_computation_omega=FREQUENCY, kbar=1, enable_posterior_sampling=False, bpi_type=BPIType.NEW_BOUND_SIMPLIFIED)

OBPI_PARAMETERS = OBPIParameters(frequency_computation=FREQUENCY, kbar=1)
# PGOBI_PARAMETERS = PGOBPIParameters(frequency_computation=100, kbar=1, learning_rate_q=0.6, learning_rate_m=0.7, mixing_parameter=0.6)
BAYES_OBPI_PARAMETERS = BayesOBPIParameters(frequency_computation=FREQUENCY, kbar=1, ensemble_size=50)
GENERATIVE_OBPI_PARAMETERS = BayesOBPIParameters(frequency_computation=FREQUENCY, kbar=1, ensemble_size=1)


def make_agent(agent_name: AgentType, agent_parameters: AgentParameters) -> Agent:
    match agent_name:
        case AgentType.Q_LEARNING:
            return QLearning(QLEARNING_PARAMETERS, agent_parameters)
        case AgentType.Q_UCB:
            return QUCB(QUCB_PARAMETERS, agent_parameters)
        case AgentType.BPI_NEW_BOUND_BAYES:
            return BPI(BPI_NEW_BOUND_BAYES_PARAMETERS, agent_parameters)
        case AgentType.BPI_NEW_BOUND:
            return BPI(BPI_NEW_BOUND_PARAMETERS, agent_parameters)
        case AgentType.MDP_NAS:
            return BPI(MDP_NAS_PARAMETERS, agent_parameters)
        case AgentType.BPI_NEW_BOUND_SIMPLIFIED_1:
            return BPI(BPI_NEW_BOUND_SIMPLIFIED_1_PARAMETERS, agent_parameters)
        case AgentType.OBPI:
            return OBPI(OBPI_PARAMETERS, agent_parameters)
        # case AgentType.PGOBPI:
        #     return PGOOBPI(PGOBI_PARAMETERS, agent_parameters)
        case AgentType.BAYESOBPI:
            return BayesOBPI(BAYES_OBPI_PARAMETERS, agent_parameters)
        case AgentType.PSRL:
            return PSRL(agent_parameters)
        case AgentType.GENEARATIVE_OBPI:
            return BayesOBPI(GENERATIVE_OBPI_PARAMETERS, agent_parameters)
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
        