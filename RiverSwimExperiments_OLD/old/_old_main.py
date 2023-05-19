from envs.riverswim import RiverSwim
from envs.forked_riverswim import ForkedRiverSwim
import numpy as np
from tqdm import tqdm
from mbbpi import MBBPI
from mfbpi import MFBPI
from mbbpi_bayes import MBBPIBayes
from mfbpi_projected import MFBPIProjected
from mfbpi_bootstrapped import MFBPIBootstrapped
from mfbpi_ucb import MFBPIUCB
import multiprocessing as mp
import pickle
from RiverSwimExperiments.agents.qlearning import QLearning
from qucb import QUCB
from onpolicy_method import OnPolicyAgent
from RiverSwimExperiments.utils.utils import policy_evaluation
import copy
from parameters import Parameters
import lzma

def run(agent_name: str, p: Parameters):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    if p.env_type == 'RiverSwim':
        env = RiverSwim(num_states=p.river_length, min_reward=p.min_reward, max_reward=p.max_reward_1)
    elif p.env_type == 'ForkedRiverSwim':
        env = ForkedRiverSwim(river_length=p.river_length, min_reward=p.min_reward, max_reward_river1=p.max_reward_1, max_reward_river2=p.max_reward_2)
    else:
        raise Exception(f'Environment {p.env_type} not found')
    s = env.reset()
    
    if agent_name == 'MFBPI':
        agent = MFBPI(p.gamma, env.ns, env.na, p.eta1, p.eta2, p.frequency_computation, True)
    elif agent_name == 'MFBPI-GEN':
        agent = MFBPI(p.gamma, env.ns, env.na, p.eta1, p.eta2, p.frequency_computation, False)
    elif agent_name == 'MBBPI':
        agent = MBBPI(p.gamma, env.ns, env.na, p.frequency_computation, True)
    elif agent_name == 'QLEARNING':
        agent = QLearning(p.gamma, env.ns, env.na, p.eta1)
    elif agent_name == 'QUCB':
        agent = QUCB(p.gamma, env.ns, env.na)
    elif agent_name == 'MBBPIBayes':
        agent = MBBPIBayes(p.gamma, env.ns, env.na, p.frequency_computation, True)
    elif agent_name == 'MFBPIProjected':
        agent = MFBPIProjected(p.gamma, env.ns, env.na, p.eta1, p.eta2, p.frequency_computation)
    elif agent_name == 'OnPolicy':
        agent = OnPolicyAgent(p.gamma, env.ns, env.na, p.eta1, p.eta2, 16, lr=1e-2)
    elif agent_name == 'MFBPIBootstrapped':
        agent = MFBPIBootstrapped(p.gamma, env.ns, env.na, p.eta1, p.eta2)
    elif agent_name == 'MFBPIUCB':
        agent = MFBPIUCB(p.gamma, env.ns, env.na, p.eta1, p.eta2)
        
    num_visits = np.zeros(env.ns)
    last_visit = np.zeros(env.ns)

    eval_greedy = policy_evaluation(p.gamma, env.transitions, env.rewards[..., np.newaxis], agent.greedy_policy)
    eval = [(agent.omega, agent.greedy_policy, num_visits, last_visit, agent.visits, eval_greedy)]
    num_visits[s] += 1
    last_visit[s] = 1

    for t in tqdm(range(p.horizon)):
        a = agent.forward(s, epsilon= max(0.1, 1 /((1+t) ** p.alpha)))
        next_state, reward = env.step(a)
        agent.backward(s, a, reward, next_state)

        s = next_state
        num_visits[s] += 1
        last_visit[s] = t+1
        
        if t % 2000 == 0:
            print(num_visits)
            import pdb
            pdb.set_trace()
        agent_greedy = agent.greedy_policy
        eval_greedy = policy_evaluation(p.gamma, env.transitions, env.rewards[..., np.newaxis], agent_greedy)
        eval.append((agent.omega, agent.greedy_policy, num_visits, last_visit, agent.visits, eval_greedy))
    return eval

def run_agent(seed: int, name: str, parameters: Parameters):
    np.random.seed(seed)
    return run(name, parameters)
    

if __name__ == '__main__':
    NUM_PROCESSES = 1
    
    for env_type, fname in [('ForkedRiverSwim', './data/data_forked_riverswim.pkl')]:
        data = {}
        parameters = Parameters(
            env_type=env_type,
            gamma = 0.99,
            river_length= 5,
            horizon = 10000,
            min_reward= 0.05,
            max_reward_1=  1,
            max_reward_2=  0.95,
            n_sims= 50,
            frequency_computation= 50,
            alpha = 0.25,
            eta1 = 0.6,
            eta2 = 0.7,
        )
        # try:
        #     with open(fname, 'rb') as f:
        #         existing_data = pickle.load(f)
        #         data = {**data, **existing_data}
        #         del existing_data
        # except:
        #     pass
            
        data['parameters'] = parameters
        data['agents'] = ['MFBPIUCB']#'MFBPIBootstrapped']#, 'MFBPI', 'MFBPI-GEN', 'MBBPI', 'QLEARNING', 'QUCB', 'MBBPIBayes', 'MFBPIProjected']
        
        if NUM_PROCESSES > 1:
            with mp.Pool(NUM_PROCESSES) as p:
                for agent in data['agents']:
                    data[f'agent_{agent}'] = list(p.starmap(run_agent,  [(seed, agent, copy.deepcopy(parameters)) for seed in  range(parameters.n_sims)]))
        else:
            for agent in data['agents']:
                    data[f'agent_{agent}'] = list(map(lambda x: run_agent(*x),  [(seed, agent, copy.deepcopy(parameters)) for seed in  range(parameters.n_sims)]))
                    
        # with open(fname, 'wb') as f:
        #     pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
 