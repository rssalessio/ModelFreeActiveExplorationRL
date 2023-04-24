import numpy as np
import multiprocessing as mp
import pickle
import copy
import lzma
import os
import pyximport
_ = pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)
from envs.riverswim import RiverSwim
from envs.forked_riverswim import ForkedRiverSwim
from tqdm import tqdm
from utils.cutils import policy_evaluation
from utils.utils import policy_evaluation as policy_evaluation_2
from simulation_parameters import SimulationParameters, EnvType
from agents.agent import AgentParameters, Experience
from utils.utils import Results
from make_agent import AgentType, make_agent
import time

def run(agent_type: AgentType, p: SimulationParameters):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    match p.env_type:
        case EnvType.RIVERSWIM:
            env = RiverSwim(num_states=p.river_length)
        case EnvType.FORKED_RIVERSWIM:
            env = ForkedRiverSwim(river_length=p.river_length)
        case _:
            raise NotImplementedError(f'Environment {env_type.value} not found.')

    start_time = time.time()
    s = env.reset()
    discount_factor = p.gamma
    agent_parameters = AgentParameters(dim_state_space=env.ns, dim_action_space=env.na, discount_factor=discount_factor)
    agent = make_agent(agent_type, agent_parameters)
    
    eval_greedy = np.asarray(policy_evaluation(p.gamma, env.transitions, env.rewards[..., np.newaxis], agent.greedy_policy))
    eval = [Results(0, agent.omega, agent.greedy_policy, agent.total_state_visits, agent.last_visit, agent.exp_visits, eval_greedy, time.time() - start_time)]

    #for t in tqdm(range(p.horizon)):
    for t in range(p.horizon):
        a = agent.forward(s, t)
        next_state, reward = env.step(a)
        exp = Experience(s, a, reward, next_state)
        agent.backward(exp, t)

        s = next_state

        if (t +1) % p.frequency_evaluation == 0:
            eval_greedy = np.asarray(policy_evaluation(p.gamma, env.transitions, env.rewards[..., np.newaxis], agent.greedy_policy))
            #print(f'{t}:{agent.total_state_visits}  - {agent.greedy_policy} --{1} - {agent.omega}')
            eval.append(Results(t, agent.omega, agent.greedy_policy, agent.total_state_visits, agent.last_visit, agent.exp_visits, eval_greedy, time.time() - start_time))
    # print(agent.total_state_visits)
    # import pdb
    # pdb.set_trace()
    return eval

def run_agent(seed: int, agent_type: AgentType, parameters: SimulationParameters):
    np.random.seed(seed)
    return run(agent_type, parameters)
# [
#                         AgentType.Q_UCB, AgentType.Q_LEARNING, AgentType.BPI_NEW_BOUND_BAYES,
#                         AgentType.BPI_NEW_BOUND,  AgentType.OBPI, AgentType.PGOBPI,
#                         AgentType.BPI_NEW_BOUND_SIMPLIFIED_1, AgentType.MDP_NAS]:
if __name__ == '__main__':
    NUM_PROCESSES = 8
    
    types = [
        (5, EnvType.RIVERSWIM, 15000),
        (10, EnvType.RIVERSWIM, 30000),
        (20, EnvType.RIVERSWIM, 50000),
        (3, EnvType.FORKED_RIVERSWIM, 15000),
        (5, EnvType.FORKED_RIVERSWIM, 30000),
        (20, EnvType.RIVERSWIM, 50000),
        (30, EnvType.RIVERSWIM, 100000),
        (15, EnvType.FORKED_RIVERSWIM, 100000)
    ]
    
    
    
    agents = [
        AgentType.Q_LEARNING, AgentType.Q_UCB,
        AgentType.OBPI, AgentType.BAYESOBPI,
        AgentType.MDP_NAS, AgentType.BPI_NEW_BOUND,  AgentType.BPI_NEW_BOUND_SIMPLIFIED_1,
        # AgentType.BPI_NEW_BOUND_BAYES,AgentType.PGOBPI,
    ]

    for length, env_type, horizon in types:
        for agent in agents:
            print(f'> Evaluating {agent.value} on {env_type.value}({length})', end='... ')
    
            path = f'./data/{env_type.value}/'
            if not os.path.exists(path):
                os.makedirs(path)

            data = {}
            
            num_states = length if env_type == EnvType.RIVERSWIM else length *2

            data['simulation_parameters'] = SimulationParameters(
                env_type=env_type,
                gamma=0.99,
                river_length=length,
                horizon=horizon,
                n_sims = int(NUM_PROCESSES * 2),
                frequency_evaluation=100
            )
            data['agent_type'] = agent

            iterations = [(seed, agent, data['simulation_parameters']) for seed in  range(data['simulation_parameters'].n_sims)]
            start_time = time.time()
            if NUM_PROCESSES > 1:
                with mp.Pool(NUM_PROCESSES) as p:
                    data_returned = list(p.starmap(run_agent, iterations))
            else:
                data_returned = list(map(lambda x: run_agent(*x), iterations))

            data['data'] = data_returned
            print(f'done in {np.round(time.time() - start_time, 2)} seconds.')
            with lzma.open(f'{path}/{agent.value}_{length}.pkl.lzma', 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
