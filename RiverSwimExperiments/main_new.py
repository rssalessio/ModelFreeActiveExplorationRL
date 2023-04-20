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

    s = env.reset()
    discount_factor = p.gamma
    agent_parameters = AgentParameters(dim_state_space=env.ns, dim_action_space=env.na, discount_factor=discount_factor)
    agent = make_agent(agent_type, agent_parameters)
    
    eval_greedy = np.asarray(policy_evaluation(p.gamma, env.transitions, env.rewards[..., np.newaxis], agent.greedy_policy))
    eval = [Results(0, agent.omega, agent.greedy_policy, agent.total_state_visits, agent.last_visit, agent.exp_visits, eval_greedy)]

    #for t in tqdm(range(p.horizon)):
    for t in range(p.horizon):
        a = agent.forward(s, t)
        next_state, reward = env.step(a)
        exp = Experience(s, a, reward, next_state)
        agent.backward(exp, t)

        s = next_state

        if (t +1) % p.frequency_evaluation == 0:
            eval_greedy = np.asarray(policy_evaluation(p.gamma, env.transitions, env.rewards[..., np.newaxis], agent.greedy_policy))
            eval.append(Results(t, agent.omega, agent.greedy_policy, agent.total_state_visits, agent.last_visit, agent.exp_visits, eval_greedy))
    print(agent.total_state_visits)
    return eval

def run_agent(seed: int, agent_type: AgentType, parameters: SimulationParameters):
    import psutil
    p = psutil.Process()

    # created = mp.Process()
    # current = mp.current_process()
    # print 'running:', current.name, current._identity

    cpu_worker= mp.current_process()._identity[0] - 1
    p.cpu_affinity([cpu_worker])
    #print(f"Child #{cpu_worker} - {mp.current_process()._identity[0]}: {p}, affinity {p.cpu_affinity()}", flush=True)

    np.random.seed(seed)
    return run(agent_type, parameters)

if __name__ == '__main__':
    NUM_PROCESSES = 8
    
    for length in [5, 10, 20]:
        for env_type in [EnvType.RIVERSWIM, EnvType.FORKED_RIVERSWIM]:
                for agent in [AgentType.MDP_NAS, AgentType.OBPI, AgentType.BPI_BAYES]:
                    print(f'> Evaluating {agent.value} on {env_type.value}({length})', end='... ')
                    

                    path = f'./data/{env_type.value}/'
                    if not os.path.exists(path):
                        os.makedirs(path)

                    data = {}

                    data['simulation_parameters'] = SimulationParameters(
                        env_type=env_type,
                        gamma=0.99,
                        river_length=length,
                        horizon=15000,
                        n_sims = int(NUM_PROCESSES * 6),
                        frequency_evaluation=50
                    )
                    data['agent_type'] = agent

                    iterations = [(seed, agent, data['simulation_parameters']) for seed in  range(data['simulation_parameters'].n_sims)]
                    start_time = time.time()
                    if NUM_PROCESSES > 1:
                        with mp.Pool(NUM_PROCESSES) as p:
                            data_returned = list(p.starmap(run_agent, iterations))
                    else:
                        data_returned = list(map(lambda x: run_agent(*x), iterations))
                    end_time = time.time()
                    data['data'] = data_returned
                    print(f'done in {np.round(end_time - start_time, 2)} seconds.')
                    with lzma.open(f'{path}/{agent.value}_{length}.pkl.lzma', 'wb') as f:
                        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                    
