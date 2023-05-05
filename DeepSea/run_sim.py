import numpy as np
import matplotlib.pyplot as plt
from run import agents, run_agent, AgentStats
from numpy.typing import NDArray
from typing import NamedTuple, Sequence, Tuple
from deepsea import MultiRewardsDeepSea
import multiprocessing as mp
from scipy.stats import t as tstudent
import torch
import pickle
from torch.multiprocessing import Pool, Process, set_start_method
import lzma
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


class Results(NamedTuple):
    training_rewards: NDArray[np.float64]
    greedy_rewards: Sequence[Tuple[int, NDArray[np.float64]]]
    regret: NDArray[np.float64]
    agent_stats: AgentStats
    

if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    FREQ_EVAL_GREEDY = 200
    NUM_EVAL_GREEDY = 20
    NUM_PROC = 8
    NUM_RUNS = 16
    SLIPPING_PROBABILITY = 0.05

    parameters = {
        10: {
            'horizon': 500,
            # 'boot_dqn_torch': {'num_ensemble': 20, 'prior_scale': 3},
            'boot_dqn_torch_modified': {'num_ensemble': 20, 'prior_scale': 3},
            # 'ids': {'num_ensemble': 20,},
            # 'explorative_generative_off_policy': {'num_ensemble': 20, 'prior_scale': 3},
            },
        15: {
            'horizon': 1000,
            'boot_dqn_torch_modified': {'num_ensemble': 20, 'prior_scale': 3},
            # 'boot_dqn_torch': {'num_ensemble': 20, 'prior_scale': 3},
            # 'ids': {'num_ensemble': 20,},
            # 'explorative_generative_off_policy': {'num_ensemble': 20, 'prior_scale': 3},
            },
        20: {
            'horizon': 1500,
            'boot_dqn_torch_modified': {'num_ensemble': 20, 'prior_scale': 5},
            # 'boot_dqn_torch': {'num_ensemble': 20, 'prior_scale': 5},
            # 'ids': {'num_ensemble': 25,},
            # 'explorative_generative_off_policy': {'num_ensemble': 20, 'prior_scale': 5},
            },
        30: {
            'horizon': 2500,
            'boot_dqn_torch_modified': {'num_ensemble': 20, 'prior_scale': 10},
            # 'boot_dqn_torch': {'num_ensemble': 20, 'prior_scale': 10},
            # 'ids': {'num_ensemble': 30,},
            # 'explorative_generative_off_policy': {'num_ensemble': 20, 'prior_scale': 10},
            },
        40: {
            'horizon': 3500,
            'boot_dqn_torch_modified': {'num_ensemble': 20, 'prior_scale': 15},
            # 'boot_dqn_torch': {'num_ensemble': 20, 'prior_scale': 15},
            # 'ids': {'num_ensemble': 40,},
            # 'explorative_generative_off_policy': {'num_ensemble': 20, 'prior_scale': 15},
            },
        50: {
            'horizon': 4500,
            'boot_dqn_torch_modified': {'num_ensemble': 20, 'prior_scale': 20},
            # 'boot_dqn_torch': {'num_ensemble': 20, 'prior_scale': 20},
            # 'ids': {'num_ensemble': 50,},
            # 'explorative_generative_off_policy': {'num_ensemble': 20, 'prior_scale': 20}
            }
    }



    with Pool(NUM_PROC) as pool:
        for size in parameters.keys():
            data = {}
            
            for agent_name in ['boot_dqn_torch_modified']: #explorative_generative_off_policy','boot_dqn_torch', 'ids']:
                HORIZON = parameters[size]['horizon']
                agent_parameters = parameters[size][agent_name]
                print(f'Running agent {agent_name} - horizon: {HORIZON} - size {size} - parameters {agent_parameters}')
                training_rewards, greedy_rewards, regret, stats = zip(*pool.starmap(
                    run_agent, [(agent_name, idx, False, size, 1, SLIPPING_PROBABILITY, HORIZON, FREQ_EVAL_GREEDY, NUM_EVAL_GREEDY, agent_parameters) for idx in range(NUM_RUNS)]))
                data[agent_name] = Results(training_rewards, greedy_rewards, regret, stats)


            with lzma.open(f'data_{size}_modified.pkl', 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)