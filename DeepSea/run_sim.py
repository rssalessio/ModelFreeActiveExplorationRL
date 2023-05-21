# Copyright (c) [2023] [NeurIPS authors, 11410]
# 
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

# Importing necessary modules and libraries
import numpy as np
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

# Set numpy print options for better visibility
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

# Define a class to hold the results of the simulations
class Results(NamedTuple):
    training_rewards: NDArray[np.float64]
    greedy_rewards: Sequence[Tuple[int, NDArray[np.float64]]]
    regret: NDArray[np.float64]
    agent_stats: AgentStats
    
def main() -> None:
    """
    Main function to control the running of the agents in the defined DeepSea environment.
    """
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    # Define constants
    FREQ_EVAL_GREEDY = 200
    NUM_EVAL_GREEDY = 20
    NUM_PROC = 8
    NUM_RUNS = 24
    SLIPPING_PROBABILITY = 0.05

    # Define parameters for the environment and agents
    parameters = {
        10: {
            'horizon': 1000,
            'bsp': {'num_ensemble': 20, 'prior_scale': 3},
            'bsp2': {'num_ensemble': 20, 'prior_scale': 3},
            'ids': {'num_ensemble': 20,},
            'dbmfbpi': {'num_ensemble': 20, 'prior_scale': 3},
            },
        20: {
            'horizon': 2000,
            'bsp2': {'num_ensemble': 20, 'prior_scale': 5},
            'bsp': {'num_ensemble': 20, 'prior_scale': 5},
            'ids': {'num_ensemble': 25,},
            'dbmfbpi': {'num_ensemble': 20, 'prior_scale': 5},
            },
        30: {
            'horizon': 3000,
            'bsp2': {'num_ensemble': 20, 'prior_scale': 10},
            'bsp': {'num_ensemble': 20, 'prior_scale': 10},
            'ids': {'num_ensemble': 30,},
            'dbmfbpi': {'num_ensemble': 20, 'prior_scale': 10},
            },
        40: {
            'horizon': 4000,
            'bsp2': {'num_ensemble': 20, 'prior_scale': 15},
            'bsp': {'num_ensemble': 20, 'prior_scale': 15},
            'ids': {'num_ensemble': 35,},
            'dbmfbpi': {'num_ensemble': 20, 'prior_scale': 15},
            },
        50: {
            'horizon': 5000,
            'bsp2': {'num_ensemble': 20, 'prior_scale': 20},
            'bsp': {'num_ensemble': 20, 'prior_scale': 20},
            'ids': {'num_ensemble': 40,},
            'dbmfbpi': {'num_ensemble': 20, 'prior_scale': 20}
            },
    }

    # Using multiprocessing to run the agents in parallel
    with Pool(NUM_PROC) as pool:
        for size in [10,20,30,40,50]:
            for agent_name in ['ids','dbmfbpi', 'bsp2', 'bsp']:
                HORIZON = parameters[size]['horizon']
                agent_parameters = parameters[size][agent_name]
                
                # Print current agent details
                print(f'Running agent {agent_name} - horizon: {HORIZON} - size {size} - parameters {agent_parameters}')
                
                # Use multiprocessing to run agents and collect their rewards and stats
                training_rewards, greedy_rewards, regret, stats = zip(*pool.starmap(
                    run_agent, [(agent_name, idx, False, size, 1, SLIPPING_PROBABILITY, HORIZON, FREQ_EVAL_GREEDY, NUM_EVAL_GREEDY, agent_parameters) for idx in range(NUM_RUNS)]))

                # Save the results of the run
                data = Results(training_rewards, greedy_rewards, regret, stats)

                # Save the data into a lzma compressed file
                with lzma.open(f'data/data_{size}_{agent_name}.pkl.lzma', 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()