import numpy as np
from experiment import agents, run_agent, AgentStats, Results, CartpoleSwingupConfig,RunConfig
from numpy.typing import NDArray
from typing import NamedTuple, Sequence, Tuple
import multiprocessing as mp
from scipy.stats import t as tstudent
import torch
import pickle
from torch.multiprocessing import Pool, Process, set_start_method
import lzma
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


if __name__ == '__main__':
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    FREQ_EVAL_GREEDY = 10
    NUM_EVAL_GREEDY = 20
    NUM_PROC = 10
    NUM_RUNS = 20

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    with Pool(NUM_PROC) as pool:
        #for size, agent_name in runs:
        for size in [1,3,5,10]:
            for agent_name in ['bsp','bsp2','dbmfbpi','ids']:
                episodes = 200
                cartpole_config = CartpoleSwingupConfig(height_threshold= size / 20, x_reward_threshold= 1 - size/20)
                print(f'Running agent {agent_name} - episodes: {episodes} - N {size} - config: {cartpole_config._asdict()}')
                
                
                run_configs = [(RunConfig(agent_name, idx, cartpole_config, episodes, FREQ_EVAL_GREEDY, NUM_EVAL_GREEDY), size) for idx in range(NUM_RUNS)]
                
                data = list(pool.starmap(run_agent, run_configs))

                with lzma.open(f'data/data_{size}_{agent_name}.pkl.lzma', 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)