# Copyright (c) [2023] [NeurIPS authors, 11410]
# 
# This file is licensed under the MIT License.
# See the LICENSE file in the project root for full license information.

import numpy as np
from numpy.typing import NDArray
from agents.agent import TimeStep, Agent
from agents.bsp import default_agent as boot_dqn_torch_default_agent
from agents.dbmfbpi import default_agent as dbmfbpi_default_agent
from agents.bsp2 import default_agent as boot_dqn_torch_default_agent_modified
from agents.ids_q import default_agent as idsq_default_agent
from deepsea import MultiRewardsDeepSea
from typing import Callable, Sequence, Tuple, Dict, Literal, Callable
from tqdm import tqdm
import torch
from typing import NamedTuple
from copy import deepcopy

def run_agent(agent_name: str, seed: int, multi_rewards: bool, size: int, max_reward: float, slipping_probability: float,
                num_episodes: int, freq_val_greedy: int, num_eval_greedy:int, kwargs):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    make_env = lambda: MultiRewardsDeepSea(size, max_reward,
                                           enable_multi_rewards=multi_rewards,
                                           slipping_probability=slipping_probability,
                                           randomize=True)
    return run(agent_name, num_episodes, make_env, freq_val_greedy, num_eval_greedy, verbose=True, **kwargs)

class AgentStats(object):
    total_num_visits: NDArray[np.float64]
    frequency_visits: NDArray[np.float64]
    last_visit: NDArray[np.float64]

    def __init__(self, N: int) -> None:
        self.frequency_visits = np.zeros((N, N))
        self.last_visit = np.zeros((N, N))
        self.total_num_visits = np.zeros((N, N))
        self._counter = 0
    
    def update(self, state: NDArray[np.float64]):
        self._counter += 1
        self.total_num_visits = self.total_num_visits +  state
        self.frequency_visits = self.total_num_visits / self._counter
        self.last_visit[state.astype(np.bool_)] = self._counter


agents: Dict[
    Literal['ids', 'bsp', 'bsp2', 'dbmfbpi'],
    Callable[[NDArray[np.float32], int], Agent]] = {
        'bsp': boot_dqn_torch_default_agent,
        'bsp2': boot_dqn_torch_default_agent_modified,
        'dbmfbpi': dbmfbpi_default_agent,
        'ids': idsq_default_agent
    }

@torch.inference_mode()
def evaluate_greedy(env: MultiRewardsDeepSea, agent: Agent, num_evaluations: int) -> NDArray[np.float64]:
    total_rewards = np.zeros(num_evaluations)
    for episode in range(num_evaluations):
        episodic_rewards = []
        s = env.reset()
        done = False
        while not done:
            action = agent.select_greedy_action(s)
            timestep = env.step(action)            
            r, s, done = timestep.reward, timestep.next_observation, timestep.done
            episodic_rewards.append(r)
        total_rewards[episode] = np.sum(episodic_rewards)
    return total_rewards
            

def run(agent_name: str,
        n_episodes: int,
        make_env: Callable[[], MultiRewardsDeepSea],
        frequency_greedy_evaluation: int,
        num_greedy_evaluations: int,
        verbose: bool, **kwargs) -> Tuple[NDArray[np.float64], Sequence[Tuple[int, NDArray[np.float64]]], NDArray[np.float64], AgentStats]:
    
    
    env = make_env()
    agent_stats = AgentStats(env._size)
    agent: Agent = agents[agent_name](env.reset(), 2, **kwargs)
    total_steps = 0
    
    training_rewards = np.zeros(n_episodes)
    regret = np.zeros(n_episodes+1)
    greedy_rewards = []
    
    tqdm_bar = tqdm(range(n_episodes))

    for episode in tqdm_bar:
        s = env.reset()
        done = False
        episode_rewards = 0
        agent_stats.update(s)

        while not done:
            action = agent.select_action(s, total_steps)
            timestep = env.step(action)
            agent.update(timestep)
            
            
            r, s, done = timestep.reward, timestep.next_observation, timestep.done
            episode_rewards += r
            total_steps += 1
            if not done:
                agent_stats.update(s)
            
            if total_steps % frequency_greedy_evaluation == 0:
                _greedy_rewards = evaluate_greedy(deepcopy(env), agent, num_greedy_evaluations)
                greedy_rewards.append((total_steps, _greedy_rewards))
        
        training_rewards[episode] = episode_rewards
        regret[episode + 1] = regret[episode] + env.optimal_return - episode_rewards

        if verbose:
            mu = np.round(0 if len(greedy_rewards) == 0 else np.mean(greedy_rewards[-1][1]), 3)
            std = np.round(0 if len(greedy_rewards) == 0 else np.std(greedy_rewards[-1][1]), 3)
            mu_tr = np.round(np.mean(training_rewards[max(0,episode-10):episode]), 3) if episode > 0 else episode_rewards
            curr_regret = np.round(regret[episode + 1], 3)
            #if episode % (n_episodes // 10) == 0:
            tqdm_bar.set_description(f'Ep. {episode} - Regret: {curr_regret} - Last 10 ep. avg ret. {mu_tr} - Last greedy avg. ret. {mu} (std {std})')
    
    return training_rewards, greedy_rewards, regret, agent_stats

def compute_statistics(stats: AgentStats, env: MultiRewardsDeepSea) -> Tuple[float, float, float]:

    mask = ~np.isclose(0, stats.total_num_visits)
    omega = stats.frequency_visits[mask]

    sparsity = -(omega * np.log(omega) ).sum()
    avg_time_spent = omega.mean()
    std_time_spent = omega.max() - omega.min()
    avg_last_visit = stats.last_visit[mask].mean()
    std_last_visit = stats.last_visit[mask].max()  - stats.last_visit[mask].min()

    z = stats.total_num_visits[mask]
    q = (stats.last_visit[mask] - z) / (stats.last_visit[mask] - z).sum()
    q = -(q * np.log(q) / np.log(len(q))).sum()
    sparsity2 = -(omega * avg_time_spent * np.log(omega) / np.log(len(omega))).sum()
    return (sparsity, sparsity2, avg_time_spent,std_time_spent,  avg_last_visit, std_last_visit, q)

if __name__ == '__main__':
    parameters = {
        10: {
            'horizon': 1000,
            'bsp': {'num_ensemble': 20, 'prior_scale': 3},
            'ids': {'num_ensemble': 20,},
            'dbmfbpi': {'num_ensemble': 20, 'prior_scale': 3},
            'bsp2': {'num_ensemble': 20, 'prior_scale': 3},
            },
        15: {
            'horizon': 500,
            'bsp': {'num_ensemble': 20, 'prior_scale': 3},
            'ids': {'num_ensemble': 20,},
            'bsp2': {'num_ensemble': 20, 'prior_scale': 3},},
        20: {
            'horizon': 2000,
            'bsp': {'num_ensemble': 20, 'prior_scale': 5},
            'ids': {'num_ensemble': 25},
            'dbmfbpi': {'num_ensemble': 20, 'prior_scale': 5},
            'bsp2': {'num_ensemble': 20, 'prior_scale': 5},
            },
        30: {
            'horizon': 3000,
            'bsp': {'num_ensemble': 20, 'prior_scale': 10},
            'ids': {'num_ensemble': 30,},
            'dbmfbpi': {'num_ensemble': 20, 'prior_scale': 10},
            'bsp2': {'num_ensemble': 20, 'prior_scale': 10},
            }
    }
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

    for SIZE in [20]:
        SEED = 100
        np.random.seed(SEED)
        torch.random.manual_seed(SEED)
        #0.1
        # SIZE = 30
        SLIPPING = 0.05
        Nsteps = parameters[SIZE]['horizon']
        make_env = lambda: MultiRewardsDeepSea(SIZE, 1, enable_multi_rewards=False, randomize=True, slipping_probability=SLIPPING)
        

        env = make_env()
        print(env._rewards)
        import seaborn as sns
        import matplotlib.pyplot as plt
        print(f'The optimal average return for this environment is {env.optimal_return}')
        
        
        #training_rewards, greedy_rewards, regret = run('dbmfbpi', 1000, make_env, 100, 50)
        from copy import deepcopy

        # training_rewards, greedy_rewards, regret, stats_boot = run('bsp', Nsteps, make_env, 200, 20, verbose=True, **parameters[SIZE]['boot_dqn_torch'])
        # print(compute_statistics(stats_boot, env))
        # fig, ax = plt.subplots(1,3)
        # with sns.axes_style("white"):
        #     sns.heatmap(stats_boot.total_num_visits, square=True,  cmap="YlGnBu", ax=ax[0])
        #     sns.heatmap(stats_boot.frequency_visits, square=True,  cmap="YlGnBu", ax=ax[1])
        #     sns.heatmap(stats_boot.last_visit, square=True,  cmap="YlGnBu", ax=ax[2])
        # plt.show()
        
        # np.random.seed(SEED)
        # torch.random.manual_seed(SEED)
        # training_rewards, greedy_rewards, regret, stats_exp = run('dbmfbpi', Nsteps, make_env, 200, 20, verbose=True, **parameters[SIZE]['explorative_generative_off_policy'])
        # print(compute_statistics(stats_exp, env))
        # fig, ax = plt.subplots(1,3)
        # with sns.axes_style("white"):
        #     sns.heatmap(stats_exp.total_num_visits, square=True,  cmap="YlGnBu", ax=ax[0])
        #     sns.heatmap(stats_exp.frequency_visits, square=True,  cmap="YlGnBu", ax=ax[1])
        #     sns.heatmap(stats_exp.last_visit, square=True,  cmap="YlGnBu", ax=ax[2])
        # plt.show()
        
        
        
        

        
        np.random.seed(SEED)
        torch.random.manual_seed(SEED)
        print('IDS')
        training_rewards, greedy_rewards, regret, stats_ids= run('ids', Nsteps, make_env, 200, 20, verbose=True, **parameters[SIZE]['ids'])
        print(compute_statistics(stats_ids, env))
        # fig, ax = plt.subplots(1,3)
        # with sns.axes_style("white"):
        #     sns.heatmap(stats_ids.total_num_visits, square=True,  cmap="YlGnBu", ax=ax[0])
        #     sns.heatmap(stats_ids.frequency_visits, square=True,  cmap="YlGnBu", ax=ax[1])
        #     sns.heatmap(stats_ids.last_visit, square=True,  cmap="YlGnBu", ax=ax[2])
        # plt.show()
        
        # np.random.seed(SEED)
        # torch.random.manual_seed(SEED)
        # training_rewards, greedy_rewards, regret, stats_ids= run('ids', Nsteps, make_env, 200, 20, verbose=True, **parameters[SIZE]['ids'])
        # print(compute_statistics(stats_ids, env))
        print('------------------------------------')
        
        # fig, ax = plt.subplots(3,3)
        # for i, stat in enumerate([stats_exp, stats_ids, stats_boot]):
        #     print(compute_statistics(stat, env))
        #     with sns.axes_style("white"):
        #         sns.heatmap(stat.total_num_visits, square=True,  cmap="YlGnBu", ax=ax[i,0])
        #         sns.heatmap(stat.frequency_visits, square=True,  cmap="YlGnBu", ax=ax[i,1])
        #         sns.heatmap(stat.last_visit, square=True,  cmap="YlGnBu", ax=ax[i,2])
        # plt.show()

    #boot_dqn_torch 390 0.893 0.852
    #explorative_generative_off_policy
    # import matplotlib.pyplot as plt
    # greedy = np.array([greedy_rewards[x][1] for x in range(len(greedy_rewards))])
    # plt.plot(greedy.mean(-1))
    # plt.plot(greedy.mean(-1) + 1.96*greedy.std(-1,ddof=1) / np.sqrt(greedy.shape[-1]), '--')
    # plt.plot(greedy.mean(-1) - 1.96*greedy.std(-1,ddof=1) / np.sqrt(greedy.shape[-1]), '--')
    # plt.show()