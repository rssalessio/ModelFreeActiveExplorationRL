import numpy as np
from numpy.typing import NDArray
from agents.agent import TimeStep, Agent
# from agents.boot_dqn import default_agent as boot_dqn_tf_default_agent
from agents.boot_dqn_torch import default_agent as boot_dqn_torch_default_agent
from agents.bdqn import default_agent as bqdn_default_agent
from agents.explorative_generative_off_policy import default_agent as explorative_generative_off_policy_default_agent
from agents.explorative_projected_on_policy import default_agent as explorative_projected_on_policy_default_agent
from agents.explorative_generative_off_policy_2 import default_agent as explorative_generative_off_policy_default_agent2
from deepsea import MultiRewardsDeepSea
from typing import Callable, Sequence, Tuple, Dict, Literal, Callable
from tqdm import tqdm
import torch

agents: Dict[
    Literal['boot_dqn_torch', 'bdqn', 'explorative_generative_off_policy', 'explorative_projected_on_policy_agent'],
    Callable[[NDArray[np.float32], int], Agent]] = {
        #'boot_dqn_tf': boot_dqn_tf_default_agent,
        'boot_dqn_torch': boot_dqn_torch_default_agent,
        'bqdn': bqdn_default_agent,
        'explorative_generative_off_policy': explorative_generative_off_policy_default_agent,
        'explorative_projected_on_policy_agent': explorative_projected_on_policy_default_agent,
        'explorative_generative_off_policy2': explorative_generative_off_policy_default_agent2
    }

def evaluate_greedy(make_env: Callable[[], MultiRewardsDeepSea], agent: Agent, num_evaluations: int) -> NDArray[np.float64]:
    total_rewards = np.zeros(num_evaluations)
    env = make_env()
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
        verbose: bool = True) -> Tuple[NDArray[np.float64], Sequence[Tuple[int, NDArray[np.float64]]], NDArray[np.float64]]:
    
    env = make_env()
    agent: Agent = agents[agent_name](env.reset(), 2)
    total_steps = 0
    
    training_rewards = np.zeros(n_episodes)
    regret = np.zeros(n_episodes+1)
    greedy_rewards = []
    
    tqdm_bar = tqdm(range(n_episodes))

    for episode in tqdm_bar:
        s = env.reset()
        done = False
        episode_rewards = 0
        
        while not done:
            action = agent.select_action(s, total_steps)
            timestep = env.step(action)
            agent.update(timestep)
            
            r, s, done = timestep.reward, timestep.next_observation, timestep.done
            episode_rewards += r
            total_steps += 1
            
            if total_steps % frequency_greedy_evaluation == 0:
                _greedy_rewards = evaluate_greedy(make_env, agent, num_greedy_evaluations)
                greedy_rewards.append((total_steps, _greedy_rewards))
        
        training_rewards[episode] = episode_rewards
        regret[episode + 1] = regret[episode] + env.optimal_return - episode_rewards

        if verbose:
            mu = np.round(0 if len(greedy_rewards) == 0 else np.mean(greedy_rewards[-1][1]), 3)
            std = np.round(0 if len(greedy_rewards) == 0 else np.std(greedy_rewards[-1][1]), 3)
            mu_tr = np.round(np.mean(training_rewards[max(0,episode-10):episode]), 3) if episode > 0 else episode_rewards
            curr_regret = np.round(regret[episode + 1], 3)
            tqdm_bar.set_description(f'Ep. {episode} - Regret: {curr_regret} - Last 10 ep. avg ret. {mu_tr} - Last greedy avg. ret. {mu} (std {std})')
    
    return training_rewards, greedy_rewards, regret


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    np.random.seed(10)
    torch.random.manual_seed(10)
    #0.1
    make_env = lambda: MultiRewardsDeepSea(20, 1, enable_multi_rewards=False, randomize=True, slipping_probability=0.0)
    
    # env1 = MultiRewardsDeepSea(10, 1, enable_multi_rewards=False, randomize=True, slipping_probability=0.)
    # env2 = MultiRewardsDeepSea(10, 1, enable_multi_rewards=False, randomize=True, slipping_probability=0.1)
    # env3 = MultiRewardsDeepSea(10, 1, enable_multi_rewards=True, randomize=True, slipping_probability=0.)
    # env4 = MultiRewardsDeepSea(10, 1, enable_multi_rewards=True, randomize=True, slipping_probability=0.1)
    # import pdb
    # pdb.set_trace()
    env = make_env()
    print(f'The optimal average return for this environment is {env.optimal_return}')
    
    #training_rewards, greedy_rewards, regret = run('explorative_projected_on_policy_agent', 1000, make_env, 100, 50)
    training_rewards, greedy_rewards, regret = run('explorative_generative_off_policy', 1000, make_env, 100, 50)
    training_rewards, greedy_rewards, regret = run('boot_dqn_torch', 1000, make_env, 100, 50)
    
    
    
    #boot_dqn_torch 390 0.893 0.852
    #explorative_generative_off_policy
    # import matplotlib.pyplot as plt
    # greedy = np.array([greedy_rewards[x][1] for x in range(len(greedy_rewards))])
    # plt.plot(greedy.mean(-1))
    # plt.plot(greedy.mean(-1) + 1.96*greedy.std(-1,ddof=1) / np.sqrt(greedy.shape[-1]), '--')
    # plt.plot(greedy.mean(-1) - 1.96*greedy.std(-1,ddof=1) / np.sqrt(greedy.shape[-1]), '--')
    # plt.show()