from maze import Maze, Action
from agents.dqn import DQN, default_agent as dqn_default_agent
from agents.boot_dqn import BootstrappedDqn, default_agent as bootstrapped_dqn_default_agent
from agents.onpolicy_agent import default_agent as onpolicy_default_agent
import numpy as np
from tqdm import tqdm
formatter = lambda x: np.round(x, 3)

def run_env(env: Maze, agent: DQN, T: int):
    
    episodes_rewards = []
    step = 0
    t_description = tqdm(range(T))
    t_description.set_description(f'Step {step} - avg rwd: {0}')
    while step < T:
        done = False
        episode_reward = []
        s = env.reset()
        losses = []
        
        while done is False and step < T:
            action = agent.select_action(s)
            next_state, reward, done = env.step(Action(action))
            loss = agent.update(s, action, reward, next_state, done)
            episode_reward.append(reward)
            
            s = next_state
            step += 1
            
            if loss is not None:
                losses.append(loss)
        episodes_rewards.append(episode_reward)
        
        cum_rew = formatter(np.sum(episode_reward))
        mean_rew = formatter(np.mean(episode_reward)) if len(episode_reward) > 0 else 0
        mean_loss = formatter(np.mean(losses)) if len(losses) > 0 else 0
        t_description.set_description(f'Step {step} - last ep rwd: {cum_rew} - avg rwd: {mean_rew} - avg loss: {mean_loss}')
        t_description.update(step - t_description.last_print_n)
        
    return episodes_rewards
            

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    env = Maze(observe_entire_grid=True, time_limit=True, movement_reward=0, slippery_probability=0.1)
    s0 = env.reset()
    num_actions = 4
    seed = 42
    T = 100000
    dqn = onpolicy_default_agent(s0, num_actions, seed=seed)
    results  = list(map(np.sum, run_env(env, dqn, T)))
    
    s = env.reset()
    done = False
    rewards = []
    import pdb
    pdb.set_trace()
    while done is False:
        action = dqn.select_action(s, greedy=True)
        print(f'{s}- {action}')
        snext, reward, done = env.step(Action(action))
        s = snext
        rewards.append(reward)
    
    print(results)
    plt.hist(results)
    plt.show()