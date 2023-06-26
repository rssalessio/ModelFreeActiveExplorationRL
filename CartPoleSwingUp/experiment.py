import numpy as np
import torch
from numpy.typing import NDArray
from env.cartpole_swingup import CartpoleSwingup, TimeStep, CartpoleSwingupConfig
from typing import Callable, Literal, Dict, Sequence, Tuple, List,NamedTuple
from agents.ids_q import default_agent as default_agent_ids
from agents.bsp import default_agent as default_agent_boot_dqn_torch
from agents.bsp2 import default_agent as default_agent_boot_dqn_modified_torch
from agents.agent import Agent
from tqdm import tqdm
from copy import deepcopy
from agents.dbmfbpi import default_agent as default_agent_dbmfbpi
from logger import Logger

class RunConfig(NamedTuple):
    agent_name: str
    seed: int
    config: CartpoleSwingup
    episodes: int
    freq_val_greedy: int
    num_eval_greedy: int

def run_agent(config: RunConfig, log_id: int, **kwargs):
    np.random.seed(config.seed)
    torch.random.manual_seed(config.seed)
    logger = Logger(f'logs/{config.agent_name}_{log_id}_{config.seed}.csv', [
        'steps', 'episode', 'total_return', 'episode_len', 'episode_return', 'total_upright', 'best_episode_reward', 'mean_greedy_rewards', 'std_greedy_rewards'
    ])
    make_env = lambda: CartpoleSwingup(config.config, config.seed)
    return run(config.agent_name, config.episodes, make_env, config.freq_val_greedy, config.num_eval_greedy, verbose=True, logger=logger, **kwargs)

class StateSpaceExploration(object):
    def __init__(self, n_cells: int = 100, x_min: float = -5., x_max: float = 5.) -> None:
        self.x_min = x_min
        self.x_max = x_max
        self.last_visit = np.zeros((n_cells, n_cells))
        self.num_visits = np.zeros((n_cells, n_cells))
        self.n_cells = n_cells
        
    def update(self, state: NDArray[np.float64], step: int) -> None:
        x = state[0,0]
        theta = np.arctan2(state[0,2], state[0,3])


        j = int(np.floor((theta + np.pi) * (self.n_cells - 1) / (2 * np.pi)))
        if x < self.x_min:
            i = 0
        elif x >  self.x_max:
            i = self.n_cells - 1
        else:
            i = int(np.floor((x - self.x_min) * (self.n_cells - 1) / (self.x_max - self.x_min)))
        self.last_visit[i,j] = step
        self.num_visits[i,j] += 1     
        
    
class AgentStats(object):
    episodes_rewards: List[Sequence[float]]
    episodes_lengths: List[int]
    greedy_rewards: Dict[int, List[float]]
    state_space_exploration: StateSpaceExploration

    def __init__(self, n_cells: int = 100) -> None:
        self.episodes_rewards: List[float] = []
        self.episodes_lengths: List[float] = []
        self.state_space_exploration = StateSpaceExploration(n_cells)
        self.greedy_rewards = {}

    def add_episode_statistics(self, rewards: List[float], steps: int):
        self.episodes_rewards.append(rewards)
        self.episodes_lengths.append(steps)
    
    def add_greedy_rewards(self, episode: int, rewards: List[float]):
        self.greedy_rewards[episode] = rewards
        
    def update_exploration(self, state: NDArray[np.float64], step: int):
        self.state_space_exploration.update(state, step)

    @property
    def num_episodes(self) -> int:
        return len(self.episodes_lengths) 
    
    @property
    def total_episodic_rewards(self) -> NDArray[np.float64]:
        return np.array([np.sum(x) for x in self.episodes_rewards])


agents: Dict[
    Literal['ids', 'bsp', 'bsp2', 'dbmfbpi'],
    Callable[[NDArray[np.float32], int], Agent]] = {
        # #'boot_dqn_tf': boot_dqn_tf_default_agent,
        'bsp': default_agent_boot_dqn_torch,
        'bsp2': default_agent_boot_dqn_modified_torch,
        'dbmfbpi': default_agent_dbmfbpi,
        'ids': default_agent_ids
    }

class Results(NamedTuple):
    training_rewards: NDArray[np.float64]
    training_steps:  NDArray[np.int64]
    greedy_rewards: Sequence[Tuple[int, NDArray[np.float64]]]
    agent_stats: AgentStats
    

    
    
@torch.inference_mode()
def evaluate_greedy(env: CartpoleSwingup, agent: Agent, num_evaluations: int) -> NDArray[np.float64]:
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
        make_env: Callable[[], CartpoleSwingup],
        frequency_greedy_evaluation: int,
        num_greedy_evaluations: int,
        verbose: bool,
        logger: Logger= None,
        **kwargs) -> Results:

    

    env = make_env()
    agent_stats = AgentStats()
    agent: Agent = agents[agent_name](env.reset(), env.num_actions, **kwargs)
    total_steps = 0
    
    training_rewards = np.zeros(n_episodes)
    training_steps = np.zeros(n_episodes)
    greedy_rewards = []
    best_episode_reward = -np.inf

    mean_greedy_rewards = 0
    std_greedy_rewards = 0
    
    tqdm_bar = tqdm(range(n_episodes))
    for episode in tqdm_bar:
        s = env.reset()
        done = False
        episode_rewards = []
        episode_steps = 0
        
        agent_stats.update_exploration(s, total_steps)

        while not done:
            action = agent.select_action(s, total_steps)
            timestep = env.step(action)
            agent.update(timestep)
            
            agent_stats.update_exploration(timestep.next_observation, total_steps)
            
            
            r, s, done = timestep.reward, timestep.next_observation, timestep.done
            episode_rewards.append(r)
            total_steps += 1
            episode_steps += 1
            
        episode_rewards = np.array(episode_rewards)
        total_reward = np.sum(episode_rewards)
        total_upright = np.sum(episode_rewards[episode_rewards > 0])

        best_episode_reward = total_reward if total_reward > best_episode_reward else best_episode_reward
        
        training_rewards[episode] = total_reward
        training_steps[episode] = episode_steps
        agent_stats.add_episode_statistics(episode_rewards, episode_steps)

        if episode % frequency_greedy_evaluation == 0:
            _greedy_rewards = evaluate_greedy(deepcopy(env), agent, num_greedy_evaluations)
            greedy_rewards.append((episode, _greedy_rewards))
            mean_greedy_rewards = np.mean(_greedy_rewards)
            std_greedy_rewards = np.std(_greedy_rewards, ddof=1)
            agent_stats.add_greedy_rewards(episode, _greedy_rewards)

        if logger:
            logger.write([total_steps, episode, np.sum(training_rewards), episode_steps, total_reward,total_upright,best_episode_reward, mean_greedy_rewards, std_greedy_rewards])
            
        if verbose:
            mu = np.round(0 if len(greedy_rewards) == 0 else np.mean(greedy_rewards[-1][1]), 3)
            std = np.round(0 if len(greedy_rewards) == 0 else np.std(greedy_rewards[-1][1]), 3)
            mu_tr = np.round(np.mean(training_rewards[max(0,episode-10):episode]), 3) if episode > 0 else total_reward
            mu_steps = np.round(np.mean(training_steps[max(0,episode-10):episode]), 3) if episode > 0 else episode_steps
            tqdm_bar.set_description(f'Ep. {episode} - Avg ep. ret./length {mu_tr}/{mu_steps} - Last greedy avg. ret. {mu} (std {std})')
    
    return Results(training_rewards, training_steps, greedy_rewards, agent_stats)


if __name__ == '__main__':
    config = RunConfig('bsp', 0,  CartpoleSwingupConfig(height_threshold= 1 / 20, x_reward_threshold= 1 - 1/20), 100, 10, 5)
    training_rwards, training_steps, greedy_rewards, agent_stats = run_agent(config, 1)
    
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(agent_stats.state_space_exploration.num_visits)
    ax[1].imshow(agent_stats.state_space_exploration.last_visit)
    plt.show()
    import pdb
    pdb.set_trace()
