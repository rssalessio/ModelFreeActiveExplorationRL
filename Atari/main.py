import gymnasium as gym
from gymnasium.wrappers import StepAPICompatibility, AtariPreprocessing, AutoResetWrapper, FrameStack
from wrappers.clip_reward import ClipReward
from agents.bsp import default_agent
import torch
import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple
from agents.agent import Agent, TimeStep
import matplotlib.pyplot as plt
from tqdm import tqdm

class AgentStatistics(object):
    def __init__(self):
        self.episodes_rewards = []
        self.episodes_steps = []
        
        self.current_rewards = []
        self.current_steps = 0
        self.num_episodes = 0
    
    def update(self, experience: TimeStep):
        self.current_steps += 1
        self.current_rewards.append(experience.reward)
        
        if experience.done:
            self.conclude_episode()
    
    def conclude_episode(self):
        self.num_episodes += 1
        self.episodes_steps.append(self.current_steps)
        self.current_steps = 0
        
        self.episodes_rewards.append(self.current_rewards)
        self.current_rewards = []
        
    @property
    def total_episodic_rewards(self) -> NDArray[np.float64]:
        return np.array([np.sum(x) for x in self.episodes_rewards])
        


def run(env_name: str, make_agent: Callable[[NDArray, NDArray], Agent], num_steps: int = 1000, seed: int = 42) -> AgentStatistics:
    env = gym.make(env_name,  frameskip=1)#render_mode="human",
    env = AtariPreprocessing(AutoResetWrapper(ClipReward(env)), scale_obs=True)
    env = StepAPICompatibility(FrameStack(env, num_stack=4), False)
    env.action_space.seed(seed)
    agent = make_agent(env.observation_space.shape, env.action_space.n)

    observation, info = env.reset(seed=seed)
    
    agent_statistics = AgentStatistics()
    
    trange = tqdm(range(num_steps))
    
    for t in range(num_steps):
        action = agent.select_action(np.array(observation), t)
        next_observation, reward, done, info = env.step(action)
        experience = TimeStep(observation, action, reward, done, next_observation)
        agent.update(experience)
        agent_statistics.update(experience)
        observation = next_observation
        
        if done:
            print(f'[Step {t}] Episode rewards: {agent_statistics.total_episodic_rewards[-1]}')
        
    env.close()
    return agent_statistics

agent_statistics = run(env_name = "ALE/Pong-v5", make_agent=default_agent, num_steps=int(1e6))
plt.plot(agent_statistics.total_episodic_rewards)