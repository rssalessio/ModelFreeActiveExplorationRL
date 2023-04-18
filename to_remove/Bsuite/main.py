import bsuite
import numpy as np
from bsuite import sweep
from bsuite.utils import gym_wrapper
from agent import RandomAgent, QlearningAgent, Agent, Experience

print('All possible values for bsuite_id:', sweep.DEEP_SEA[0])
SAVE_PATH_RAND = './tmp/bsuite/'



def make_agent(env: gym_wrapper.GymFromDMEnv, type: str) -> Agent:
    if type == 'random':
        return RandomAgent(int(env.observation_space.shape[0] ** 2) , env.action_spec().num_values, 0.99)
    elif type == 'qlearning':
        return QlearningAgent(int(env.observation_space.shape[0] ** 2), env.action_spec().num_values, 0.99, 0.5, 0.3)

def run_random_agent(bsuite_id, agent_type: str):
    raw_env = bsuite.load_and_record(bsuite_id, save_path=f'{SAVE_PATH_RAND}{agent_type}', overwrite=True)
    env = gym_wrapper.GymFromDMEnv(raw_env)
    agent = make_agent(env, agent_type)
    
    t = 0
    for episode in range(env.bsuite_num_episodes):
        state = env.reset()
        done = False
        state = state.argmax()
        while not done:
            action = agent.forward(state, t)
            next_state, reward, done, info = env.step(action)
            next_state = next_state.argmax()
            exp = Experience(state, action, reward, next_state, done)
            agent.backward(exp)
            state = next_state
            t += 1
        
for bsuite_id in [sweep.DEEP_SEA[0], sweep.DEEP_SEA[1], sweep.DEEP_SEA[2], sweep.DEEP_SEA[3], sweep.DEEP_SEA[4]] :
  run_random_agent(bsuite_id, 'random')
  run_random_agent(bsuite_id, 'qlearning')