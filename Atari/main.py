import gymnasium as gym
from gymnasium.wrappers import StepAPICompatibility, AtariPreprocessing, AutoResetWrapper, FrameStack
from wrappers.clip_reward import ClipReward
from agents.networks import make_single_network

def run(env_name: str, num_steps: int = 1000, seed: int = 42):
    env = gym.make(env_name,  frameskip=1)#render_mode="human",
    env = AtariPreprocessing(AutoResetWrapper(ClipReward(env)), scale_obs=True)
    env = StepAPICompatibility(FrameStack(env, num_stack=4), False)
    env.action_space.seed(seed)

    cnn = make_single_network(env.observation_space, 4, 53, ensemble_size=5)

    observation, info = env.reset(seed=seed)

    for t in range(num_steps):
        observation, reward, done, info = env.step(env.action_space.sample())
    env.close()

run(env_name = "ALE/Pong-v5")