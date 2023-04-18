from typing import NamedTuple

class Parameters(NamedTuple):
    env_type: str
    gamma: float
    river_length: int
    horizon: int
    min_reward: float
    max_reward_1: float
    max_reward_2: float
    n_sims: int
    frequency_computation: int
    alpha: float
    eta1: float
    eta2: float