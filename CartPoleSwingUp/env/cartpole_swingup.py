# python3
# pylint: disable=g-bad-file-header
# This file was modified from the BSuite repository.
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# ============================================================================

"""A swing up experiment in Cartpole."""
import numpy as np
from numpy.typing import NDArray
from .cartpole import  TimeStep, CartpoleState, CartpoleConfig, step_cartpole
from typing import NamedTuple


# SETTINGS = tuple({'height_threshold': n / 20, 'x_reward_threshold': 1 - n / 20}
#                  for n in range(20))

class CartpoleSwingupConfig(NamedTuple):
  height_threshold: float = 0.5
  theta_dot_threshold: float = 1.
  x_reward_threshold: float = 1.
  move_cost: float = 0.1
  x_threshold: float = 3.
  timescale: float = 0.01 #0.01
  max_time: float = 10. #10.
  init_range: float = 0.05

class CartpoleSwingup(object):
  """A difficult 'swing up' version of the classic Cart Pole task.

  In this version of the problem the pole begins downwards, and the agent must
  swing the pole up in order to see reward. Unlike the typical cartpole task
  the agent must pay a cost for moving, which aggravates the explore-exploit
  tradedoff. Algorithms without 'deep exploration' will simply remain still.
  """

  def __init__(self,
               config: CartpoleSwingupConfig,
               seed: int = None):
    # Setup.
    self._state: CartpoleState = CartpoleState(0, 0, 0, 0, 0)
    self._config = config
    self._rng = np.random.RandomState(seed)
    self._init_fn = lambda: self._rng.uniform(low=-config.init_range, high=config.init_range)


    # Reward/episode logic
    self._height_threshold = config.height_threshold
    self._theta_dot_threshold = config.theta_dot_threshold
    self._x_reward_threshold = config.x_reward_threshold
    self._move_cost = config.move_cost
    self._x_threshold = config.x_threshold
    self._timescale = config.timescale
    self._max_time = config.max_time

    # Problem config
    self._cartpole_config = CartpoleConfig(
        mass_cart=1.,
        mass_pole=0.1,
        length=0.5,
        force_mag=10.,
        gravity=9.8,
    )

  def reset(self) -> NDArray[np.float32]:
    self._reset_next_step = False
    self._state = CartpoleState(
        x=self._init_fn(),
        x_dot=self._init_fn(),
        theta=np.pi + self._init_fn(),
        theta_dot=self._init_fn(),
        time_elapsed=0.,
    )
    self._episode_return = 0.
    return self.observation

  def step(self, action: int) -> TimeStep:
    current_observation = self.observation.copy()
    if self._reset_next_step:
      raise ValueError('Error! Did you forget to reset the environment at the end of the episode?')

    self._state = step_cartpole(
        action=action,
        timescale=self._timescale,
        state=self._state,
        config=self._cartpole_config,
    )

    # Rewards only when the pole is central and balanced
    is_upright = (np.cos(self._state.theta) > self._height_threshold
                  and np.abs(self._state.theta_dot) < self._theta_dot_threshold
                  and np.abs(self._state.x) < self._x_reward_threshold)
    reward = -1. * np.abs(action - 1) * self._move_cost

    if is_upright:
      reward += 1.

    done = (self._state.time_elapsed > self._max_time or np.abs(self._state.x) > self._x_threshold)

    self._reset_next_step = done
    return TimeStep(current_observation, action, reward, done, self.observation)

  @property
  def num_actions(self) -> int:
    return 3
  
  @property
  def dim_state_space(self) -> int:
    return 8

  @property
  def observation(self) -> NDArray[np.float32]:
    """Approximately normalize output."""
    obs = np.zeros((1,8), dtype=np.float32)
    obs[0,0] = self._state.x / self._x_threshold
    obs[0,1] = self._state.x_dot / self._x_threshold
    obs[0,2] = np.sin(self._state.theta)
    obs[0,3] = np.cos(self._state.theta)
    obs[0,4] = self._state.theta_dot
    obs[0,5] = self._state.time_elapsed / self._max_time
    obs[0,6] = 1. if np.abs(self._state.x) < self._x_reward_threshold else -1.
    theta_dot = self._state.theta_dot
    obs[0,7] = 1. if np.abs(theta_dot) < self._theta_dot_threshold else -1.
    return obs

