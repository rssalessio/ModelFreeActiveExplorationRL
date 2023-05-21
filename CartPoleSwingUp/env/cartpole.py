# python3
# This file was modified from the BSuite repository, Copyright 2019
# DeepMind Technologies Limited. 
# 
# The file was originally Licensed under the Apache License, Version 2.0
# (the "License"), which  can be found in the root folder of this repository.
# Changes and additions are licensed under the MIT license, 
# Copyright (c) [2023] [NeurIPS authors, 11410] (see the LICENSE file
# in the project root for full information)
# ============================================================================
"""The Cartpole reinforcement learning environment."""

import collections
import numpy as np
from typing import NamedTuple
from numpy.typing import NDArray

class TimeStep(NamedTuple):
    observation: NDArray[np.float32]
    action: int
    reward: float
    done: bool
    next_observation: NDArray[np.float32]

class CartpoleState(NamedTuple):
  x: float
  x_dot: float
  theta: float
  theta_dot: float
  time_elapsed: float

class CartpoleConfig(NamedTuple):
  mass_cart: float
  mass_pole: float
  length: float
  force_mag: float
  gravity: float



def step_cartpole(action: int,
                  timescale: float,
                  state: CartpoleState,
                  config: CartpoleConfig) -> CartpoleState:
  """Helper function to step cartpole state under given config."""
  # Unpack variables into "short" names for mathematical equation
  force = (action - 1) * config.force_mag
  cos = np.cos(state.theta)
  sin = np.sin(state.theta)
  pl = config.mass_pole * config.length
  l = config.length
  m_pole = config.mass_pole
  m_total = config.mass_cart + config.mass_pole
  g = config.gravity

  # Compute the physical evolution
  temp = (force + pl * state.theta_dot**2 * sin) / m_total
  theta_acc = (g * sin - cos * temp) / (l * (4/3 - m_pole * cos**2 / m_total))
  x_acc = temp - pl * theta_acc * cos / m_total

  # Update states according to discrete dynamics
  x = state.x + timescale * state.x_dot
  x_dot = state.x_dot + timescale * x_acc
  theta = np.remainder(
      state.theta + timescale * state.theta_dot, 2 * np.pi)
  theta_dot = state.theta_dot + timescale * theta_acc
  time_elapsed = state.time_elapsed + timescale

  return CartpoleState(x, x_dot, theta, theta_dot, time_elapsed)


class Cartpole(object):
  """This implements a version of the classic Cart Pole task.

  For more information see:
  https://webdocs.cs.ualberta.ca/~sutton/papers/barto-sutton-anderson-83.pdf
  The observation is a vector representing:
    `(x, x_dot, sin(theta), cos(theta), theta_dot, time_elapsed)`

  The actions are discrete ['left', 'stay', 'right']. Episodes start with the
  pole close to upright. Episodes end when the pole falls, the cart falls off
  the table, or the max_time is reached.
  """

  def __init__(self,
               height_threshold: float = 0.8,
               x_threshold: float = 3.,
               timescale: float = 0.01,
               max_time: float = 10.,
               init_range: float = 0.05,
               seed: int = None):
    # Setup.
    self._state = CartpoleState(0, 0, 0, 0, 0)
    self._rng = np.random.RandomState(seed)
    self._init_fn = lambda: self._rng.uniform(low=-init_range, high=init_range)

    # Reward/episode logic
    self._height_threshold = height_threshold
    self._x_threshold = x_threshold
    self._timescale = timescale
    self._max_time = max_time

    # Problem config
    self._cartpole_config = CartpoleConfig(
        mass_cart=1.,
        mass_pole=0.1,
        length=0.5,
        force_mag=10.,
        gravity=9.8,
    )

  # Overrides the super method.
  def reset(self) -> NDArray[np.float64]:
    self._reset_next_step = False
    self._state = CartpoleState(
        x=self._init_fn(),
        x_dot=self._init_fn(),
        theta=self._init_fn(),
        theta_dot=self._init_fn(),
        time_elapsed=0.,
    )
    self._episode_return = 0
    return self.observation

  # Overrides the super method (we implement special auto-reset behavior here).
  def step(self, action):
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
    is_reward = (np.cos(self._state.theta) > self._height_threshold
                 and np.abs(self._state.x) < self._x_threshold)
    reward = 1. if is_reward else 0.

    done = self._state.time_elapsed > self._max_time or not is_reward
    self._reset_next_step = done
  
    return TimeStep(current_observation, action, reward, done, self.observation)

  @property
  def num_actions(self) -> int:
    return 3
  
  @property
  def dim_state_space(self) -> int:
    return 6

  @property
  def observation(self) -> np.ndarray:
    """Approximately normalize output."""
    obs = np.zeros(6, dtype=np.float32)
    obs[0] = self._state.x / self._x_threshold
    obs[1] = self._state.x_dot / self._x_threshold
    obs[2] = np.sin(self._state.theta)
    obs[3] = np.cos(self._state.theta)
    obs[4] = self._state.theta_dot
    obs[5] = self._state.time_elapsed / self._max_time
    return obs

