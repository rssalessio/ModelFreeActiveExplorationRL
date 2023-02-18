
import copy
from typing import Optional, Sequence

from bsuite.baselines import base
from bsuite.baselines.utils import replay

import dm_env
from dm_env import specs
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


golden_ratio = (1+np.sqrt(5))/2
golden_ratio_sq = golden_ratio ** 2

class ExplorativeAgent(base.Agent):
  """A simple DQN agent using TF2."""

  def __init__(
      self,
      action_spec: specs.DiscreteArray,
      q_network: nn.Module,
      v_network: nn.Module,
      batch_size: int,
      discount: float,
      replay_capacity: int,
      min_replay_size: int,
      sgd_period: int,
      target_update_period: int,
      optimizer: optim.Optimizer,
      optimizer_v: optim.Optimizer,
      epsilon: float,
      delta_min: float,
      seed: Optional[int] = None,
  ):

    # Internalise hyperparameters.
    self._num_actions = action_spec.num_values
    self._discount = discount
    self._batch_size = batch_size
    self._sgd_period = sgd_period
    self._target_update_period = target_update_period
    self._epsilon = epsilon
    self._min_replay_size = min_replay_size
    self._delta_min = delta_min

    # Seed the RNG.
    torch.random.manual_seed(seed)
    self._rng = np.random.RandomState(seed)

    # Internalise the components (networks, optimizer, replay buffer).
    self._optimizer = optimizer
    self._optimizer_v = optimizer_v
    self._replay = replay.Replay(capacity=replay_capacity)
    self.q_network = q_network
    self.target_q_network = copy.deepcopy(q_network)
    self.v_network = v_network
    self._total_steps = 0

  def select_action(self, timestep: dm_env.TimeStep) -> base.Action:

    # Epsilon-greedy policy.
    if self._rng.rand() < self._epsilon:
      return self._rng.randint(self._num_actions)

    observation = torch.from_numpy(timestep.observation[None, ...])
    
    # Greedy policy, breaking ties uniformly at random.
    q_values = self.q_network(observation)
    best_q, best_action = q_values.max(1)
    delta_squared = torch.clip(best_q.unsqueeze(1) - q_values, self._delta_min, np.infty)** 2
    
    
    V = self.v_network(observation)
    p = (2+8*golden_ratio*V) /delta_squared
    #p[:, best_action] = p[:, best_action] / (1-self._discount)**2

    p[:, best_action] = torch.sqrt(p[:, best_action]*(p.sum(-1)[:, None] -p[:, best_action]))
    p = (p/p.sum(1).unsqueeze(-1)).cumsum(axis=1)
    
    u = torch.rand(size=(p.shape[0],1))
    action = (u<p).max(axis=1)[1].item()#.detach().numpy()
  
    return int(action)

  def update(
      self,
      timestep: dm_env.TimeStep,
      action: base.Action,
      new_timestep: dm_env.TimeStep,
  ):
    # Add this transition to replay.
    self._replay.add([
        timestep.observation,
        action,
        new_timestep.reward,
        new_timestep.discount,
        new_timestep.observation,
    ])

    self._total_steps += 1
    if self._total_steps % self._sgd_period != 0 or self._replay.size < self._min_replay_size:
      return

    # Do a batch of SGD.
    transitions = self._replay.sample(self._batch_size)
    self._training_step(transitions)


  def _training_step(self, transitions: Sequence[torch.Tensor]) -> float:
    """Does a step of SGD on a batch of transitions."""
    o_tm1, a_tm1, r_t, d_t, o_t = transitions
    r_t = torch.tensor(r_t, dtype=torch.float32)  # [B]
    d_t = torch.tensor(d_t, dtype=torch.float32)  # [B]
    o_tm1 = torch.tensor(o_tm1)
    o_t = torch.tensor(o_t)
    a_tm1 = torch.tensor(a_tm1)

    # Train Q Values
    with torch.no_grad():
      q_t = self.target_q_network(o_t).max(1)[0]  # [B]
      target = r_t + d_t * self._discount * q_t
    
    qa_tm1 = self.q_network(o_tm1).gather(1, a_tm1.unsqueeze(-1)).flatten()
    loss = nn.HuberLoss()(qa_tm1, target.detach()) * 0.5

    # Update the online network via SGD.
    self._optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1, norm_type = "inf")
    self._optimizer.step()

    # Periodically copy online -> target network variables.
    if self._total_steps % self._target_update_period == 0:
      self.target_q_network.load_state_dict(self.q_network.state_dict())
      
    # Train Var Network
    with torch.no_grad():
      q_values: torch.Tensor = self.target_q_network(o_tm1)
      q_values = q_values.gather(1, a_tm1.unsqueeze(-1)).flatten()
      q_next_values: torch.Tensor = self.target_q_network(o_t)
      W = (r_t + d_t * self._discount * q_next_values.max(1)[0] - q_values) / (self._discount)
      W_target = W ** 2
    
    v_tm1 = self.v_network(o_tm1).gather(1, a_tm1.unsqueeze(-1)).flatten()
    loss_v = nn.HuberLoss()(v_tm1, W_target.detach())
    self._optimizer_v.zero_grad()
    loss_v.backward()
    torch.nn.utils.clip_grad_norm_(self.v_network.parameters(), 1, norm_type = "inf")
    self._optimizer_v.step()
    
    return loss.item()

def default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray):
  """Initialize a DQN agent with default parameters."""
  #del obs_spec  # Unused.
  
  size_s = np.prod(obs_spec.shape)
  network = nn.Sequential(*[
      nn.Flatten(),
      nn.Linear(size_s, 50),
      nn.ReLU(),
      # nn.Linear(50, 50),
      # nn.ReLU(),
      nn.Linear(50, action_spec.num_values),
  ])
  
  v_network = nn.Sequential(*[
      nn.Flatten(),
      nn.Linear(size_s, 50),
      nn.ReLU(),
      # nn.Linear(50, 50),
      # nn.ReLU(),
      nn.Linear(50, action_spec.num_values),
      nn.ReLU()
  ])

  optimizer = optim.Adam(network.parameters(), lr=1e-3)
  optimizer_v = optim.Adam(v_network.parameters(), lr=5e-4)
  return ExplorativeAgent(
      action_spec=action_spec,
      q_network=network,
      v_network=v_network,
      batch_size=32,
      discount=0.99,
      replay_capacity=10000,
      min_replay_size=100,
      sgd_period=1,
      target_update_period=4,
      optimizer=optimizer,
      optimizer_v=optimizer_v,
      epsilon=0.05,
      delta_min=1e-3,
      seed=42)