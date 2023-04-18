
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

class Network(nn.Module):
  def __init__(self, input_size: int, output_size: int, hidden_size: int = 32):
    self._feat_extractor = nn.Sequential(*[
        nn.Flatten(),
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU()
    ])
    
    self._Q_head = nn.Linear(hidden_size, output_size)
    self._M_head = nn.Sequential(*[nn.Linear(hidden_size, output_size), nn.ReLU()])
  
  def forward(self, x: torch.Tensor, require_M: bool = False) -> torch.Tensor:
    features = self._feat_extractor(x)
    q_values = self._Q_head(features)
    
    if require_M is False:
      return q_values
    
    m_values = self._M_head(features)
    return q_values, m_values
    

class ExplorativeAgent(base.Agent):
  """A simple DQN agent using TF2."""

  def __init__(
      self,
      action_spec: specs.DiscreteArray,
      network: Network,
      batch_size: int,
      discount: float,
      replay_capacity: int,
      min_replay_size: int,
      sgd_period: int,
      target_update_period: int,
      optimizer: optim.Optimizer,
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
    
    self._replay = replay.Replay(capacity=replay_capacity)
    self._ensemble_size = 5
    self.q_network = [copy.deepcopy(q_network) for i in range(self._ensemble_size)]
    self.target_q_network = [copy.deepcopy(q_network) for i in range(self._ensemble_size)]
    self.v_network = [copy.deepcopy(v_network) for i in range(self._ensemble_size)]
    self._total_steps = 0
    
    self._optimizer = [optim.Adam(self.q_network[idx].parameters(), lr = 1e-3) for idx in range(self._ensemble_size)]
    self._optimizer_v = [optim.Adam(self.v_network[idx].parameters(), lr = 5e-4) for idx in range(self._ensemble_size)]

  def select_action(self, timestep: dm_env.TimeStep) -> base.Action:

    # Epsilon-greedy policy.
    if self._rng.rand() < self._epsilon:
      return self._rng.randint(self._num_actions)

  
    observation = torch.from_numpy(timestep.observation[None, ...])
    
    idx = np.random.choice(self._ensemble_size)
    # Greedy policy, breaking ties uniformly at random.
    q_values = self.q_network[idx](observation)
    best_q, best_action = q_values.max(1)
    delta_squared = torch.clip(best_q.unsqueeze(1) - q_values, self._delta_min, np.infty)** 2
    
    idx = np.random.choice(self._ensemble_size)
    V = self.v_network[idx](observation)
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
    for idx in range(self._ensemble_size):
      transitions = self._replay.sample(self._batch_size)
      transitions2 = self._replay.sample(self._batch_size)
      self._training_step(transitions, transitions2, idx)


  def _training_step(self, transitions: Sequence[torch.Tensor], transitions2: Sequence[torch.Tensor], idx: int) -> float:
    """Does a step of SGD on a batch of transitions."""
    o_tm1, a_tm1, r_t, d_t, o_t = transitions
    r_t = torch.tensor(r_t, dtype=torch.float32)  # [B]
    d_t = torch.tensor(d_t, dtype=torch.float32)  # [B]
    o_tm1 = torch.tensor(o_tm1)
    o_t = torch.tensor(o_t)
    a_tm1 = torch.tensor(a_tm1)

    # Train Q Values
    with torch.no_grad():
      q_t = self.target_q_network[idx](o_t).max(1)[0]  # [B]
      target = r_t + d_t * self._discount * q_t
    
    qa_tm1 = self.q_network[idx](o_tm1).gather(1, a_tm1.unsqueeze(-1)).flatten()
    loss = nn.HuberLoss()(qa_tm1, target.detach()) * 0.5

    # Update the online network via SGD.
    self._optimizer[idx].zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.q_network[idx].parameters(), 1, norm_type = "inf")
    self._optimizer[idx].step()

    # Periodically copy online -> target network variables.
    if self._total_steps % self._target_update_period == 0:
      self.target_q_network[idx].load_state_dict(self.q_network[idx].state_dict())
    
    o_tm1, a_tm1, r_t, d_t, o_t = transitions2
    r_t = torch.tensor(r_t, dtype=torch.float32)  # [B]
    d_t = torch.tensor(d_t, dtype=torch.float32)  # [B]
    o_tm1 = torch.tensor(o_tm1)
    o_t = torch.tensor(o_t)
    a_tm1 = torch.tensor(a_tm1)
    # Train Var Network
    with torch.no_grad():
      q_values: torch.Tensor = self.target_q_network[idx](o_tm1)
      q_values = q_values.gather(1, a_tm1.unsqueeze(-1)).flatten()
      q_next_values: torch.Tensor = self.target_q_network[idx](o_t)
      W = (r_t + d_t * self._discount * q_next_values.max(1)[0] - q_values) / (self._discount)
      W_target = W ** 2
    
    v_tm1 = self.v_network[idx](o_tm1).gather(1, a_tm1.unsqueeze(-1)).flatten()
    loss_v = nn.HuberLoss()(v_tm1, W_target.detach())
    self._optimizer_v[idx].zero_grad()
    loss_v.backward()
    torch.nn.utils.clip_grad_norm_(self.v_network[idx].parameters(), 1, norm_type = "inf")
    self._optimizer_v[idx].step()
    
    return loss.item()
  
  


def default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray):
  """Initialize a DQN agent with default parameters."""
  #del obs_spec  # Unused.
  
  size_s = np.prod(obs_spec.shape)

  network = Network(size_s, action_spec.num_values, 32)
  optimizer = optim.Adam(network.parameters(), lr=1e-3)
  return ExplorativeAgent(
      action_spec=action_spec,
      network=network,
      batch_size=128,
      discount=0.99,
      replay_capacity=100000,
      min_replay_size=128,
      sgd_period=1,
      target_update_period=4,
      optimizer=optimizer,
      epsilon=0.05,
      delta_min=1e-3,
      seed=42)