
import copy
from typing import Optional, Sequence, Callable

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

def make_linear_layer(input_size: int, output_size: int):
  layer = nn.Linear(input_size, output_size)
  #torch.nn.init.xavier_normal_(layer.weight)
  #torch.nn.init.xavier_normal_(layer.bias)
  return layer

class Network(nn.Module):
  def __init__(self, input_size: int, output_size: int, hidden_size: int = 32):
    super().__init__()
    self._feat_extractor = nn.Sequential(*[
        nn.Flatten(),
        make_linear_layer(input_size, hidden_size),
        nn.ReLU(),
        make_linear_layer(hidden_size, hidden_size),
        nn.ReLU()
    ])
    
    self._Q_head = make_linear_layer(hidden_size, output_size)
    self._M_head = nn.Sequential(*[make_linear_layer(hidden_size, output_size), nn.ReLU()])
  
  def forward(self, x: torch.Tensor, require_M: bool = False) -> torch.Tensor:
    features = self._feat_extractor(x)
    q_values = self._Q_head(features)
    
    if require_M is False:
      return q_values
    
    m_values = self._M_head(features.detach())
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
      delta_min: float,
      epsilon_fn: Callable[[int], float] = lambda _: 0.,
      seed: Optional[int] = None,
  ):

    # Internalise hyperparameters.
    self._num_actions = action_spec.num_values
    self._discount = discount
    self._batch_size = batch_size
    self._sgd_period = sgd_period
    self._target_update_period = target_update_period
    self._epsilon_fn = epsilon_fn
    self._min_replay_size = min_replay_size
    self._delta_min = delta_min

    # Seed the RNG.
    torch.random.manual_seed(seed)
    self._rng = np.random.RandomState(seed)

    # Internalise the components (networks, optimizer, replay buffer).
    
    self._replay = replay.Replay(capacity=replay_capacity)
    self._network = network
    self._target_network = copy.deepcopy(network)
    self._total_steps = 0
    self._optimizer = optimizer

  @torch.no_grad()
  def select_action(self, timestep: dm_env.TimeStep) -> base.Action:

    # Epsilon-greedy policy.
    if self._rng.rand() < self._epsilon_fn(self._total_steps):
      return self._rng.randint(self._num_actions)

  
    observation = torch.from_numpy(timestep.observation[None, ...])
  
    # Greedy policy, breaking ties uniformly at random.
    q_values, V = self._network.forward(observation, require_M=True)

    delta = (q_values[0].max() - q_values[0]).numpy() #, self._delta_min, np.infty)
    idxs = np.isclose(delta,0)
    
    V = V[0].numpy()
    try:
      delta[idxs] = np.clip(delta[~idxs].min(), self._delta_min, np.inf) * (1-self._discount)
    except Exception:
      import pdb
      pdb.set_trace()
    
    H = (2 + 8 * golden_ratio_sq * V) /  (delta ** 2)
    #idxs = np.isclose(H.max() - H, 0)
    H[idxs] = np.sqrt(H[idxs] * H[~idxs].sum())
    
    p = H / H.sum(-1, keepdims=True)   
    if np.random.uniform()<1e-2:
      print(p)
    
    # delta_squared = torch.clip(best_q.unsqueeze(1) - q_values, self._delta_min, np.infty)** 2
    
    # p = (2+8*golden_ratio*V) /delta_squared
    # #p[:, best_action] = p[:, best_action] / (1-self._discount)**2

    # p[:, best_action] = torch.sqrt(p[:, best_action]*(p.sum(-1)[:, None] -p[:, best_action]))
    # p = (p/p.sum(1).unsqueeze(-1)).cumsum(axis=1)
    
    # u = torch.rand(size=(p.shape[0],1))
    # action = (u<p).max(axis=1)[1].item()#.detach().numpy()
  
    return int(np.random.choice(self._num_actions, p = p))

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

    
    if self._total_steps % self._sgd_period != 0 or self._replay.size < self._min_replay_size:
      return

    transitions = self._replay.sample(self._batch_size)
    self._training_step(transitions)
    self._total_steps += 1


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
      q_t = self._target_network.forward(o_t, require_M=False).max(1)[0]  # [B]
      target = r_t + d_t * self._discount * q_t
    
    # Update the online network via SGD.
    self._optimizer.zero_grad()
    
    
    qa_tm1 = self._network.forward(o_tm1, require_M=False).gather(1, a_tm1.unsqueeze(-1)).flatten()
    loss_q: torch.Tensor = nn.HuberLoss()(qa_tm1, target.detach()) * 0.5    

    # Train Var Network
    with torch.no_grad():
      q_values: torch.Tensor = self._target_network.forward(o_tm1, require_M=False)
      q_values = q_values.gather(1, a_tm1.unsqueeze(-1)).flatten()
      q_next_values: torch.Tensor = self._target_network.forward(o_t, require_M=False)
      W = (r_t + d_t * self._discount * q_next_values.max(1)[0] - q_values) / (self._discount)
      W_target = W ** 2
    
    v_tm1 = self._network.forward(o_tm1, require_M=True)[0].gather(1, a_tm1.unsqueeze(-1)).flatten()
    loss_v: torch.Tensor = nn.HuberLoss()(v_tm1, W_target.detach())


    loss = loss_q + loss_v
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self._network.parameters(), 1, norm_type = "inf")
    self._optimizer.step()
    
    # Periodically copy online -> target network variables.
    if self._total_steps % self._target_update_period == 0:
      self._target_network.load_state_dict(self._network.state_dict())
    
    return loss.item()
  
  


def default_agent(obs_spec: specs.Array,
                  action_spec: specs.DiscreteArray):
  """Initialize a DQN agent with default parameters."""
  #del obs_spec  # Unused.
  
  size_s = np.prod(obs_spec.shape)

  network = Network(size_s, action_spec.num_values, 50)
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
      epsilon_fn=lambda t: 10 / (10 + t),
      delta_min=0.3,
      seed=42)