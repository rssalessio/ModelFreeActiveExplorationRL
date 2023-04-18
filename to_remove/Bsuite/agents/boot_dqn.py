import copy
from typing import Callable, NamedTuple, Optional, Sequence
from .replay_buffer import ReplayBuffer
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from itertools import chain
import dm_env
from dm_env import specs

class BootstrappedDqn(object):
    """Bootstrapped DQN with additive prior functions."""
    def __init__(
            self,
            state_dim: int,
            num_actions: int,
            ensemble: Sequence[nn.Module],
            prior_ensemble: Sequence[nn.Module],
            batch_size: int,
            discount: float,
            replay_capacity: int,
            min_replay_size: int,
            sgd_period: int,
            target_update_period: int,
            optimizer: torch.optim.Optimizer,
            mask_prob: float,
            noise_scale: float,
            epsilon_fn: Callable[[int], float] = lambda _: 0.,
            prior_scale: float = 3,
            seed: Optional[int] = None):
        """Bootstrapped DQN with additive prior functions."""
        # Agent components.
        self._state_dim = state_dim
        self._ensemble = ensemble
        self._prior_ensemble = prior_ensemble
        
        self._target_ensemble = [copy.deepcopy(network) for network in ensemble]

        self._num_ensemble = len(ensemble)
        self._optimizer = optimizer
        self._replay = ReplayBuffer(capacity=replay_capacity)
        self._prior_scale = prior_scale

        # Agent hyperparameters.
        self._num_actions = num_actions
        self._batch_size = batch_size
        self._sgd_period = sgd_period
        self._target_update_period = target_update_period
        self._min_replay_size = min_replay_size
        self._epsilon_fn = epsilon_fn
        self._mask_prob = mask_prob
        self._noise_scale = noise_scale
        self._rng = np.random.RandomState(seed)
        self._discount = discount

        # Agent state.
        self._total_steps = 1
        self._active_head = 0
        torch.random.manual_seed(seed)
        
    def _forward(self, x: torch.Tensor, head: int, target: bool = False) -> torch.Tensor:
        net = self._ensemble[head] if target is False else self._target_ensemble[head]
        return net(x) + self._prior_scale * self._prior_ensemble[head](x).detach()

    def _step(self, transitions: Sequence[torch.Tensor]):
        """Does a step of SGD for the whole ensemble over `transitions`."""
        o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t = transitions
        a_tm1 = torch.tensor(a_tm1, dtype=torch.int64, requires_grad=False)
        r_t = torch.tensor(r_t, dtype=torch.float32, requires_grad=False)
        d_t = torch.tensor(d_t, dtype=torch.float32, requires_grad=False)
        o_tm1 = torch.tensor(o_tm1, dtype=torch.float32, requires_grad=False)
        o_t = torch.tensor(o_t, dtype=torch.float32, requires_grad=False)
        m_t = torch.tensor(m_t, dtype=torch.float32, requires_grad=False)
        z_t = torch.tensor(z_t, dtype=torch.float32, requires_grad=False)
        # if np.random.uniform() < 1e-2:
        #     import pdb
        #     pdb.set_trace()
        losses = []
        self._optimizer.zero_grad()
        for k in range(self._num_ensemble):
            
            with torch.no_grad():
                q_target = self._forward(o_t, k, target=True).max(-1)[0]
                target_y = r_t + z_t[:, k] + self._discount * (1-d_t) * q_target
            
            q_values = self._forward(o_tm1, k, target=False).gather(-1, a_tm1.unsqueeze(-1)).flatten()
            losses.append((torch.square(q_values - target_y.detach()) * m_t[:, k]))
        
        loss = torch.mean(torch.stack(losses))
        loss.backward()
        self._optimizer.step()
        
        if np.random.uniform() < 0.002:
            print(f'Loss: {loss.item()}')
        self._total_steps += 1

        # Periodically update the target network.
        if self._total_steps % self._target_update_period == 0:
            for k in range(self._num_ensemble):
                self._target_ensemble[k].load_state_dict(self._ensemble[k].state_dict())
        return loss.item()#np.mean(losses)

    @torch.no_grad()
    def _select_action(self, observation: NDArray[np.float64]) -> int:
        if self._rng.rand() < self._epsilon_fn(self._total_steps):
            return self._rng.randint(self._num_actions)
        
        observation = torch.tensor(observation[None, ...], dtype=torch.float32)
        # Greedy policy, breaking ties uniformly at random.
        q_values = self._forward(observation, self._active_head, target=False).numpy()[0]
        idxs = np.isclose(q_values.max() - q_values, 0)
        action = self._rng.choice(idxs)
        return int(action)
    
    def select_action(self, timestep: dm_env.TimeStep):
        return self._select_action(timestep.observation)      
    
    def update(
        self,
        timestep: dm_env.TimeStep,
        action: int,
        new_timestep: dm_env.TimeStep) -> None:
        return self._update(
            timestep.observation, action, np.float32(new_timestep.reward),
            np.float32(new_timestep.observation), 1-new_timestep.discount)
    
    def _update(
            self,
            observation: NDArray[np.float64],
            action: int,
            reward: float,
            new_observation: NDArray[np.float64],
            done: bool) -> Optional[float]:
        """Update the agent: add transition to replay and periodically do SGD."""
        if done:
            self._active_head = self._rng.randint(self._num_ensemble)
            print(f'New head {self._active_head} - Done: {done} - steps {self._total_steps}')

        self._replay.add(
            TransitionWithMaskAndNoise(
                o_tm1=observation,
                a_tm1=action,
                r_t=np.float32(reward),
                d_t=np.float32(done),
                o_t=new_observation,
                m_t=self._rng.binomial(1, self._mask_prob,
                                    self._num_ensemble).astype(np.float32),
                z_t=self._rng.randn(self._num_ensemble).astype(np.float32) *
                self._noise_scale,
            ))

        if self._replay.size < self._min_replay_size:
            return None

        if self._total_steps % self._sgd_period != 0:
            return None
        minibatch = self._replay.sample(self._batch_size)
        return self._step(minibatch)


class TransitionWithMaskAndNoise(NamedTuple):
    o_tm1: NDArray[np.float64]
    a_tm1: int
    r_t: float
    d_t: float
    o_t: NDArray[np.float64]
    m_t: NDArray[np.int64]
    z_t: NDArray[np.float64]
    
    
def _make_linear(input: int, output:int):
    layer1 = nn.Linear(input, output)
    nn.init.orthogonal_(layer1.weight)
    nn.init.zeros_(layer1.bias)
    # std = 1 / np.sqrt(input)
    # torch.nn.init.trunc_normal_(layer1.weight, std=std, a=-3*std, b=3*std)
    return layer1     
        

def make_single_network(input_size: int, output_size: int) -> nn.Module:
    return nn.Sequential(*[
        nn.Flatten(),
        _make_linear(input_size, 20),
        nn.ReLU(),
        _make_linear(20, 20),
        nn.ReLU(),
        _make_linear(20, output_size)])
  
  
def default_agent(
        obs_spec: specs.Array,
        action_spec: specs.DiscreteArray,
        num_ensemble: int = 10,
        prior_scale: float = 20,
        seed: int = 42) -> BootstrappedDqn:
    """Initialize a Bootstrapped DQN agent with default parameters."""

    state_dim = np.prod(obs_spec.shape)
    num_actions = action_spec.num_values
    ensemble = [make_single_network(state_dim, num_actions) for i in range(num_ensemble)]
    prior_ensemble = [make_single_network(state_dim, num_actions) for i in range(num_ensemble)]
    
    
    
    for net in prior_ensemble:
        for layer in net.children():
            for par in layer.parameters():
                par.requires_grad = False
    
    parameters = []
    for net in ensemble:
        parameters += list(net.parameters())
    
    optimizer = torch.optim.Adam(parameters, lr=1e-3)

    return BootstrappedDqn(
        state_dim=state_dim,
        num_actions=num_actions,
        ensemble=ensemble,
        prior_ensemble=prior_ensemble,
        prior_scale=prior_scale,
        batch_size=256,
        discount=.99,
        replay_capacity=100000,
        min_replay_size=2000,
        sgd_period=8,
        target_update_period=32,
        optimizer=optimizer,
        mask_prob=.5,
        noise_scale=0.0,
        epsilon_fn=lambda t: 10 / (10 + t),
        seed=seed,
    )