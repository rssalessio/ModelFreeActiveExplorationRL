# pylint: disable=g-bad-file-header
# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""A simple implementation of Bootstrapped DQN with prior networks.

References:
1. "Deep Exploration via Bootstrapped DQN" (Osband et al., 2016)
2. "Deep Exploration via Randomized Value Functions" (Osband et al., 2017)
3. "Randomized Prior Functions for Deep RL" (Osband et al, 2018)

Links:
1. https://arxiv.org/abs/1602.04621
2. https://arxiv.org/abs/1703.07608
3. https://arxiv.org/abs/1806.03335

Notes:

- This agent is implemented with TensorFlow 2 and Sonnet 2. For installation
  instructions for these libraries, see the README.md in the parent folder.
- This implementation is potentially inefficient, as it does not parallelise
  computation across the ensemble for simplicity and readability.
"""

import copy
from typing import Callable, NamedTuple, Optional, Sequence
from .replay_buffer import ReplayBuffer
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from itertools import chain


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
        
        # self._forward_ensemble: Sequence[Callable[[torch.Tensor], torch.Tensor]] = [
        #     lambda x: net(x) + prior_scale * prior_net(x).detach() for net, prior_net in zip(self._ensemble, self._prior_ensemble)]
        # self._target_forward_ensemble: Sequence[Callable[[torch.Tensor], torch.Tensor]]  = [
        #     lambda x: net(x) + prior_scale * prior_net(x).detach() for net, prior_net in zip(self._target_ensemble, self._prior_ensemble)]
        
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
        
        
        #losses = []
        self._optimizer.zero_grad()
        loss = 0
        for k in range(self._num_ensemble):
            # net = self._forward_ensemble[k]
            # target_net = self._target_forward_ensemble[k]
            
            with torch.no_grad():
                q_target = self._forward(o_t, k, target=True).max(-1)[0]
                target_y = (r_t + z_t[:, k] + self._discount * (1-d_t) * q_target) * m_t[:,k]
                target_y = target_y * m_t[:,k]
            # Q-learning loss with added reward noise + half-in bootstrap.
            
            q_values = self._forward(o_tm1, k, target=False).gather(-1, a_tm1.unsqueeze(-1)).flatten() * m_t[:,k]
            loss = loss+ nn.HuberLoss()(q_values, target_y.detach())
            if loss.item() > 1e2:
                import pdb
                pdb.set_trace()
        
        loss.backward()
        for k in range(self._num_ensemble):
            torch.nn.utils.clip_grad.clip_grad_norm_(self._ensemble[k].parameters(), 1.)
        
        self._optimizer.step() 
        # if self._total_steps % 500 == 0:
        #     print(q_target)
        # loss = torch.mean(torch.stack(losses))
        # loss.backward()
        # self._optimizer.step()
        
        self._total_steps += 1

        # Periodically update the target network.
        if self._total_steps % self._target_update_period == 0:
            for k in range(self._num_ensemble):
                self._target_ensemble[k].load_state_dict(self._ensemble[k].state_dict())
        return loss.item()#np.mean(losses)

    @torch.no_grad()
    def select_action(self, observation: NDArray[np.float64]) -> int:
        if self._rng.rand() < self._epsilon_fn(self._total_steps):
            return self._rng.randint(self._num_actions)
        
        observation = torch.tensor(observation[None, ...], dtype=torch.float32)
        # Greedy policy, breaking ties uniformly at random.
        q_values = self._forward(observation, self._active_head, target=False).numpy()[0]
        idxs = np.isclose(q_values.max() - q_values, 0)
        action = self._rng.choice(idxs)
        return int(action)
    
    def update(
            self,
            observation: NDArray[np.float64],
            action: int,
            reward: float,
            new_observation: NDArray[np.float64],
            done: bool) -> Optional[float]:
        """Update the agent: add transition to replay and periodically do SGD."""
        if done:
            self._active_head = self._rng.randint(self._num_ensemble)

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


def make_single_network(input_size: int, output_size: int) -> nn.Module:
    return nn.Sequential(*[
        nn.Flatten(),
        nn.Linear(input_size, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, output_size)])


def default_agent(
        state_dim: int,
        num_actions: int,
        num_ensemble: int = 20,
        prior_scale: float = 3,
        seed: int = 42) -> BootstrappedDqn:
    """Initialize a Bootstrapped DQN agent with default parameters."""

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
        batch_size=128,
        discount=.99,
        replay_capacity=10000,
        min_replay_size=128,
        sgd_period=1,
        target_update_period=32,
        optimizer=optimizer,
        mask_prob=0.5,
        noise_scale=0.0,
        epsilon_fn=lambda t: 10 / (10 + t),
        seed=seed,
    )