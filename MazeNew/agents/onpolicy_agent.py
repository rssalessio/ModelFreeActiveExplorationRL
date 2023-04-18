from typing import Any, Callable, NamedTuple, Sequence, Optional
from numpy.typing import NDArray
import torch
import torch.nn as nn
import numpy as np
import copy
from .replay_buffer import ReplayBuffer
from .sequence_buffer import SequenceBuffer, Trajectory

golden_ratio_sq = ((1 + np.sqrt(5))/2) ** 2

class OnPolicyExplorativeAgent(object):
    """A simple DQN agent using TF2."""

    def __init__(
        self,
        state_dim: int,
        state_spec: NDArray[np.float64],
        num_actions: int,
        q_network: nn.Module,
        m_network: nn.Module,
        reward_network: nn.Module,
        policy_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        m_optimizer: torch.optim.Optimizer,
        policy_optimizer: torch.optim.Optimizer,
        sequence_length: int,
        batch_size: int,
        discount: float,
        replay_capacity: int,
        min_replay_size: int,
        sgd_period: int,
        target_update_period: int,
        epsilon_fn: Callable[[int], float] = lambda _: 0.05,
        delta_min: float = 1e-3,
        seed: Optional[int] = None,
    ):

        # Internalise hyperparameters.
        self._num_actions = num_actions
        self._state_dim = state_dim
        self._discount = discount
        self._batch_size = batch_size
        self._reward_network = reward_network
        self._sgd_period = sgd_period
        self._target_update_period = target_update_period
        self._epsilon_fn = epsilon_fn
        self._min_replay_size = min_replay_size
        self._delta_min = delta_min
        self._sequence_length = sequence_length
        self._state_spec = state_spec

        # Seed the RNG.
        torch.random.manual_seed(seed)
        self._rng = np.random.RandomState(seed)

        # Internalise the components (networks, optimizer, replay buffer).
        self._q_optimizer = q_optimizer
        self._m_optimizer = m_optimizer
        self._policy_optimizer = policy_optimizer
        self._replay = ReplayBuffer(capacity=replay_capacity)
        self._seq_buffer = SequenceBuffer(self._state_spec, self._sequence_length, discount)
        self._q_network = q_network
        self._policy_network = policy_network
        self._m_network = m_network
        self._q_target_network = copy.deepcopy(q_network)
        self._total_steps = 1

    @torch.no_grad()
    def select_action(self, observation: NDArray[np.float64], greedy: bool = False) -> int:
        # Epsilon-greedy policy.
        # if self._rng.rand() < self._epsilon_fn(self._total_steps):
        #     return self._rng.randint(self._num_actions)

        observation = torch.tensor(observation[None, ...], dtype=torch.float32)
        if greedy is False:
            p = self._policy_network(observation).numpy()[0]
            
            p = np.exp(p - p.max())
            p = p / p.sum(-1, keepdims=True)
            
            if self._total_steps % 200 == 0:
                print(f'{observation} - {p}')
                
            if np.any(np.isnan(p)):
                print(p)
                import pdb
                pdb.set_trace()
            
            return np.random.choice(self._num_actions, p=p)
        else:
            q = self._q_network(observation).numpy()[0]
            print(f'q: {q}')
            return q.argmax()

    def update(
        self,
        observation: NDArray[np.float64],
        action: int,
        reward: float,
        new_observation: NDArray[np.float64],
        done: bool) -> Optional[float]:
        
        with torch.no_grad():
            observation = torch.tensor(observation[None, ...], dtype=torch.float32)
            random_reward = self._reward_network(observation)[0][action]
        # Add this transition to replay.
        self._replay.add([
            observation,
            action,
            reward + random_reward,
            new_observation,
            done
        ])

        self._seq_buffer.append(
            observation, action, reward + random_reward, new_observation, done
        )

        self._total_steps += 1
        loss = 0
        if self._replay.size >= self._min_replay_size:        
            if self._total_steps % self._sgd_period == 0:
                transitions = self._replay.sample(self._batch_size)
                loss_q = self._training_step(transitions)
                loss += loss_q
                
        if self._seq_buffer.full() or done:
            loss_policy = self._policy_training_step(self._seq_buffer.drain())
            loss += loss_policy
        return loss
    
    
    @torch.no_grad()
    def _compute_advantage(self, trajectory: Trajectory) -> torch.Tensor:
        advantage = torch.zeros(trajectory.length, requires_grad=False, dtype=torch.float32)
        for i in range(trajectory.length):
            obs = trajectory.observations[i][None, ...]
            q = self._q_network(obs)[0]
            advantage[i] = q[trajectory.actions[i][0]]# - q.max(-1)[0]
        return advantage
    
    @torch.no_grad()
    def _compute_policy_generative(self, observation: torch.Tensor) -> NDArray:
        q_values = self._q_network(observation).numpy()[0]
        m_values = self._m_network(observation).numpy()[0]

        delta = q_values.max() - q_values
        idxs = np.isclose(delta, 0)
        
        delta_min = delta[~idxs].min()
        delta[idxs] = max(delta_min, self._delta_min)
        delta_sq = delta ** 2
        
        m_values = self._m_network(observation).numpy()[0]
        
        H = (2 + golden_ratio_sq * m_values) / delta_sq
        H[idxs] *= 1 / (1 - self._discount) ** 2
        H[idxs] = np.sqrt(H[idxs] * H[~idxs].sum())        
        
        return H / H.sum(-1, keepdims=True)

    def _policy_training_step(self, trajectory: Trajectory) -> float:
        advantage = self._compute_advantage(trajectory)
        if len(advantage) > 1:
            advantage = (advantage - advantage.mean()) / (1e-8 + advantage.std())
        loss = 0
        # import pdb
        # pdb.set_trace()
        
        for k in range(5):
            losses = []
            self._policy_optimizer.zero_grad()
            for i in range(trajectory.length):
                obs = trajectory.observations[i][None, ...]
                pi_s = self._policy_network(obs)[0]#[trajectory.actions[i]][0]
                # import pdb
                # pdb.set_trace()
                # TODO compute wg
                wg_s = self._compute_policy_generative(obs)[trajectory.actions[i]]
                # if np.random.uniform() < 0.005:
                #     print(pi_s.max())
                # pi_s = torch.exp(pi_s - pi_s.max())
                # pi_s = pi_s / pi_s.sum(-1, keepdims=True)
                #ratio =  / (1e-8 + wg_s)#torch.clip(pi_s / (1e-8 + wg_s), 1e-7, 1e7)
                # if self._total_steps  % 101 == 0:
                #     print(pi_s.max())
                
                ratio = pi_s[trajectory.actions[i]][0] - pi_s.max() - np.log(1e-8+wg_s)
                obj = ratio* (trajectory.cumulative_reward[i] - advantage[i])
                if torch.isnan(obj):
                    import pdb
                    pdb.set_trace()
                losses.append(obj)
            
            loss = torch.mean(torch.stack(losses))

            loss.backward()
            
            # for param in self._policy_network.parameters():
            #     if  not torch.isfinite(param.grad).all():
            #         import pdb
            #         pdb.set_trace()
                
            #torch.nn.utils.clip_grad.clip_grad_norm_(self._policy_network.parameters(), 1)
            self._policy_optimizer.step()
            
            # for param in self._policy_network.parameters():
            #     if  torch.isnan(param).any():
            #         print(param)
            #         import pdb
            #         pdb.set_trace()
        return loss.item()
            

    def _training_step(self, transitions: Sequence[NDArray]) -> float:
        """Does a step of SGD on a batch of transitions."""
        o_tm1, a_tm1, r_t, o_t, d_t = transitions

        
        a_tm1 = torch.tensor(a_tm1, dtype=torch.int64, requires_grad=False)
        r_t = torch.tensor(r_t, dtype=torch.float32, requires_grad=False)
        d_t = torch.tensor(d_t, dtype=torch.float32, requires_grad=False)
        o_tm1 = torch.tensor(o_tm1, dtype=torch.float32, requires_grad=False)
        o_t = torch.tensor(o_t, dtype=torch.float32, requires_grad=False)

        # Q training
        with torch.no_grad():
            q_t = self._q_target_network(o_t).gather(-1, self._q_network(o_t).argmax(-1).unsqueeze(-1)).flatten()
            target = r_t + (1-d_t) * self._discount * q_t
        
        self._q_optimizer.zero_grad()
        q_tm1 = self._q_network(o_tm1).gather(-1, a_tm1.unsqueeze(-1)).flatten()
        qloss = nn.HuberLoss()(q_tm1, target.detach())
        qloss.backward()
        #torch.nn.utils.clip_grad.clip_grad_norm_(self._q_network.parameters(), 1)
        self._q_optimizer.step()

        # Periodically copy online -> target network variables.
        if self._total_steps % self._target_update_period == 0:
            self._q_target_network.load_state_dict(self._q_network.state_dict())
        
        # M Training
        with torch.no_grad():
            qmax =  self._q_target_network(o_t).gather(-1, self._q_network(o_t).argmax(-1).unsqueeze(-1)).flatten()
            q = self._q_target_network(o_tm1).gather(-1, a_tm1.unsqueeze(-1)).flatten() 
            delta = r_t + (1-d_t) * self._discount * qmax - q
            m_target = (delta / self._discount) ** 2
        
        self._m_optimizer.zero_grad()
        mvalues = self._m_network(o_tm1).gather(-1, a_tm1.unsqueeze(-1)).flatten()
        mloss = nn.HuberLoss()(mvalues, m_target.detach())
        mloss.backward()
        #torch.nn.utils.clip_grad.clip_grad_norm_(self._m_network.parameters(), 1)
        self._m_optimizer.step() 
        
        return qloss.item() + mloss.item()

class TanhActivation(nn.Module):
    def __init__(self, max: float):
        super().__init__()
        self.max = torch.tensor(max, dtype=torch.float32, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.max * torch.tanh(x)# /2
        #return torch.maximum(-self.max, torch.minimum(x, self.max))

def default_agent(state_spec: NDArray[np.float64],
                  num_actions: int,
                  seed: int = 0) -> OnPolicyExplorativeAgent:
    """Initialize a DQN agent with default parameters."""
    
    def make_policy_network(input_size: int, output_size: int) -> nn.Module:
        return nn.Sequential(*[
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
            ])

    def make_Q_network(input_size: int, output_size: int) -> nn.Module:
        return nn.Sequential(*[
            nn.Flatten(),
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)])
        
    def make_M_network(input_size: int, output_size: int) -> nn.Module:
        return nn.Sequential(*[
            nn.Flatten(),
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
            nn.ReLU()])
        
    def make_reward_network(input_size: int, output_size: int) -> nn.Module:
        return nn.Sequential(*[
            nn.Flatten(),
            nn.Linear(input_size, 32),
            TanhActivation(0.2),
            nn.Linear(32, 32),
            TanhActivation(0.1),
            nn.Linear(32, output_size),
            TanhActivation(0.05)])
    
    state_dim = len(state_spec.flatten())

    q_network = make_Q_network(state_dim, num_actions)
    q_optim = torch.optim.Adam(q_network.parameters(), lr=1e-3)
    
    m_network = make_M_network(state_dim, num_actions)
    m_optim = torch.optim.Adam(m_network.parameters(), lr=1e-3)
    
    policy_network = make_policy_network(state_dim, num_actions)
    policy_optim = torch.optim.Adam(policy_network.parameters(), lr=1e-4)
    
    return OnPolicyExplorativeAgent(
        state_dim=state_dim,
        state_spec=state_spec,
        num_actions=num_actions,
        q_network=q_network,
        m_network=m_network,
        reward_network = make_reward_network(state_dim, num_actions),
        policy_network=policy_network,
        q_optimizer=q_optim,
        m_optimizer=m_optim,
        policy_optimizer=policy_optim,
        sequence_length=64,
        batch_size=128,
        discount=0.99,
        replay_capacity=20000,
        min_replay_size=100,
        sgd_period=1,
        target_update_period=64,
        epsilon_fn=lambda t: 10 / (10 + t),
        seed=seed,
    )