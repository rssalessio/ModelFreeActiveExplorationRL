
import copy
from typing import Optional, Sequence, Callable, NamedTuple
from bsuite.baselines.utils import replay
from .agent import TimeStep, Agent
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from numpy.typing import NDArray
from .ensemble_linear_layer import EnsembleLinear

golden_ratio = (1+np.sqrt(5))/2
golden_ratio_sq = golden_ratio ** 2

class EnsembleWithPrior(nn.Module):
    def __init__(self, input_size: int, output_size: int, prior_scale: float, ensemble_size: int, final_activation: bool, hidden_size: int = 32):
        super().__init__()
        
        self.ensemble_size = ensemble_size
        self._network = make_single_network(input_size, output_size, hidden_size, ensemble_size, final_activation)
        self._prior_network = make_single_network(input_size, output_size, hidden_size, ensemble_size, final_activation)
        self._prior_scale = prior_scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[None, ...].repeat(self.ensemble_size, 1, 1, 1)
        values = self._network.forward(x).swapaxes(0,1)
        prior_values = self._prior_network(x).swapaxes(0,1)
        return values + self._prior_scale * prior_values.detach()

class ExplorativeAgent(Agent):
    """A simple DQN agent using TF2."""

    def __init__(
        self,
        num_actions: int,
        Q_network: EnsembleWithPrior,
        M_network: EnsembleWithPrior,
        batch_size: int,
        discount: float,
        replay_capacity: int,
        min_replay_size: int,
        sgd_period: int,
        target_update_period: int,
        Q_optimizer: optim.Optimizer,
        M_optimizer: optim.Optimizer,
        mask_prob: float = .5,
        noise_scale: float = 0.0,
        delta_min: Callable[[int], float] = lambda _: 1e-3,
        epsilon_fn: Callable[[int], float] = lambda _: 0.,
        seed: Optional[int] = None,
    ):

        # Internalise hyperparameters.
        self._num_actions = num_actions
        self._discount = discount
        self._batch_size = batch_size
        self._sgd_period = sgd_period
        self._target_update_period = target_update_period
        self._epsilon_fn = epsilon_fn
        self._min_replay_size = min_replay_size
        self._delta_min = delta_min
        self._mask_prob = mask_prob
        self._noise_scale= noise_scale

        # Seed the RNG.
        torch.random.manual_seed(seed)
        self._rng = np.random.RandomState(seed)

        # Internalise the components (networks, optimizer, replay buffer).
        
        self._replay = replay.Replay(capacity=replay_capacity)
        self._ensemble_Q_network = Q_network
        self._ensemble_M_network = M_network
        self._ensemble_Q_target_network = copy.deepcopy(Q_network)
        self._total_steps = 0
        self._Q_optimizer = Q_optimizer
        self._M_optimizer = M_optimizer
        self._num_ensemble = Q_network.ensemble_size
        
        self._active_head = 0


    @torch.no_grad()
    def select_greedy_action(self, observation: NDArray[np.float32]) -> int:
        observation = torch.from_numpy(observation[None, ...])
        return self._ensemble_Q_network(observation)[0, self._active_head].argmax().item()
    
    @torch.no_grad()
    def select_action(self, observation: NDArray[np.float32], step: int) -> int:
        # Epsilon-greedy policy.
        if self._rng.rand() < self._epsilon_fn(self._total_steps):
            return self._rng.randint(self._num_actions)

        observation = torch.from_numpy(observation[None, ...])
        # import pdb
        # pdb.set_trace()
        q_values: NDArray[np.float64] = self._ensemble_Q_network(observation)[0, self._active_head].numpy()
        m_values: NDArray[np.float64] = self._ensemble_M_network(observation)[0, self._active_head].numpy()

        if np.any(m_values < 0):
            print(m_values)
        delta = np.clip(q_values.max() - q_values,  self._delta_min(self._total_steps),np.inf)
        idxs = np.argwhere(delta == delta.max())
        
        if np.sum(idxs) > 1:
            import pdb
            pdb.set_trace()
        delta[idxs] *= (1-self._discount)
        # idxs = np.isclose(delta,0)
        # delta[idxs] = np.clip(delta[~idxs].min(), self._delta_min(self._total_steps), np.inf) * (1-self._discount)
    
        H = (2 + 8 * golden_ratio_sq * m_values) /  (delta ** 2)
        H[idxs] = np.sqrt(H[idxs] * H[~idxs].sum())
        p = H / H.sum(-1, keepdims=True)
        return int(np.random.choice(self._num_actions, p = p))

    def update(self, timestep: TimeStep):
        if timestep.done:
            self._active_head = self._rng.randint(self._num_ensemble)
            
            
        # Add this transition to replay.
        self._replay.add(TransitionWithMaskAndNoise(
            timestep.observation,
            timestep.action,
            np.float32(timestep.reward),
            np.float32(timestep.done),
            timestep.observation,
            m_t=self._rng.binomial(1, self._mask_prob,
                                    self._num_ensemble).astype(np.float32),
            z_t=self._rng.randn(self._num_ensemble).astype(np.float32) *
                self._noise_scale,))

    
        if self._total_steps % self._sgd_period != 0 or self._replay.size < self._min_replay_size:
            return

        transitions = self._replay.sample(self._batch_size)
        self._training_step(transitions)
        self._total_steps += 1


    def _training_step(self, transitions: Sequence[TimeStep]) -> float:
        """Does a step of SGD on a batch of transitions."""
        o_tm1, a_tm1, r_t, d_t, o_t, m_t, z_t = transitions
        r_t = torch.tensor(r_t, dtype=torch.float32, requires_grad=False)  # [B]
        d_t = torch.tensor(d_t, dtype=torch.float32, requires_grad=False)  # [B]
        o_tm1 = torch.tensor(o_tm1, dtype=torch.float32, requires_grad=False)
        o_t = torch.tensor(o_t, dtype=torch.float32, requires_grad=False)
        a_tm1 = torch.tensor(a_tm1, dtype=torch.int64, requires_grad=False)
        m_t = torch.tensor(m_t, dtype=torch.float32, requires_grad=False)
        z_t = torch.tensor(z_t, dtype=torch.float32, requires_grad=False)

        # Train Q Values
        with torch.no_grad():
            q_t = self._ensemble_Q_target_network(o_t).max(-1)[0]  # [B]
            target_y = r_t.unsqueeze(-1) + z_t + (1-d_t.unsqueeze(-1)) * self._discount * q_t
    
        # Update the online network via SGD.
        self._Q_optimizer.zero_grad()
        
        q_values = self._ensemble_Q_network(o_tm1).gather(-1, a_tm1[:, None, None].repeat(1, self._num_ensemble, 1)).squeeze(-1)
        q_values = torch.mul(q_values, m_t)
        target_y = torch.mul(target_y, m_t)
        
        loss_q: torch.Tensor = nn.HuberLoss()(q_values, target_y.detach()) * 0.5    
        loss_q.backward()
        self._Q_optimizer.step()
        # Periodically copy online -> target network variables.
        if self._total_steps % self._target_update_period == 0:
            self._ensemble_Q_target_network.load_state_dict(self._ensemble_Q_network.state_dict())
            
            
        # Train Var Network
        with torch.no_grad():
            q_values: torch.Tensor = self._ensemble_Q_target_network(o_tm1)
            q_values = q_values.gather(-1, a_tm1[:, None, None].repeat(1, self._num_ensemble, 1)).squeeze(-1)
            q_next_values: torch.Tensor = self._ensemble_Q_target_network(o_t).max(-1)[0]
            W = (r_t.unsqueeze(-1) + z_t + (1-d_t.unsqueeze(-1)) * self._discount * q_next_values - q_values) / (self._discount)
            W_target = W ** 2
    
        self._M_optimizer.zero_grad()
        m_values = self._ensemble_M_network(o_tm1)
        m_values = m_values.gather(-1, a_tm1[:, None, None].repeat(1, self._num_ensemble, 1)).squeeze(-1)
        
        m_values = torch.mul(m_values, m_t)
        W_target = torch.mul(W_target, m_t)
        
        loss_v: torch.Tensor = nn.HuberLoss()(m_values, W_target.detach())

        loss_v.backward()
        self._M_optimizer.step()
        return (loss_q+loss_v).item()
  
  
class TransitionWithMaskAndNoise(NamedTuple):
    o_tm1: NDArray[np.float64]
    a_tm1: int
    r_t: float
    d_t: float
    o_t: NDArray[np.float64]
    m_t: NDArray[np.int64]
    z_t: NDArray[np.float64]

def make_single_network(input_size: int, output_size: int, hidden_size: int, ensemble_size: int, final_activation: bool = False) -> nn.Module:
    def init_weights(m):
        torch.nn.init.trunc_normal_(m.weight, 0.5, 1, 1e-3, 2)
        m.bias.data.fill_(1e-2)
            
    net = [
        nn.Flatten(start_dim=-2),
        EnsembleLinear(input_size, hidden_size, ensemble_size),
        nn.ReLU(),
        EnsembleLinear(hidden_size, hidden_size, ensemble_size),
        nn.ReLU(),
        EnsembleLinear(hidden_size, output_size, ensemble_size)]
    net[-1].apply(init_weights)
    if final_activation:
        net.append(nn.ReLU())
        
    return nn.Sequential(*net)


    

def default_agent(obs_spec: NDArray, num_actions: int, num_ensemble: int = 20, prior_scale: float = 5):
    size_s = np.prod(obs_spec.shape)

    hidden_size = 32

    Q_network = EnsembleWithPrior(
        size_s, num_actions, prior_scale=prior_scale, ensemble_size=num_ensemble, final_activation=False, hidden_size=hidden_size)
    M_network = EnsembleWithPrior(
        size_s, num_actions, prior_scale=prior_scale, ensemble_size=num_ensemble, final_activation=True, hidden_size=hidden_size)        
        
    Q_optimizer = optim.Adam(Q_network.parameters(), lr=1e-3)
    M_optimizer = optim.Adam(M_network.parameters(), lr=1e-4)
    return ExplorativeAgent(
        num_actions=num_actions,
        Q_network=Q_network,
        M_network=M_network,
        batch_size=128,
        discount=0.99,
        replay_capacity=100000,
        min_replay_size=128,
        sgd_period=1,
        target_update_period=4,
        Q_optimizer=Q_optimizer,
        M_optimizer=M_optimizer,
        mask_prob=.5,
        noise_scale=0.0,
        epsilon_fn=lambda t: 10 / (10 + t),
        delta_min=lambda t: 0.05,
        seed=42)