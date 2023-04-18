
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

class Values(NamedTuple):
    q_values: torch.Tensor
    m_values: torch.Tensor
    
def make_base(input_size: int,  hidden_size: int, output_size: int, ensemble_size: int):
    return [
        nn.Flatten(start_dim=-2),
        EnsembleLinear(input_size, hidden_size, ensemble_size),
        nn.ReLU(),
        EnsembleLinear(hidden_size, hidden_size, ensemble_size),
        nn.ReLU(),
        EnsembleLinear(hidden_size, output_size, ensemble_size)]

class EnsembleWithPrior(nn.Module):
    def __init__(self, input_size: int, output_size: int, prior_scale: float, ensemble_size: int, hidden_size: int = 32):
        super().__init__()
        
        self.ensemble_size = ensemble_size
        
        self._Q_head = nn.Sequential(*make_base(input_size, hidden_size, output_size, ensemble_size))
        # self._M_head = make_base(input_size, hidden_size, output_size, ensemble_size) + [nn.ReLU()]
        # self._M_head = nn.Sequential(*self._M_head)
        
        
        self._Q_prior = nn.Sequential(*make_base(input_size, hidden_size, output_size, ensemble_size))
        # self._M_prior = make_base(input_size, hidden_size, output_size, ensemble_size) + [nn.ReLU()]
        # self._M_prior = nn.Sequential(*self._M_prior)
        
        def init_weights(m):
            if isinstance(m, EnsembleLinear):
                torch.nn.init.trunc_normal_(m.weight, 0.5, 1, 1e-3, 2)
                m.bias.data.fill_(1e-2)

        # self._M_head.apply(init_weights)
        # self._M_prior.apply(init_weights)
        self._prior_scale = prior_scale
    
    def forward(self, x: torch.Tensor) -> Values:
        x = x[None, ...].repeat(self.ensemble_size, 1, 1, 1)
        
        q_values, q_prior = self._Q_head(x).swapaxes(0,1), self._Q_prior(x).swapaxes(0,1)
        #m_values, m_prior = self._M_head(x).swapaxes(0,1), self._M_prior(x).swapaxes(0,1)
        
        q_values = q_values + self._prior_scale * q_prior.detach()
        #m_values = m_values + self._prior_scale * m_prior.detach()
        return Values(q_values, None)

class ExplorativeAgent(Agent):
    """A simple DQN agent using TF2."""

    def __init__(
        self,
        num_actions: int,
        ensemble_network: EnsembleWithPrior,
        batch_size: int,
        discount: float,
        replay_capacity: int,
        min_replay_size: int,
        sgd_period: int,
        target_update_period: int,
        optimizer: optim.Optimizer,
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
        self._ensemble = ensemble_network
        self._ensemble_target = copy.deepcopy(ensemble_network)
        self._total_steps = 0
        self._optimizer = optimizer
        self._num_ensemble = ensemble_network.ensemble_size
        
        self._active_head = 0


    @torch.no_grad()
    def select_greedy_action(self, observation: NDArray[np.float32]) -> int:
        observation = torch.from_numpy(observation[None, ...])
        values: Values = self._ensemble(observation)
        return values.q_values[0, self._active_head].argmax().item()
    
    @torch.no_grad()
    def select_action(self, observation: NDArray[np.float32], step: int) -> int:
        # Epsilon-greedy policy.
        if self._rng.rand() < self._epsilon_fn(self._total_steps):
            return self._rng.randint(self._num_actions)

        observation = torch.from_numpy(observation[None, ...])

        values: Values = self._ensemble(observation)
        q_values: NDArray[np.float64] = values.q_values[0, self._active_head].numpy()
        #m_values: NDArray[np.float64] = values.m_values[0, self._active_head].numpy()
        
        if True:
            return int(q_values.argmax())
        

        if np.any(m_values < 0):
            print(m_values)
            
        idx = q_values.argmax()
        delta = q_values[idx] - q_values
        delta[idx] = self._delta_min(self._total_steps) * (1-self._discount)
        # delta = np.clip(_delta,  self._delta_min(self._total_steps),np.inf)
        # print(f'{q_values} - {_delta} -> {delta}')
        # idxs = np.argwhere(delta == delta.max())
        
        # if np.sum(idxs) > 1:
        #     import pdb
        #     pdb.set_trace()
        #delta[idxs] *= (1-self._discount)
        # idxs = np.isclose(delta,0)
        # delta[idxs] = np.clip(delta[~idxs].min(), self._delta_min(self._total_steps), np.inf) * (1-self._discount)
    
        H = (2 + 8 * golden_ratio_sq * m_values) /  (delta ** 2)
        H[idx] = np.sqrt(H[idx] * (H.sum() - H[idx]))
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

        # import pdb
        # pdb.set_trace()
        # Train Q Values
        with torch.no_grad():
            q_t = self._ensemble_target.forward(o_t).q_values.max(-1)[0]  # [B]
            target_y = r_t.unsqueeze(-1) + z_t + (1-d_t.unsqueeze(-1)) * self._discount * q_t
    
        
        if False:
            # Train Var Network
            with torch.no_grad():
                q_values: torch.Tensor = self._ensemble_target.forward(o_tm1).q_values
                q_values = q_values.gather(-1, a_tm1[:, None, None].repeat(1, self._num_ensemble, 1)).squeeze(-1)
                q_next_values: torch.Tensor = self._ensemble_target.forward(o_t).q_values.max(-1)[0]
                W = (r_t.unsqueeze(-1) + z_t + (1-d_t.unsqueeze(-1)) * self._discount * q_next_values - q_values) / (self._discount)
                W_target = W ** 2
                
        values_ensemble: Values = self._ensemble.forward(o_tm1)
        q_values = values_ensemble.q_values.gather(-1, a_tm1[:, None, None].repeat(1, self._num_ensemble, 1)).squeeze(-1)
        
        #m_values = values_ensemble.m_values.gather(-1, a_tm1[:, None, None].repeat(1, self._num_ensemble, 1)).squeeze(-1)
        q_values = torch.mul(q_values, m_t)
        target_y = torch.mul(target_y, m_t)
        
        #m_values = torch.mul(m_values, m_t)
        #W_target = torch.mul(W_target, m_t)
        
        self._optimizer.zero_grad()
        loss_q: torch.Tensor = nn.HuberLoss()(q_values, target_y.detach())      
        loss_v = 0
        #loss_v: torch.Tensor = nn.HuberLoss()(m_values, W_target.detach())
        loss = loss_q  + loss_v
        loss.backward()
        self._optimizer.step()
        
        # Periodically copy online -> target network variables.
        if self._total_steps % self._target_update_period == 0:
            self._ensemble_target.load_state_dict(self._ensemble.state_dict())
        return loss.item()
  
  
class TransitionWithMaskAndNoise(NamedTuple):
    o_tm1: NDArray[np.float64]
    a_tm1: int
    r_t: float
    d_t: float
    o_t: NDArray[np.float64]
    m_t: NDArray[np.int64]
    z_t: NDArray[np.float64]

    

def default_agent(obs_spec: NDArray, num_actions: int, num_ensemble: int = 20, prior_scale: float = 3, seed: int =0):
    size_s = np.prod(obs_spec.shape)

    hidden_size = 32

    network = EnsembleWithPrior(
        size_s, num_actions, prior_scale=prior_scale, ensemble_size=num_ensemble, hidden_size=hidden_size)
    optimizer = optim.Adam(network.parameters(), lr=1e-3)
    return ExplorativeAgent(
        num_actions=num_actions,
        ensemble_network=network,
        batch_size=128,
        discount=0.99,
        replay_capacity=100000,
        min_replay_size=128,
        sgd_period=1,
        target_update_period=4,
        optimizer=optimizer,
        mask_prob=.5,
        noise_scale=0.0,
        epsilon_fn=lambda t: 10 / (10 + t),
        delta_min=lambda t: 1,
        seed=seed)