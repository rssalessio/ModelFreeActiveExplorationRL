import copy
from typing import Callable, NamedTuple, Optional, Sequence
from .replay_buffer import ReplayBuffer
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from .agent import TimeStep, Agent
from .ensemble_linear_layer import EnsembleLinear
from .quantile_network import QuantileNetwork, MLPFeaturesExtractor, quantile_huber_loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IDSQ(Agent):
    """Bootstrapped DQN with additive prior functions."""
    def __init__(
            self,
            state_dim: int,
            num_actions: int,
            ensemble: nn.Module,
            quantile_network: nn.Module,
            batch_size: int,
            discount: float,
            replay_capacity: int,
            min_replay_size: int,
            sgd_period: int,
            target_update_period: int,
            optimizer: torch.optim.Optimizer,
            optimizer_quantile_network: torch.optim.Optimizer,
            epsilon_fn: Callable[[int], float] = lambda _: 0.,
            seed: Optional[int] = None):
        """Bootstrapped DQN with additive prior functions."""
        # Agent components.
        self._state_dim = state_dim
        self._ensemble = ensemble        
        self._target_ensemble = copy.deepcopy(ensemble)
        self._quantile_network = quantile_network

        self._num_ensemble = ensemble.ensemble_size
        self._optimizer = optimizer
        self._optimizer_quantile_network = optimizer_quantile_network
        self._replay = ReplayBuffer(capacity=replay_capacity)

        # Agent hyperparameters.
        self._num_actions = num_actions
        self._batch_size = batch_size
        self._sgd_period = sgd_period
        self._target_update_period = target_update_period
        self._min_replay_size = min_replay_size
        self._epsilon_fn = epsilon_fn
        self._rng = np.random.RandomState(seed)
        self._discount = discount

        # Agent state.
        self._total_steps = 1
        self._total_episodes = 0
        torch.random.manual_seed(seed)
        self._pool_states = []

    def _step(self, transitions: Sequence[torch.Tensor]):
        o_tm1, a_tm1, r_t, d_t, o_t = transitions
        a_tm1 = torch.tensor(a_tm1, dtype=torch.int64, requires_grad=False, device=device)
        r_t = torch.tensor(r_t, dtype=torch.float32, requires_grad=False, device=device)
        d_t = torch.tensor(d_t, dtype=torch.float32, requires_grad=False, device=device)
        o_tm1 = torch.tensor(o_tm1, dtype=torch.float32, requires_grad=False, device=device)
        o_t = torch.tensor(o_t, dtype=torch.float32, requires_grad=False, device=device)

        with torch.no_grad():
            next_actions = self._ensemble(o_t).max(-1)[1]
            q_target = self._target_ensemble(o_t)
            q_target = q_target.gather(-1, next_actions.unsqueeze(-1)).squeeze(-1)
            target_y = r_t.unsqueeze(-1) + self._discount * (1-d_t.unsqueeze(-1)) * q_target
        
        q_values = self._ensemble(o_tm1).gather(-1, a_tm1[:, None, None].repeat(1, self._ensemble.ensemble_size, 1)).squeeze(-1)

        
        self._optimizer.zero_grad()
        # loss = nn.HuberLoss()(q_values, target_y.detach())
        loss = torch.square(q_values - target_y.detach()).mean() / self._num_ensemble
        loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(self._ensemble.parameters(), 1)
        self._optimizer.step()
    
        self._total_steps += 1

        # Periodically update the target network.
        if self._total_steps % self._target_update_period == 0:
            self._target_ensemble.load_state_dict(self._ensemble.state_dict())

       # Train quantile network
        with torch.no_grad():
            #next_actions_target = self._ensemble(o_t).mean(1).max(-1)[1]
            target_quantiles = self._quantile_network(o_t)
            next_actions_target = target_quantiles.mean(1).max(-1)[1]

            n_quantiles = target_quantiles.shape[1]
            actions_target = next_actions_target.unsqueeze(1)[..., None].long().expand(next_actions.shape[0], n_quantiles, 1)
            target_quantiles = torch.gather(target_quantiles, dim=2, index=actions_target).squeeze(-1)
            
            target_quantiles = r_t.unsqueeze(-1) + self._discount * (1-d_t.unsqueeze(-1)) * target_quantiles
        
        current_quantiles = self._quantile_network(o_tm1)
        # Make "n_quantiles" copies of actions, and reshape to (batch_size, n_quantiles, 1).
        actions_copy = a_tm1.unsqueeze(1)[..., None].long().expand(a_tm1.shape[0], n_quantiles, 1)
        # Retrieve the quantiles for the actions from the replay buffer
        current_quantiles = torch.gather(current_quantiles, dim=2, index=actions_copy).squeeze(dim=2)

        # Optimize the quantile network
        self._optimizer_quantile_network.zero_grad()
        # Compute Quantile Huber loss, summing over a quantile dimension as in the paper.
        loss_quantile = quantile_huber_loss(current_quantiles, target_quantiles.detach(), sum_over_quantiles=True)
        torch.nn.utils.clip_grad.clip_grad_norm_(self._quantile_network.parameters(), 1)
        loss_quantile.backward()
        self._optimizer_quantile_network.step()

        
        return loss.item() + loss_quantile.item()

    @torch.no_grad()
    def _select_action(self, observation: NDArray[np.float32], greedy: bool=False) -> int:
        if greedy is False and self._rng.rand() < self._epsilon_fn(self._total_steps):
            return self._rng.randint(self._num_actions)

        observation = torch.tensor(observation, dtype=torch.float32, device=device)
        # Greedy policy, breaking ties uniformly at random.
        q_values = self._ensemble(observation)[0].cpu().numpy()
        mu = q_values.mean(0)

        if greedy:
            return mu.argmax()
        sigma = q_values.std(0)

        lmbd = 5
        eps1 = 1e-5
        eps2 = eps1
        delta = np.max(mu + lmbd * sigma) - (mu - lmbd * sigma)

        quantiles = self._quantile_network(observation)[0].cpu().numpy()
        #muz = quantiles.mean(0)
        var_quant = quantiles.var(0)
        rho_sq = np.maximum(0.25, var_quant / (eps1 + var_quant.mean()))
        I = np.log(1 + (sigma ** 2) / rho_sq) + eps2
        psi = (delta ** 2) / I

        # if np.random.uniform() < 1e-2:
        #    print(f'{var_quant} -{quantiles.mean(0)} - {var_quant.mean()} - {sigma**2} - {I} - {psi} -{delta}')
        return self._rng.choice(np.flatnonzero(psi == psi.min()))
        
    def select_action(self, observation: NDArray[np.float32], step: int) -> int:
        return self._select_action(observation)

    def select_greedy_action(self, observation: NDArray[np.float32]) -> int:
        return self._select_action(observation, greedy=True)     
    
    def update(self, timestep: TimeStep) -> None:
        return self._update(
            np.float32(timestep.observation), timestep.action, np.float32(timestep.reward),
            np.float32(timestep.next_observation), timestep.done)
    
    def _update(
            self,
            observation: NDArray[np.float64],
            action: int,
            reward: float,
            new_observation: NDArray[np.float64],
            done: bool) -> Optional[float]:
        """Update the agent: add transition to replay and periodically do SGD."""
        if done:
            self._total_episodes += 1
            
            if self._total_steps > 1:
                with torch.no_grad():
                    qvalues = self._ensemble(self._pool_states)
                    # qprior = self._ensemble._prior_network(self._pool_states[None, ...].repeat(self._num_ensemble, 1, 1)).swapaxes(0,1)
                    print(f'Eps:{ self._epsilon_fn(self._total_steps)}-Mu: {qvalues.mean(1).mean(0)} - std: {qvalues.std(1).mean(0) + qvalues.mean(1).std(0)} ')


        self._replay.add(
            Transition(
                o_tm1=observation.flatten(),
                a_tm1=action,
                r_t=np.float32(reward),
                d_t=np.float32(done),
                o_t=new_observation.flatten(),
            ))

        if self._replay.size < self._min_replay_size:
            self._pool_states.append(observation.flatten())
            return None

        if self._total_steps % self._sgd_period != 0:
            return None
        
        if self._total_steps == 1:
            self._pool_states = torch.tensor(np.vstack(self._pool_states), dtype=torch.float32, requires_grad=False).to(device)

        minibatch = self._replay.sample(self._batch_size)
        return self._step(minibatch)


class Transition(NamedTuple):
    o_tm1: NDArray[np.float64]
    a_tm1: int
    r_t: float
    d_t: float
    o_t: NDArray[np.float64]
    

def make_single_network(input_size: int, output_size: int, hidden_size: int, ensemble_size: int) -> nn.Module:
    return nn.Sequential(*[
        EnsembleLinear(input_size, hidden_size, ensemble_size),
        nn.ReLU(),
        EnsembleLinear(hidden_size, hidden_size, ensemble_size),
        nn.ReLU(),
        EnsembleLinear(hidden_size, output_size, ensemble_size)])

class EnsembleQ(nn.Module):
    def __init__(self, input_size: int, output_size: int, ensemble_size: int, hidden_size: int = 32):
        super().__init__()
        
        self.ensemble_size = ensemble_size
        self._network = make_single_network(input_size, output_size, hidden_size, ensemble_size)
        
        def init_weights(m):
            if isinstance(m, EnsembleLinear):
                stddev = 1 / np.sqrt(m.weight.shape[1])
                torch.nn.init.trunc_normal_(m.weight, mean=0, std=stddev, a=-2*stddev, b=2*stddev)
                torch.nn.init.zeros_(m.bias.data)
        self._network.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[None, ...].repeat(self.ensemble_size,  1, 1)
        q_values = self._network.forward(x).swapaxes(0,1)
        return q_values
  
def default_agent(
        obs_spec: NDArray,
        num_actions: int,
        num_ensemble: int = 20,
        seed: int = 0) -> IDSQ:
    """Initialize a Bootstrapped DQN agent with default parameters."""

    state_dim = np.prod(obs_spec.shape)
    ensemble = EnsembleQ(state_dim, num_actions, num_ensemble, 50).to(device)
    feat_ext = MLPFeaturesExtractor(state_dim, 50, hidden_size=50).to(device)
    quantile_net = QuantileNetwork(state_dim, num_actions, feat_ext, n_quantiles=50).to(device)
    
    optimizer = torch.optim.Adam(ensemble.parameters(), lr=4e-5)
    optimizer_quantile_net = torch.optim.Adam(quantile_net.parameters(), lr =1e-6)

    return IDSQ(
        state_dim=state_dim,
        num_actions=num_actions,
        ensemble=ensemble,
        quantile_network=quantile_net,
        batch_size=128,
        discount=.99,
        replay_capacity=100000,
        min_replay_size=128,
        sgd_period=1,
        target_update_period=4,
        optimizer=optimizer,
        optimizer_quantile_network=optimizer_quantile_net,
        epsilon_fn=lambda t: 10/ (10 + t),
        seed=seed,
    )