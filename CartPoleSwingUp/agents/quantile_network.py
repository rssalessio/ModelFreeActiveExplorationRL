import torch
import torch.nn as nn
from typing import List, Type, Optional
import numpy as np



class MLPFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.
    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, dim_input: int, features_dim: int, hidden_size: int = 32,  activation_fn: Type[nn.Module] = nn.ReLU,):
        super().__init__()
        self._dim_input = dim_input
        self._features_dim = features_dim

        self.extractor = nn.Sequential(*[
            nn.Flatten(),
            nn.Linear(dim_input, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, hidden_size),
            activation_fn(),
            nn.Linear(hidden_size, features_dim),
            activation_fn()
        ])
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                stddev = 2 / np.sqrt(m.weight.shape[1])
                torch.nn.init.trunc_normal_(m.weight, mean=0, std=stddev, a=-2*stddev, b=2*stddev)
                torch.nn.init.zeros_(m.bias.data)
        self.extractor.apply(init_weights)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.extractor(observations)



class QuantileNetwork(nn.Module):
    """
    Quantile network
    """

    def __init__(
        self,
        dim_input: int,
        dim_action: int,
        features_extractor: Type[MLPFeaturesExtractor],
        n_quantiles: int = 50
    ):
        super().__init__()
        self.features_extractor = features_extractor
        self.features_dim = features_extractor._features_dim
        self.n_quantiles = n_quantiles
        self.dim_input = dim_input
        self.dim_action = dim_action

        self.output_layer = nn.Linear(self.features_dim, dim_action * self.n_quantiles)
        self._probabilities = np.linspace(1 / self.n_quantiles, 1, self.n_quantiles)
        
        def init_weights(m):
            if isinstance(m, nn.Linear):
                stddev = 2 / np.sqrt(m.weight.shape[1])
                torch.nn.init.trunc_normal_(m.weight, mean=0, std=stddev, a=-2*stddev, b=2*stddev)
                torch.nn.init.zeros_(m.bias.data)
        self.output_layer.apply(init_weights)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Predict the quantiles.
        :param obs: Observation
        :return: The estimated quantiles for each action.
        """
        quantiles = self.output_layer(self.features_extractor(obs))
        ret = quantiles.view(-1, self.n_quantiles, self.dim_action)
        return ret
    

def quantile_huber_loss(
    current_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    cum_prob: Optional[torch.Tensor] = None,
    sum_over_quantiles: bool = True,
) -> torch.Tensor:
    """
    The quantile-regression loss, as described in the QR-DQN and TQC papers.
    Partially taken from https://github.com/bayesgroup/tqc_pytorch.
    :param current_quantiles: current estimate of quantiles, must be either
        (batch_size, n_quantiles) or (batch_size, n_critics, n_quantiles)
    :param target_quantiles: target of quantiles, must be either (batch_size, n_target_quantiles),
        (batch_size, 1, n_target_quantiles), or (batch_size, n_critics, n_target_quantiles)
    :param cum_prob: cumulative probabilities to calculate quantiles (also called midpoints in QR-DQN paper),
        must be either (batch_size, n_quantiles), (batch_size, 1, n_quantiles), or (batch_size, n_critics, n_quantiles).
        (if None, calculating unit quantiles)
    :param sum_over_quantiles: if summing over the quantile dimension or not
    :return: the loss
    """
    if current_quantiles.ndim != target_quantiles.ndim:
        raise ValueError(
            f"Error: The dimension of curremt_quantile ({current_quantiles.ndim}) needs to match "
            f"the dimension of target_quantiles ({target_quantiles.ndim})."
        )
    if current_quantiles.shape[0] != target_quantiles.shape[0]:
        raise ValueError(
            f"Error: The batch size of curremt_quantile ({current_quantiles.shape[0]}) needs to match "
            f"the batch size of target_quantiles ({target_quantiles.shape[0]})."
        )
    if current_quantiles.ndim not in (2, 3):
        raise ValueError(f"Error: The dimension of current_quantiles ({current_quantiles.ndim}) needs to be either 2 or 3.")

    if cum_prob is None:
        n_quantiles = current_quantiles.shape[-1]
        # Cumulative probabilities to calculate quantiles.
        cum_prob = (torch.arange(n_quantiles, device=current_quantiles.device, dtype=torch.float) + 0.5) / n_quantiles
        if current_quantiles.ndim == 2:
            # For QR-DQN, current_quantiles have a shape (batch_size, n_quantiles), and make cum_prob
            # broadcastable to (batch_size, n_quantiles, n_target_quantiles)
            cum_prob = cum_prob.view(1, -1, 1)
        elif current_quantiles.ndim == 3:
            # For TQC, current_quantiles have a shape (batch_size, n_critics, n_quantiles), and make cum_prob
            # broadcastable to (batch_size, n_critics, n_quantiles, n_target_quantiles)
            cum_prob = cum_prob.view(1, 1, -1, 1)

    # QR-DQN
    # target_quantiles: (batch_size, n_target_quantiles) -> (batch_size, 1, n_target_quantiles)
    # current_quantiles: (batch_size, n_quantiles) -> (batch_size, n_quantiles, 1)
    # pairwise_delta: (batch_size, n_target_quantiles, n_quantiles)
    # TQC
    # target_quantiles: (batch_size, 1, n_target_quantiles) -> (batch_size, 1, 1, n_target_quantiles)
    # current_quantiles: (batch_size, n_critics, n_quantiles) -> (batch_size, n_critics, n_quantiles, 1)
    # pairwise_delta: (batch_size, n_critics, n_quantiles, n_target_quantiles)
    # Note: in both cases, the loss has the same shape as pairwise_delta
    pairwise_delta = target_quantiles.unsqueeze(-2) - current_quantiles.unsqueeze(-1)
    abs_pairwise_delta = torch.abs(pairwise_delta)
    huber_loss = torch.where(abs_pairwise_delta > 1, abs_pairwise_delta - 0.5, pairwise_delta**2 * 0.5)
    loss = torch.abs(cum_prob - (pairwise_delta.detach() < 0).float()) * huber_loss
    if sum_over_quantiles:
        loss = loss.sum(dim=-2).mean()
    else:
        loss = loss.mean()
    return loss

