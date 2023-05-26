import torch
import torch.nn as nn
import numpy as np
from .ensemble_linear_layer import EnsembleLinear
from .ensemble_conv2d import EnsembleConv2d
import gymnasium as gym

def reshape1(x, cnn, ensemble_size):
    y = cnn(x)[None, ...]
    return y.reshape(ensemble_size, 1, -1)

def reshape2(x, cnn, ensemble_size):
    y = cnn(x)
    z = []

    r = y.shape[1] // ensemble_size
    for i in range(ensemble_size):
        q = y[:, i*r : (i+1)*r, ...]
        z.append(q)

    return torch.vstack(z).reshape(ensemble_size, -1)[:, None, :]

def make_single_network(observation_space: gym.Space, output_size: int, hidden_size: int, ensemble_size: int) -> nn.Module:
    n_input_channels = observation_space.shape[0]
    cnn = nn.Sequential(*[
        EnsembleConv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0, ensemble_size=ensemble_size),
        nn.ReLU(),
        EnsembleConv2d(32, 64, kernel_size=4, stride=2, padding=0, ensemble_size=ensemble_size),
        nn.ReLU(),
        EnsembleConv2d(64, 64, kernel_size=3, stride=1, padding=0, ensemble_size=ensemble_size),
        nn.ReLU(),]
    )
    import pdb
    pdb.set_trace()
    with torch.no_grad():
        x = torch.as_tensor(observation_space.sample()[None]).float()

        y1 = reshape1(x, cnn, ensemble_size)
        y2 = reshape2(x, cnn, ensemble_size)
        #n_flatten = cnn().shape[1]

    return cnn
    #linear = nn.Sequential(EnsembleLinear(n_flatten, features_dim), nn.ReLU())



class EnsembleWithPrior(nn.Module):
    def __init__(self, input_size: int, output_size: int, prior_scale: float, ensemble_size: int, hidden_size: int = 32):
        super().__init__()
        
        self.ensemble_size = ensemble_size
        self._network = make_single_network(input_size, output_size, hidden_size, ensemble_size)
        self._prior_network = make_single_network(input_size, output_size, hidden_size, ensemble_size)
        self._prior_scale = prior_scale

        def init_weights(m):
            if isinstance(m, EnsembleLinear):
                stddev = 1 / np.sqrt(m.weight.shape[1])
                torch.nn.init.trunc_normal_(m.weight, mean=0, std=stddev, a=-2*stddev, b=2*stddev)
                torch.nn.init.zeros_(m.bias.data)

        self._prior_network.apply(init_weights)
        self._network.apply(init_weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[None, ...].repeat(self.ensemble_size, 1, 1)
        q_values = self._network.forward(x).swapaxes(0,1)
        prior_q_values = self._prior_network(x).swapaxes(0,1)
        return q_values + self._prior_scale * prior_q_values.detach()

