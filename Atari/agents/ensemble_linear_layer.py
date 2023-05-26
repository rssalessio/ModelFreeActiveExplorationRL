import math
import torch
import torch.nn as nn

class EnsembleLinear(nn.Module):
    """
        Fully connected layers for ensemble models
    """
    __constants__ = ['ensemble_size', 'in_features', 'out_features']
    in_features: int
    out_features: int
    ensemble_size: int
    weight_decay: float
    weight: torch.Tensor
    device: torch.device
    inject_noise: bool
    noise_dim: int

    def __init__(self, 
            in_features: int,
            out_features: int,
            ensemble_size: int,
            weight_decay: float = 0.,
            bias: bool = True,
            device: torch.device = torch.device('cpu')) -> None:
        super(EnsembleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size
        self.device = device
        self.weight = nn.Parameter(torch.Tensor(self.ensemble_size, self.in_features, self.out_features)).to(device)
        self.weight_decay = weight_decay
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.ensemble_size, self.out_features)).to(device)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, mode='fan_in', nonlinearity='relu')
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # w times x + b
        w_times_x = torch.bmm(input, self.weight)
        return torch.add(w_times_x, self.bias[:, None, :])

    def extra_repr(self) -> str:
        return f'ensemble_size={self.ensemble_size}, in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

    def get_decay_loss(self, exponent: float = 2) -> float:
        return (self.weight_decay * self.weight.pow(exponent).sum()) / exponent