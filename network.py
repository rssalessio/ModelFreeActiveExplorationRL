import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, ns: int, na: int, hidden: int, lr: float = 1e-2):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(ns, hidden),
            nn.SiLU(),
            nn.Linear(hidden, na),
            nn.Softmax()
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def backward(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        torch.nn.utils.clip_grad.clip_grad_norm_(
            self.network.parameters(), 1
        )
        
        
