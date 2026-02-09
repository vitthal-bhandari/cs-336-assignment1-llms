import torch
import torch.nn as nn

class PositionwiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device = None, dtype = None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = nn.Parameter(torch.empty(d_ff, d_model, **factory_kwargs))
        self.W2 = nn.Parameter(torch.empty(d_model, d_ff, **factory_kwargs))
        self.W3 = nn.Parameter(torch.empty(d_ff, d_model, **factory_kwargs))

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.silu(x @ self.W1.T) * (x @ self.W3.T) @ self.W2.T