import math
import torch
import torch.nn as nn

class RMSNormModule(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype
        self.W = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms_norm = x/(torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps))
        return rms_norm.to(in_dtype) * self.W