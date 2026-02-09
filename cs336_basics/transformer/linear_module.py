import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModule(nn.Module):
    def __init__(self, in_features: int, out_features: int, device = None, dtype = None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        self.std = math.sqrt(2/(in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0, std=self.std, a=-3*self.std, b=3*self.std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T