import torch
import torch.nn as nn

class Utility():
    def __init__(self, device=None):
        self.device = device

    def compute_softmax(self, in_features: torch.Tensor, dim: int) -> torch.Tensor:
        max_val = torch.max(in_features, dim=dim, keepdim=True).values
        exp_in_features = torch.exp(in_features - max_val)
        return exp_in_features / torch.sum(exp_in_features, dim=dim, keepdim=True)
    
    def scaled_dot_product_attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        QK = torch.einsum('... qd, ... kd -> ... qk', Q, K)
        QK = QK / torch.sqrt(torch.tensor(Q.shape[-1]))
        if mask is not None:
            QK = QK.masked_fill(mask == 0, float('-inf'))
        return torch.einsum('... qk, ... kd -> ... qd', self.compute_softmax(QK, dim=-1), V)