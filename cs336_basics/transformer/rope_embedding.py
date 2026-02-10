import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        k = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)  # [0, 2, 4, ..., d_k-2]
        freqs = theta ** (-k / d_k)  # shape: (d_k/2,)
        positions = torch.arange(0, max_seq_len, device=device, dtype=torch.float32)  # shape: (max_seq_len,)
        outer_prod = torch.outer(positions, freqs)

        pe_cos = torch.cos(outer_prod)
        pe_sin = torch.sin(outer_prod)

        self.register_buffer('pe_cos', pe_cos, persistent=False)
        self.register_buffer('pe_sin', pe_sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        req_pe_cos = self.pe_cos[token_positions]
        req_pe_sin = self.pe_sin[token_positions]

        x_odd = x[..., 1::2]
        x_even= x[..., ::2]
        pe_odd, pe_even = x_even*req_pe_cos - x_odd*req_pe_sin, x_even*req_pe_sin + x_odd*req_pe_cos
        return torch.stack([pe_odd, pe_even], dim=-1).reshape(x.shape)

