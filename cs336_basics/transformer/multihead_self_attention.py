import torch
import torch.nn as nn
from einops import rearrange
from cs336_basics.transformer import LinearModule, Utility, RotaryPositionalEmbedding

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float | None = None, max_seq_len: int | None = None, device = None, dtype = None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = LinearModule(d_model, d_model, **factory_kwargs)
        self.W_k = LinearModule(d_model, d_model, **factory_kwargs)
        self.W_v = LinearModule(d_model, d_model, **factory_kwargs)
        self.W_o = LinearModule(d_model, d_model, **factory_kwargs)
        if theta and max_seq_len:
            self.rope_embedding = RotaryPositionalEmbedding(self.theta, self.d_k, self.max_seq_len)
        else:
            self.rope_embedding = None

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        Q = rearrange(Q, "b s (h d) -> b h s d", h =self.num_heads)
        K = rearrange(K, "b s (h d) -> b h s d", h =self.num_heads)
        V = rearrange(V, "b s (h d) -> b h s d", h =self.num_heads)
        if self.rope_embedding is not None:
            if token_positions is None:
                # positions: (batch, seq_len)
                token_positions = torch.arange(x.shape[1], device=x.device)
                token_positions = token_positions.unsqueeze(0).expand(x.shape[0], -1)
            Q = self.rope_embedding(Q, token_positions)
            K = self.rope_embedding(K, token_positions)
        causal_mask = torch.tril(torch.ones(Q.shape[2], Q.shape[2], device=Q.device))
        causal_self_attention = Utility().scaled_dot_product_attention(Q, K, V, causal_mask)
        causal_self_attention = rearrange(causal_self_attention, "b h s d -> b s (h d)")
        return self.W_o(causal_self_attention)
