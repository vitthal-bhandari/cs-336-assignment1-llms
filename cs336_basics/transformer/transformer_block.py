import torch
import torch.nn as nn
from cs336_basics.transformer import MultiheadSelfAttention, PositionwiseFFN, RMSNormModule

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float | None = None, max_seq_len: int | None = None, device = None, dtype = None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.multihead_attention = MultiheadSelfAttention(d_model, num_heads, theta, max_seq_len, **factory_kwargs)
        self.rms_norm1 = RMSNormModule(d_model, **factory_kwargs)
        self.positionwise_ffn = PositionwiseFFN(d_model, d_ff, **factory_kwargs)
        self.rms_norm2 = RMSNormModule(d_model, **factory_kwargs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = x + self.multihead_attention(self.rms_norm1(x))
        y2 = y1 + self.positionwise_ffn(self.rms_norm2(y1))
        return y2