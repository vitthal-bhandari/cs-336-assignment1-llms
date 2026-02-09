import torch
import torch.nn as nn

class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device = None, dtype = None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.E = nn.Parameter(torch.empty(num_embeddings, embedding_dim, **factory_kwargs))
        nn.init.trunc_normal_(self.E, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.E[token_ids]