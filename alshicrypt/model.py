import torch.nn as nn

class Architecture(nn.Module):
    """Tiny per-character model: Embedding -> Linear"""
    def __init__(self, num_chars: int, emb_dim: int = 64):
        super().__init__()
        self.emb = nn.Embedding(num_chars, emb_dim)
        self.out = nn.Linear(emb_dim, num_chars)
    def forward(self, x):          # x: [B]
        e = self.emb(x)            # [B, E]
        return self.out(e)         # [B, V]
