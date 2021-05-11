import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size: int, n_heads: int, dropout: float = 0.):
        super().__init__()
        self.emb_size = emb_size
        self.n_heads = n_heads

        self.q, self.k, self.v = [nn.Linear(emb_size, emb_size) for _ in range(3)]

        self.out = nn.Linear(emb_size, emb_size)
        self.dropout = dropout

    def forward(self, x: Tensor):
        queries, keys, values = [rearrange(m(x), 'b n (h d) -> b h n d', h=self.n_heads) for m in [self.q, self.k, self.v]]

        score = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        probs = F.softmax(score / (self.emb_size / self.n_heads)**.5, dim=-1)

        context = torch.einsum('bhal, bhlv -> bhav ', F.dropout(probs, p=self.dropout), values)
        context = rearrange(context, "b h n d -> b n (h d)")

        attn = F.dropout(self.out(context), p=self.dropout)

        return attn, probs

class MLP(nn.Module):
    def __init__(self, emb_size: int, mlp_dim: int, dropout: float = 0.):
        super().__init__()
        self.fc1 = nn.Linear(emb_size, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, emb_size)
        self.dropout = dropout

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = F.dropout(F.gelu(self.fc1(x)), p=self.dropout)
        x = F.dropout(self.fc2(x), p=self.dropout)
        return x

class Embeddings(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 emb_size: int = 768,
                 img_size: int = 224,
                 # patch_embedding: nn.Module = patch_embedding_conv,
                 dropout: float = 0.):
        super().__init__()
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                         out_channels=emb_size,
                                         kernel_size=patch_size,
                                         stride=patch_size)
        self.pos_embeddings = nn.Parameter(torch.zeros(1, (img_size // patch_size)**2 + 1, emb_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_size))
        self.dropout = dropout

    def forward(self, x: Tensor) -> Tensor:
        cls_token = repeat(self.cls_token, '() n e -> b n e', b=x.size(0))

        x = self.patch_embeddings(x)
        x = rearrange(x, 'b e (h) (w) -> b (h w) e')
        x = torch.cat([cls_token, x], dim=1)

        embeddings = x + self.pos_embeddings
        return F.dropout(embeddings, p =self.dropout)

class EncoderBlock(nn.Module):
    def __init__(self, emb_size: int, mlp_dim: int, n_heads: int,
                 mlp_dropout: float = 0., attn_dropout: float = 0.):
        super().__init__()
        self.emb_size = emb_size
        self.attn_norm = nn.LayerNorm(emb_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(emb_size, eps=1e-6)
        self.ffn = MLP(emb_size=emb_size,
                       mlp_dim=mlp_dim,
                       dropout=mlp_dropout)
        self.attn = MultiHeadAttention(emb_size=emb_size,
                                       n_heads=n_heads,
                                       dropout=attn_dropout)

    def forward(self, x: Tensor) -> (Tensor, [Tensor]):
        _x, attn_probs = self.attn(self.attn_norm(x))
        x = _x + x
        x = x + self.ffn(self.ffn_norm(x))
        return x, attn_probs

class Encoder(nn.Module):

    def __init__(self, emb_size: int, depth: int, n_heads: int, mlp_dim: int,
                 attn_dropout: float = 0., mlp_dropout: float = 0.):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(emb_size=emb_size, mlp_dim=mlp_dim, n_heads=n_heads,
                         mlp_dropout=mlp_dropout, attn_dropout=attn_dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_size, eps=1e-6)

    def forward(self, x: Tensor) -> (Tensor, [Tensor]):
        attn_probs = []
        for l in self.layers:
            x, attn_prob = l(x)
            attn_probs.append(attn_prob.detach())
        encoded = self.norm(x)
        return encoded, attn_probs

class ViT(nn.Module):

    def __init__(self,
                 in_channels: int = 3,
                 patch_size: int = 16,
                 img_size: int = 224,
                 n_classes: int = 1000,
                 emb_size: int = 768,
                 depth: int = 12,
                 n_heads: int = 12,
                 mlp_dim: int = 4 * 768,
                 attn_dropout: float = 0.,
                 mlp_dropout: float = 0.):
        super().__init__()
        self.embeddings = Embeddings(in_channels=in_channels, patch_size=patch_size,
                                     emb_size=emb_size, img_size=img_size, dropout=mlp_dropout)
        self.encoder = Encoder(emb_size=emb_size, depth=depth, n_heads=n_heads, mlp_dim=mlp_dim,
                               attn_dropout=attn_dropout, mlp_dropout=mlp_dropout)
        self.classification_head = nn.Linear(emb_size, n_classes)

    def forward(self, x: Tensor) -> (Tensor, [Tensor]):
        embedding_output = self.embeddings(x)
        encoded, attn_probs = self.encoder(embedding_output)
        logits = self.classification_head(encoded[:, 0])
        return logits, attn_probs
