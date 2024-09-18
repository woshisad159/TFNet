import torch.nn as nn
import torch
from einops import rearrange
import numpy as np

def key_padding_mask(l):
    """Blank is True
    Args:
        l: lenghts (b)
    Returns:
        mask: (b l)
    """
    mask = torch.zeros(len(l), max(l)).bool()
    for i, li in enumerate(l):
        mask[i, li:] = True
    return mask

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None, rpe_q=None, rpe_v=None):
        """
        Args:
            q: query (*, query_len, dim)
            k: key (*, key_len, dim)
            v: value (*, key_len, dim)
            mask: (*, query_len, key_len), True will be masked out
            rpe_q : (query_len, key_len, dim)
            rpe_v : (query_len, key_len, dim)
        Returns:
            context: (*, query_len, dim)
            alignment: (*, query_len, key_len)
        """
        dim = q.shape[-1]

        q /= dim ** 0.5
        energy = q @ k.transpose(-2, -1)

        if rpe_q is not None:
            energy += torch.einsum("...qd,qkd->...qk", q, rpe_q)

        if mask is not None:
            energy = energy.masked_fill(mask, np.NINF)

        alignment = torch.softmax(energy, dim=-1)
        context = self.dropout(alignment) @ v

        if rpe_v is not None:
            context += torch.einsum("...qk,qkd->...qd", alignment, rpe_v)

        return context, alignment

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dropout, rpe_k=0):
        assert (
            dim % heads == 0
        ), "dim should be a multiple of heads, \
            got {} and {}".format(
            dim, heads
        )

        super().__init__()

        self.dim = dim
        self.heads = heads

        self.q_linear = nn.Linear(dim, dim)
        self.k_linear = nn.Linear(dim, dim)
        self.v_linear = nn.Linear(dim, dim)

        self.rpe_k = rpe_k
        if rpe_k > 0:
            self.rpe_w = nn.Embedding(rpe_k * 2 + 1, 2 * dim // heads)

        self.attn = ScaledDotProductAttention(dropout)
        self.fc = nn.Linear(dim, dim)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: query (batch, query_len, dim)
            k: key (batch, key_len, dim)
            v: value (batch, key_len, dim)
            mask: (batch, query_len, key_len)
        Returns:
            context: (batch, query_len, dim)
            alignment: (bs, head, ql, kl)
        """

        bs, ql, kl = (*q.shape[:2], k.shape[1])

        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        split_heads = lambda x: rearrange(x, "b t (h d) -> b h t d", h=self.heads)
        q, k, v = map(split_heads, (q, k, v))

        # add head dim for mask
        if mask is not None:
            mask = mask.unsqueeze(1)

        if self.rpe_k > 0:
            distance = self.relative_distance(max(ql, kl), self.rpe_k)
            distance = distance[:ql, :kl].to(q.device)
            rpe_q, rpe_v = self.rpe_w(distance).chunk(2, dim=-1)
            context, alignment = self.attn(q, k, v, mask, rpe_q, rpe_v)
        else:
            context, alignment = self.attn(q, k, v, mask)

        # swap len and head back
        context = rearrange(context, "b h t d -> b t (h d)")
        context = self.fc(context)

        return context, alignment

    @staticmethod
    def relative_distance(length, k):
        indices = torch.arange(length)
        indices = indices.unsqueeze(1).expand(-1, length)
        distance = indices - indices.transpose(0, 1)
        distance = distance.clamp(-k, k) + k
        return distance

class PositionwiseFeedForward(nn.Module):
    def __init__(self, dim, ffn_dim, dropout):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w2(self.dropout(torch.relu(self.w1(x))))

class PreNorm(nn.Module):
    def __init__(self, dim, model):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.model = model

    def forward(self, x):
        return self.model(self.norm(x))

class Residual(nn.Sequential):
    def __init__(self, *layers):
        super().__init__(*layers)

    def forward(self, x):
        return super().forward(x) + x

class Applier(nn.Module):
    def __init__(self, model, applier):
        super().__init__()
        self.model = model
        self.applier = applier

    def forward(self, x):
        return self.applier(self.model, x)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, heads, dropout=0.1, rpe_k=0):
        super().__init__()
        attn = MultiHeadAttention(dim, heads, dropout, rpe_k)
        ffn = PositionwiseFeedForward(dim, 4 * dim, dropout)
        wrap = lambda m: Residual(PreNorm(dim, m), nn.Dropout(dropout))
        self.attn = wrap(Applier(attn, lambda m, x: m(x, x, x, self.xm)[0]))
        self.ffn = wrap(ffn)

    def forward(self, x, xm):
        # hack the mask here
        self.xm = xm
        x = self.attn(x)
        del self.xm
        x = self.ffn(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim, heads, num_layers, dropout=0.1, rpe_k=8):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norm = nn.LayerNorm(dim)
        for i in range(num_layers):
            self.layers += [
                TransformerEncoderLayer(
                    dim=dim,
                    heads=heads,
                    dropout=dropout,
                    rpe_k=rpe_k,
                )
            ]

    def forward(self, x):
        """
        Args:
            x: [(t d)]
        Returns:
            x: [(t d)]
        """
        xl = list(map(len, x))
        # x = pad_sequence(x, True)
        xm = key_padding_mask(xl).to(x.device)
        xm = xm.unsqueeze(dim=1)  # repeat mask for all targets
        for layer in self.layers:
            x = layer(x, xm)
        x = self.norm(x)
        # return unpad_padded(x, xl)
        return x