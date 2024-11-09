import copy
from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor
from torch.nn.functional import dropout
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
import torch.nn as nn
from einops import rearrange, repeat, reduce, pack, unpack
from torch import einsum
from einops_exts import rearrange_many
from rotary_embedding_torch import RotaryEmbedding

def exists(x):
    return x is not None

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
            self,
            x,
            pos_bias=None,
    ):  # temperal: 'b (h w) f c'  ; spatial :  'b f (h w) c'
        n, device = x.shape[-2], x.device

        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # if exists(focus_present_mask) and focus_present_mask.all():
        #     # if all batch samples are focusing on present
        #     # it would be equivalent to passing that token's values through to the output
        #     values = qkv[-1]
        #     return self.to_out(values)

        # split out heads

        q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        # if exists(focus_present_mask) and not (~focus_present_mask).all():
        #     attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
        #     attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

        #     mask = torch.where(
        #         rearrange(focus_present_mask, 'b -> b 1 1 1 1'),
        #         rearrange(attend_self_mask, 'i j -> 1 1 1 i j'),
        #         rearrange(attend_all_mask, 'i j -> 1 1 1 i j'),
        #     )

        #     sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

class Attention_2(nn.Module):
    def __init__(
            self,
            dim,
            heads=4,
            dim_head=32,
            rotary_emb=None
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_k = nn.Linear(dim, hidden_dim, bias=False)
        self.to_v = nn.Linear(dim, hidden_dim, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(
            self,
            q,
            k,
            v,
            pos_bias=None,
            focus_present_mask=None
    ):  # temperal: 'b (h w) f c'  ; spatial :  'b f (h w) c'

        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        # split out heads
        q = rearrange(q, '... n (h d) -> ... h n d', h=self.heads) # b, head, fn, c
        k = rearrange(k, '... n (h d) -> ... h n d', h=self.heads)
        v = rearrange(v, '... n (h d) -> ... h n d', h=self.heads)
        # q, k, v = rearrange_many(qkv, '... n (h d) -> ... h n d', h=self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum('... h i d, ... h j d -> ... h i j', q, k)

        # relative positional bias

        if exists(pos_bias):
            sim = sim + pos_bias

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum('... h i j, ... h j d -> ... h i d', attn, v)
        out = rearrange(out, '... h n d -> ... n (h d)')
        return self.to_out(out)

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedforwardLayer, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, rotary_emb):
        super(DecoderLayer, self).__init__()

        self.self_attn = Attention(dim = d_model, heads = num_heads, rotary_emb = rotary_emb) # , rotary_emb = rotary_emb)
        self.multihead_attn = Attention_2(dim = d_model, heads = num_heads, rotary_emb = rotary_emb)

        self.ffn = PositionwiseFeedforwardLayer(d_model, d_ff, dropout)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt = self.layer_norm1(tgt + self.dropout1(self.self_attn(tgt, tgt_mask)))
        tgt = self.layer_norm2(tgt + self.dropout2(self.multihead_attn(tgt, memory, memory, memory_mask)))
        tgt = self.layer_norm3(tgt + self.dropout3(self.ffn(tgt)))

        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward, dropout):
        super(TransformerDecoder, self).__init__()

        self.num_layers = num_layers
        rotary_emb = RotaryEmbedding(16)
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model = d_model, num_heads = num_heads, d_ff = dim_feedforward, dropout = dropout, rotary_emb = rotary_emb) for _ in range(num_layers)])
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        output = tgt

        for layer in self.decoder_layers:
            output = layer(output, memory, tgt_mask, memory_mask)

        return output