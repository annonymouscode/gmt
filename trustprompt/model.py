"""CLIP text encoder and prompt modules shared by the prompt learners.

One forward pass. Layers 1..k0 of the CLIP image encoder run unprompted (identical to
zero-shot CLIP). A trust estimator reads the layer-k0 CLS token and outputs g(x) in [0,1].
From layer k0+1, deep prompt tokens are inserted and every token's attention to them is
biased by log g(x); text classifiers are a g-weighted mix of learned-context and
hand-crafted embeddings. As g -> 0 the model reduces to zero-shot CLIP.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from .datasets import TEMPLATES


def block_forward(block, x, attn_bias=None):
    """ResidualAttentionBlock forward with an optional float attention bias (N*heads, L, L)."""
    h = block.ln_1(x)
    a = block.attn(h, h, h, need_weights=False, attn_mask=attn_bias)[0]
    x = x + a
    return x + block.mlp(block.ln_2(x))


class TextEncoder(nn.Module):
    def __init__(self, cm):
        super().__init__()
        self.transformer = cm.transformer
        self.positional_embedding = cm.positional_embedding
        self.ln_final = cm.ln_final
        self.text_projection = cm.text_projection

    def forward(self, prompts, tokenized):
        x = prompts + self.positional_embedding
        x = self.transformer(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.ln_final(x)
        return x[torch.arange(x.shape[0]), tokenized.argmax(-1)] @ self.text_projection


class TextPromptLearner(nn.Module):
    """CoOp-style shared learnable context, initialised from ctx_init."""

    def __init__(self, cm, n_ctx=4, ctx_init="a photo of a"):
        super().__init__()
        self.token_embedding = cm.token_embedding  # frozen, shared with CLIP
        width = cm.ln_final.weight.shape[0]
        self.n_ctx = n_ctx
        if ctx_init and len(ctx_init.split()) == n_ctx:
            with torch.no_grad():
                emb = cm.token_embedding(clip.tokenize(ctx_init))
            ctx = emb[0, 1:1 + n_ctx].clone()
            self.prefix_str = ctx_init
        else:
            ctx = torch.randn(n_ctx, width) * 0.02
            self.prefix_str = " ".join(["X"] * n_ctx)
        self.ctx = nn.Parameter(ctx.float())
        self._cache = {}

    def _build(self, classnames, device):
        key = (tuple(classnames), str(device))
        if key not in self._cache:
            tok = clip.tokenize([f"{self.prefix_str} {c}." for c in classnames]).to(device)
            with torch.no_grad():
                emb = self.token_embedding(tok)
            self._cache[key] = (tok, emb[:, :1], emb[:, 1 + self.n_ctx:])
        return self._cache[key]

    def forward(self, classnames, detach=False):
        tok, prefix, suffix = self._build(classnames, self.ctx.device)
        ctx = self.ctx.detach() if detach else self.ctx
        ctx = ctx.unsqueeze(0).expand(len(classnames), -1, -1).to(prefix.dtype)
        return torch.cat([prefix, ctx, suffix], 1), tok








