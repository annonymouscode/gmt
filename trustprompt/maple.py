"""MaPLe-style multi-modal prompt learning, instrumented to separate the two gradient paths.

MaPLe couples the branches: vision prompts are a linear map of the language prompts,
P~_i = F_i(P_i). The shared parameters (P, F) therefore receive gradient from two routes:

  text route    loss -> class text embeddings t_c -> language prompts P
  vision route  loss -> image embedding f(x)      -> vision prompts F(P)

We can isolate them by detaching one side of the cosine similarity. Comparing the two gradients
tells us whether the branches agree; MaPLe assumes they do ("mutual synergy") but never checks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from .model import TextEncoder, block_forward
from .datasets import TEMPLATES


class MaPLe(nn.Module):
    def __init__(self, cm, dataset, depth=9, n_ctx=2, ctx_init="a photo of a"):
        super().__init__()
        self.cm = cm
        for p in cm.parameters():
            p.requires_grad_(False)
        self.text_encoder = TextEncoder(cm)
        self.token_embedding = cm.token_embedding
        self.visual = cm.visual
        self.depth, self.n_ctx = depth, n_ctx
        d_t = cm.ln_final.weight.shape[0]
        d_v = cm.visual.conv1.out_channels
        # language prompts at each of the first `depth` layers; the first is initialised from text
        ctx = torch.randn(depth, n_ctx, d_t) * 0.02
        if ctx_init:
            with torch.no_grad():
                emb = cm.token_embedding(clip.tokenize(ctx_init))[0, 1:1 + n_ctx]
            if emb.shape[0] == n_ctx:
                ctx[0] = emb
        self.P = nn.Parameter(ctx)                                  # shared by both branches
        self.coupling = nn.ModuleList([nn.Linear(d_t, d_v) for _ in range(depth)])
        self.template = TEMPLATES.get(dataset, "a photo of a {}.")
        self.logit_scale = float(cm.logit_scale.exp())
        self._tok = {}

    def shared_parameters(self):
        """Parameters that both routes write into."""
        return [self.P] + [p for m in self.coupling for p in m.parameters()]

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("cm.")]

    def _tokens(self, classnames):
        key = tuple(classnames)
        if key not in self._tok:
            dev = self.P.device
            tok = clip.tokenize([f"X {' '.join(['X'] * (self.n_ctx - 1))} {c}." if self.n_ctx > 1
                                 else f"X {c}." for c in classnames]).to(dev)
            with torch.no_grad():
                emb = self.token_embedding(tok)
            self._tok[key] = (tok, emb)
        return self._tok[key]

    def _causal_mask(self, L, device, dtype):
        return torch.full((L, L), float("-inf"), device=device, dtype=dtype).triu(1)

    def text_features(self, classnames):
        tok, emb = self._tokens(classnames)
        C = len(classnames)
        x = emb.clone()
        x[:, 1:1 + self.n_ctx] = self.P[0].unsqueeze(0).expand(C, -1, -1).to(x.dtype)
        x = x + self.text_encoder.positional_embedding
        x = x.permute(1, 0, 2)
        mask = self._causal_mask(x.shape[0], x.device, x.dtype)   # CLIP's text encoder is causal
        for i, blk in enumerate(self.text_encoder.transformer.resblocks):
            if 0 < i < self.depth:
                x = torch.cat([x[:1], self.P[i].unsqueeze(1).expand(-1, C, -1).to(x.dtype),
                               x[1 + self.n_ctx:]], 0)
            x = block_forward(blk, x, mask)
        x = self.text_encoder.ln_final(x.permute(1, 0, 2))
        z = x[torch.arange(C), tok.argmax(-1)] @ self.text_encoder.text_projection
        return F.normalize(z.float(), dim=-1)

    def image_features(self, images):
        v = self.visual
        x = v.conv1(images.to(v.conv1.weight.dtype))
        B = x.shape[0]
        x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype) + torch.zeros(B, 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], 1) + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x).permute(1, 0, 2)
        n = self.n_ctx
        for i, blk in enumerate(v.transformer.resblocks):
            if i < self.depth:
                p = self.coupling[i](self.P[i]).to(x.dtype).unsqueeze(1).expand(-1, B, -1)
                x = torch.cat([x, p], 0) if i == 0 else torch.cat([x[:-n], p], 0)
            x = block_forward(blk, x)
        return F.normalize((v.ln_post(x[0]) @ v.proj).float(), dim=-1)

    def forward(self, images, classnames, route="both"):
        """route: 'both' = normal; 'text' = detach image side; 'vision' = detach text side."""
        f = self.image_features(images)
        t = self.text_features(classnames)
        if route == "text":
            f = f.detach()
        elif route == "vision":
            t = t.detach()
        return self.logit_scale * f @ t.t()

    @torch.no_grad()
    def zero_shot(self, images, classnames):
        tok = clip.tokenize([self.template.format(c) for c in classnames]).to(self.P.device)
        t = F.normalize(self.cm.encode_text(tok).float(), dim=-1)
        f = F.normalize(self.cm.encode_image(images).float(), dim=-1)
        return self.logit_scale * f @ t.t()


def route_gradients(model, images, labels, classnames, params):
    """Gradient of the loss w.r.t. `params` through the vision route and the text route."""
    out = {}
    for route in ("vision", "text"):
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(images, classnames, route=route).float(), labels)
        g = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        out[route] = torch.cat([(gi if gi is not None else torch.zeros_like(p)).reshape(-1)
                                for gi, p in zip(g, params)])
    model.zero_grad(set_to_none=True)
    return out["vision"], out["text"]


def conflict_cosine(gv, gt):
    d = gv.norm() * gt.norm()
    return float((gv @ gt) / d) if d > 0 else 0.0


def pcgrad(gv, gt):
    """Project out the conflicting component (PCGrad-style), then sum."""
    c = gv @ gt
    if c >= 0:
        return gv + gt, False
    gv_p = gv - (c / (gt @ gt).clamp_min(1e-12)) * gt
    gt_p = gt - (c / (gv @ gv).clamp_min(1e-12)) * gv
    return gv_p + gt_p, True


def set_flat_grad(params, flat):
    i = 0
    for p in params:
        n = p.numel()
        p.grad = flat[i:i + n].view_as(p).clone()
        i += n
