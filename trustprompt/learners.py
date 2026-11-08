"""Prompt learners used as base models for GMT. Same interface for all of them:

  model(images, classnames)            -> logits (B, C), scaled by CLIP's logit scale
  model.zero_shot(images, classnames)  -> frozen hand-crafted-prompt CLIP logits (B, C)
  model.trainable_parameters()         -> list of learnable tensors
  train_loss(model, images, labels, classnames) -> scalar training loss

  coop    CoOp (Zhou et al., IJCV 2022): shared learnable text context, frozen image encoder
  kgcoop  KgCoOp (Yao et al., CVPR 2023): CoOp + lambda * (1 - cos(t_learned, t_handcrafted))
  ivlp    Independent V-L prompting (baseline in MaPLe, CVPR 2023): deep text prompts and deep
          vision prompts, learned separately (no coupling)
  maple   MaPLe (Khattak et al., CVPR 2023): trustprompt/maple.py, unchanged
  promptsrc  PromptSRC (Khattak et al., ICCV 2023): IVLP prompts + self-regularisation towards frozen
          CLIP (text L1, image L1, logit KL) + Gaussian prompt aggregation (GPA) over epochs
  cocoop  CoCoOp (Zhou et al., CVPR 2022): CoOp context shifted per image by a small meta-net

DEFAULTS holds each method's official base-to-novel hyperparameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from .model import TextEncoder, TextPromptLearner, block_forward
from .datasets import TEMPLATES
from .maple import MaPLe
from .imagenet_templates import IMAGENET_TEMPLATES

# Official base-to-novel configs of each method's released code (ViT-B/16, 16 shots):
#   MaPLe   multimodal-prompt-learning  MaPLe/vit_b16_c2_ep5_batch4_2ctx.yaml      depth 9, 2 ctx
#   IVLP    multimodal-prompt-learning  IVLP/vit_b16_c2_ep5_batch4_2+2ctx.yaml    depth 12 (V and L), 2 ctx
#   CoOp    KaiyangZhou/CoOp            scripts/cocoop/base2new_train.sh -> CoOp/vit_b16_ctxv1.yaml
#   KgCoOp  htyao89/KgCoOp              scripts/base2new_train.sh -> KgCoOp/vit_b16_ep100_ctxv1.yaml, W=8.0
DEFAULTS = {
    "maple":  dict(depth=9,  n_ctx=2, epochs=5,   bs=4,   lr=0.0035),
    "ivlp":   dict(depth=12, n_ctx=2, epochs=5,   bs=4,   lr=0.0035),
    "coop":   dict(depth=1,  n_ctx=4, epochs=200, bs=32,  lr=0.002),
    "kgcoop": dict(depth=1,  n_ctx=4, epochs=100, bs=128, lr=0.002, kg_lambda=8.0),
    "promptsrc": dict(depth=9, n_ctx=4, epochs=20, bs=4,  lr=0.0025),
    "cocoop": dict(depth=1,  n_ctx=4, epochs=10,  bs=1,   lr=0.002),
}
#   PromptSRC  muzairkhattak/PromptSRC  PromptSRC/vit_b16_c2_ep20_batch4_4+4ctx.yaml: depth 9 (V and L),
#              4 ctx, text/image L1 weights 25/10, GPA mean 15 std 1, fp16
#   CoCoOp     KaiyangZhou/CoOp  CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml, fp16


class _ZeroShotMixin:
    """Frozen hand-crafted-prompt CLIP; identical for every learner, so GMT's gate sees the same
    signal whatever the base model is."""

    @torch.no_grad()
    def zero_shot(self, images, classnames):
        dev = images.device
        tok = clip.tokenize([self.template.format(c) for c in classnames]).to(dev)
        t = F.normalize(self.cm.encode_text(tok).float(), dim=-1)
        f = F.normalize(self.cm.encode_image(images).float(), dim=-1)
        return self.logit_scale * f @ t.t()

    @torch.no_grad()
    def handcrafted_text(self, classnames, device):
        key = tuple(classnames)
        if key not in self._hc:
            tok = clip.tokenize([self.template.format(c) for c in classnames]).to(device)
            self._hc[key] = F.normalize(self.cm.encode_text(tok).float(), dim=-1)
        return self._hc[key]

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("cm.")]


class CoOp(_ZeroShotMixin, nn.Module):
    def __init__(self, cm, dataset, n_ctx=4, ctx_init="a photo of a"):
        super().__init__()
        self.cm = cm
        for p in cm.parameters():
            p.requires_grad_(False)
        self.prompt = TextPromptLearner(cm, n_ctx=n_ctx, ctx_init=ctx_init)
        self.text_encoder = TextEncoder(cm)       # cm.transformer keeps CLIP's causal mask
        self.template = TEMPLATES.get(dataset, "a photo of a {}.")
        self.logit_scale = float(cm.logit_scale.exp())
        self._hc = {}

    def text_features(self, classnames):
        prompts, tok = self.prompt(classnames)
        return F.normalize(self.text_encoder(prompts, tok).float(), dim=-1)

    def image_features(self, images):
        with torch.no_grad():                     # image encoder is frozen in CoOp/KgCoOp
            return F.normalize(self.cm.encode_image(images).float(), dim=-1)

    def forward(self, images, classnames):
        return self.logit_scale * self.image_features(images) @ self.text_features(classnames).t()


class KgCoOp(CoOp):
    def __init__(self, cm, dataset, n_ctx=4, ctx_init="a photo of a", kg_lambda=8.0):
        super().__init__(cm, dataset, n_ctx, ctx_init)
        self.kg_lambda = kg_lambda

    def train_loss(self, images, labels, classnames):
        t = self.text_features(classnames)
        logits = self.logit_scale * self.image_features(images) @ t.t()
        t0 = self.handcrafted_text(classnames, t.device)
        kg = 1.0 - (t * t0).sum(-1).mean()        # 1 - mean cosine, as in the official code
        return F.cross_entropy(logits.float(), labels) + self.kg_lambda * kg


class IVLP(_ZeroShotMixin, nn.Module):
    """Deep prompts in both encoders, as MaPLe but with separate vision prompts V instead of F(P)."""

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
        ctx = torch.randn(depth, n_ctx, d_t) * 0.02
        if ctx_init:
            with torch.no_grad():
                emb = cm.token_embedding(clip.tokenize(ctx_init))[0, 1:1 + n_ctx]
            if emb.shape[0] == n_ctx:
                ctx[0] = emb
        self.P = nn.Parameter(ctx)                                   # text prompts
        self.V = nn.Parameter(torch.randn(depth, n_ctx, d_v) * 0.02)  # vision prompts, independent
        self.template = TEMPLATES.get(dataset, "a photo of a {}.")
        self.logit_scale = float(cm.logit_scale.exp())
        self._tok, self._hc = {}, {}

    # text side identical to MaPLe
    _tokens = MaPLe._tokens
    _causal_mask = MaPLe._causal_mask
    text_features = MaPLe.text_features

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
                p = self.V[i].to(x.dtype).unsqueeze(1).expand(-1, B, -1)
                x = torch.cat([x, p], 0) if i == 0 else torch.cat([x[:-n], p], 0)
            x = block_forward(blk, x)
        return F.normalize((v.ln_post(x[0]) @ v.proj).float(), dim=-1)

    def forward(self, images, classnames):
        return self.logit_scale * self.image_features(images) @ self.text_features(classnames).t()


class PromptSRC(IVLP):
    def __init__(self, cm, dataset, depth=9, n_ctx=4, text_w=25.0, image_w=10.0,
                 gpa_mean=15.0, gpa_std=1.0):
        super().__init__(cm, dataset, depth, n_ctx)
        self.text_w, self.image_w = text_w, image_w
        self.gpa_mean, self.gpa_std = gpa_mean, gpa_std
        self._fixed, self._gpa, self._gw = {}, None, None

    @torch.no_grad()
    def fixed_text(self, classnames, device):
        """Mean of frozen CLIP text features over the 60 ImageNet templates, then L2-normalised."""
        key = tuple(classnames)
        if key not in self._fixed:
            acc = 0
            for t in IMAGENET_TEMPLATES:
                tok = clip.tokenize([t.replace("{}", c) for c in classnames]).to(device)
                acc = acc + self.cm.encode_text(tok).float()
            self._fixed[key] = F.normalize(acc / len(IMAGENET_TEMPLATES), dim=-1)
        return self._fixed[key]

    def train_loss(self, images, labels, classnames):
        t = self.text_features(classnames)
        f = self.image_features(images)
        logits = (self.logit_scale * f @ t.t()).float()
        fixed = self.fixed_text(classnames, t.device)
        with torch.no_grad():
            zf = F.normalize(self.cm.encode_image(images).float(), dim=-1)
            zl = self.logit_scale * zf @ fixed.t()
        l_text = F.l1_loss(t.float(), fixed) * self.text_w
        l_img = F.l1_loss(f.float(), zf) * self.image_w
        l_kl = F.kl_div(F.log_softmax(logits, 1), F.log_softmax(zl, 1), reduction="sum",
                        log_target=True) / logits.numel()
        return F.cross_entropy(logits, labels) + l_kl + l_text + l_img

    @torch.no_grad()
    def end_epoch(self, ep, total):
        """Gaussian prompt aggregation: weighted average of the prompts after every epoch."""
        if self._gw is None:
            x = torch.arange(1, total + 1, dtype=torch.float64)
            g = torch.exp(-0.5 * ((x - self.gpa_mean) / self.gpa_std) ** 2)
            self._gw = (g / g.sum()).tolist()
        params = dict((n, p) for n, p in self.named_parameters() if p.requires_grad)
        w = self._gw[ep]
        if self._gpa is None:
            self._gpa = {n: w * p.detach().clone() for n, p in params.items()}
        else:
            for n, p in params.items():
                self._gpa[n] += w * p.detach()
        if ep + 1 == total:
            for n, p in params.items():
                p.copy_(self._gpa[n])


class CoCoOp(CoOp):
    def __init__(self, cm, dataset, n_ctx=4, ctx_init="a photo of a"):
        super().__init__(cm, dataset, n_ctx, ctx_init)
        vis_dim = cm.visual.output_dim
        ctx_dim = cm.ln_final.weight.shape[0]
        self.meta_net = nn.Sequential(nn.Linear(vis_dim, vis_dim // 16), nn.ReLU(inplace=True),
                                      nn.Linear(vis_dim // 16, ctx_dim))

    def logits_from_feats(self, f, classnames):
        """f: (B, D) L2-normalised image features -> (B, C) logits, one prompt set per image."""
        tok, prefix, suffix = self.prompt._build(classnames, f.device)
        B, C = f.shape[0], len(classnames)
        ctx = self.prompt.ctx.unsqueeze(0) + self.meta_net(f).unsqueeze(1)          # (B, n_ctx, d)
        ctx = ctx.unsqueeze(1).expand(-1, C, -1, -1).to(prefix.dtype)
        pre = prefix.unsqueeze(0).expand(B, -1, -1, -1)
        suf = suffix.unsqueeze(0).expand(B, -1, -1, -1)
        prompts = torch.cat([pre, ctx, suf], 2).reshape(B * C, -1, prefix.shape[-1])
        t = self.text_encoder(prompts, tok.repeat(B, 1))
        t = F.normalize(t.float(), dim=-1).reshape(B, C, -1)
        return self.logit_scale * torch.einsum("bd,bcd->bc", f.float(), t)

    def forward(self, images, classnames):
        return self.logits_from_feats(self.image_features(images), classnames)


def build_learner(name, cm, dataset, depth, n_ctx, kg_lambda=8.0):
    if name == "maple":
        return MaPLe(cm, dataset, depth, n_ctx)
    if name == "ivlp":
        return IVLP(cm, dataset, depth, n_ctx)
    if name == "coop":
        return CoOp(cm, dataset, n_ctx)
    if name == "kgcoop":
        return KgCoOp(cm, dataset, n_ctx, kg_lambda=kg_lambda)
    if name == "promptsrc":
        return PromptSRC(cm, dataset, depth, n_ctx)
    if name == "cocoop":
        return CoCoOp(cm, dataset, n_ctx)
    raise ValueError(f"unknown learner {name}")


def train_loss(model, images, labels, classnames):
    if hasattr(model, "train_loss"):
        return model.train_loss(images, labels, classnames)
    return F.cross_entropy(model(images, classnames).float(), labels)


def end_epoch(model, ep, total):
    if hasattr(model, "end_epoch"):
        model.end_epoch(ep, total)


def fill_defaults(a):
    """Fill argparse values left as None from DEFAULTS[a.learner]."""
    for k, v in DEFAULTS[a.learner].items():
        if getattr(a, k, None) is None:
            setattr(a, k, v)
    return a
