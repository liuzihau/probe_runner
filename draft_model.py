"""Self-Conditioned Δh Drafter — Part I model architecture (scaffold).

Implements `fork_bounded_surrogate/draft_model_design.md` Part I, Phase-A (one-shot) path. Self-contained
(torch only) so it can be unit-tested / toy-trained anywhere; warm-start from DMax bottom layers is a TODO
hook (load_state_dict from the heavy's decoder layers + freeze embed/lm_head).

Design (per draft forward, over the DENOISE/masked region as queries):
  inputs from the heavy's PREVIOUS run:
    o_denoise      [B, C]      canvas token ids for the masked block (mask id where masked)
    heavy_logits   [B, C, V]   heavy logits over the block            -> soft-embed S (DiffusionGemma)
    H_sel_denoise  [B, C, m*d] heavy hidden @ sel_layers, denoise pos  -> FuseDenoise -> F_dn (added at input)
    H_sel_prefix   [B, P, m*d] heavy hidden @ sel_layers, prefix pos   -> FusePrefix_l -> per-layer KV inject (DFlash+D3)
    h_last_denoise [B, C, d]   heavy LAST-LAYER hidden, denoise pos     -> Δh residual base
  forward:
    S    = GatedMLP( softmax(heavy_logits/τ) · W_E )                    # self-conditioning soft-embed
    F_dn = FuseDenoise( H_sel_denoise )
    x    = E(o_denoise) + S + F_dn                                      # input assembly
    for l: x = Layer_l(x, prefix_kv = FusePrefix_l(H_sel_prefix))      # DFlash-style per-layer KV injection
    Δh   = W_δ · norm(x) ;  h_draft = h_last_denoise + Δh
    logits = FROZEN lm_head( final_norm(h_draft) )                     # heavy head, frozen
    conf   = σ( W_c · norm(x) )                                        # "will the heavy accept this draft?"

References: DFlash `model.py` (KV injection, fuse), DiffusionGemma §2.1 (soft-embed + τ), DMax/LLaDA-2.0 layer.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- config
@dataclass
class DraftConfig:
    vocab_size: int = 512
    hidden_size: int = 128
    n_layers: int = 5                  # L (DMax-bottom thin)
    n_heads: int = 4
    n_kv_heads: int = 4                # GQA: set < n_heads to share
    intermediate_size: int = 384
    sel_layers: tuple = (1, 3, 5)      # which heavy layers feed the fuses (m = len)
    per_layer_prefix_fuse: bool = True # D3: per-layer prefix fuse vs one shared
    rms_eps: float = 1e-5
    rope_theta: float = 10000.0
    soft_embed_temp: float = 0.8       # τ (DiffusionGemma; can anneal at inference)
    mask_token_id: int = 0

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.n_heads
    @property
    def m(self) -> int:
        return len(self.sel_layers)


# ---------------------------------------------------------------- primitives
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x):
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.w * x).to(self.w.dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def rope_cos_sin(positions, head_dim, theta, device, dtype):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    ang = positions.float()[:, None] * inv[None, :]          # [T, hd/2]
    emb = torch.cat([ang, ang], dim=-1)                      # [T, hd]
    return emb.cos().to(dtype), emb.sin().to(dtype)


class GatedMLP(nn.Module):
    """SwiGLU-style gate; used both for the soft-embed injection and the FFN."""
    def __init__(self, d_in, d_hidden, d_out=None):
        super().__init__()
        d_out = d_out or d_in
        self.gate = nn.Linear(d_in, d_hidden, bias=False)
        self.up = nn.Linear(d_in, d_hidden, bias=False)
        self.down = nn.Linear(d_hidden, d_out, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    b, h, t, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, t, d).reshape(b, h * n_rep, t, d)


# ---------------------------------------------------------------- attention with KV injection (DFlash)
class DraftAttention(nn.Module):
    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.cfg = cfg; hd = cfg.head_dim
        self.q = nn.Linear(cfg.hidden_size, cfg.n_heads * hd, bias=False)
        self.k = nn.Linear(cfg.hidden_size, cfg.n_kv_heads * hd, bias=False)
        self.v = nn.Linear(cfg.hidden_size, cfg.n_kv_heads * hd, bias=False)
        self.o = nn.Linear(cfg.n_heads * hd, cfg.hidden_size, bias=False)
        self.qn = RMSNorm(hd, cfg.rms_eps); self.kn = RMSNorm(hd, cfg.rms_eps)

    def forward(self, x, prefix_kv, cos, sin):
        """x: [B,C,d] denoise queries. prefix_kv: [B,P,d] context to inject into K/V (DFlash)."""
        cfg = self.cfg; B, C, _ = x.shape; hd = cfg.head_dim
        P = prefix_kv.shape[1] if prefix_kv is not None else 0
        q = self.qn(self.q(x).view(B, C, cfg.n_heads, hd)).transpose(1, 2)          # [B,H,C,hd]
        # keys/values from BOTH prefix context and the denoise tokens, via the SAME projections (DFlash)
        kx = self.k(x); vx = self.v(x)
        if P:
            kc = self.k(prefix_kv); vc = self.v(prefix_kv)
            kcat = torch.cat([kc, kx], dim=1); vcat = torch.cat([vc, vx], dim=1)    # [B,P+C,*]
        else:
            kcat, vcat = kx, vx
        T = P + C
        k = self.kn(kcat.view(B, T, cfg.n_kv_heads, hd)).transpose(1, 2)            # [B,Hkv,T,hd]
        v = vcat.view(B, T, cfg.n_kv_heads, hd).transpose(1, 2)
        # RoPE: positions 0..T-1 for keys; queries are the LAST C positions
        q = q * cos[-C:].view(1, 1, C, hd) + rotate_half(q) * sin[-C:].view(1, 1, C, hd)
        k = k * cos[:T].view(1, 1, T, hd) + rotate_half(k) * sin[:T].view(1, 1, T, hd)
        k = repeat_kv(k, cfg.n_heads // cfg.n_kv_heads); v = repeat_kv(v, cfg.n_heads // cfg.n_kv_heads)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=False)  # bidirectional
        return self.o(out.transpose(1, 2).reshape(B, C, -1))


class DraftLayer(nn.Module):
    def __init__(self, cfg: DraftConfig):
        super().__init__()
        self.attn = DraftAttention(cfg)
        self.mlp = GatedMLP(cfg.hidden_size, cfg.intermediate_size)
        self.ln1 = RMSNorm(cfg.hidden_size, cfg.rms_eps)
        self.ln2 = RMSNorm(cfg.hidden_size, cfg.rms_eps)
    def forward(self, x, prefix_kv, cos, sin):
        x = x + self.attn(self.ln1(x), prefix_kv, cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


# ---------------------------------------------------------------- the drafter
class DraftModel(nn.Module):
    def __init__(self, cfg: DraftConfig, frozen_embed: nn.Embedding | None = None,
                 frozen_lm_head: nn.Linear | None = None, frozen_final_norm: RMSNorm | None = None):
        super().__init__()
        self.cfg = cfg
        d, md = cfg.hidden_size, cfg.m * cfg.hidden_size
        # frozen heavy pieces (random for toy; load real ones for warm-start)
        self.embed = frozen_embed or nn.Embedding(cfg.vocab_size, d)
        self.lm_head = frozen_lm_head or nn.Linear(d, cfg.vocab_size, bias=False)
        self.final_norm = frozen_final_norm or RMSNorm(d, cfg.rms_eps)
        for p in (*self.embed.parameters(), *self.lm_head.parameters(), *self.final_norm.parameters()):
            p.requires_grad_(False)
        # soft-embed (DiffusionGemma): probs·W_E then a gated MLP
        self.soft_embed_mlp = GatedMLP(d, cfg.intermediate_size, d)
        # fuses (DFlash): denoise = 1; prefix = L (D3) or 1 shared
        self.fuse_dn = nn.Sequential(nn.Linear(md, d, bias=False), RMSNorm(d, cfg.rms_eps))
        n_pre = cfg.n_layers if cfg.per_layer_prefix_fuse else 1
        self.fuse_pre = nn.ModuleList(
            [nn.Sequential(nn.Linear(md, d, bias=False), RMSNorm(d, cfg.rms_eps)) for _ in range(n_pre)])
        # body + heads
        self.layers = nn.ModuleList([DraftLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(d, cfg.rms_eps)
        self.delta_head = nn.Linear(d, d, bias=False)         # W_δ  (Δh)
        self.conf_head = nn.Linear(d, 1)                      # W_c  (confidence)

    def forward(self, o_denoise, heavy_logits, H_sel_denoise, H_sel_prefix, h_last_denoise,
                position_ids=None, tau=None):
        cfg = self.cfg; B, C = o_denoise.shape; P = H_sel_prefix.shape[1]
        dev, dt = o_denoise.device, self.embed.weight.dtype
        tau = tau or cfg.soft_embed_temp
        # --- input assembly: E(o) + soft-embed(S) + F_dn ---
        probs = torch.softmax(heavy_logits.float() / tau, dim=-1).to(dt)
        S = self.soft_embed_mlp(probs @ self.embed.weight)                # [B,C,d]
        F_dn = self.fuse_dn(H_sel_denoise)                               # [B,C,d]
        x = self.embed(o_denoise) + S + F_dn
        # --- per-layer prefix fuse (D3) -> KV injection ---
        F_pre = [f(H_sel_prefix) for f in self.fuse_pre]                 # each [B,P,d]
        # --- RoPE for the concatenated [prefix; denoise] of length P+C ---
        pos = position_ids if position_ids is not None else torch.arange(P + C, device=dev)
        cos, sin = rope_cos_sin(pos, cfg.head_dim, cfg.rope_theta, dev, dt)
        for i, layer in enumerate(self.layers):
            x = layer(x, F_pre[i if cfg.per_layer_prefix_fuse else 0], cos, sin)
        hb = self.norm(x)
        # --- Δh residual on the heavy's last hidden, then FROZEN head ---
        h_draft = h_last_denoise + self.delta_head(hb)
        logits = self.lm_head(self.final_norm(h_draft))
        conf = torch.sigmoid(self.conf_head(hb)).squeeze(-1)            # [B,C]
        return {"logits": logits, "conf": conf, "h_draft": h_draft, "delta": self.delta_head(hb)}

    def num_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------- toy data (learnable synthetic task)
def make_toy_batch(cfg: DraftConfig, B, C, P, device, embed_weight, noise=0.6,
                   hard_frac=0.3, hard_noise=1.5):
    """A controllable task. A `hard_frac` of DENOISE positions carry NO usable signal (pure noise) — the
    drafter can't recover them (→ wrong → negatives), while their inputs are statistically distinguishable,
    so the confidence head can learn to flag them. The rest carry the true token (noisily) and are recoverable.
    Confidence label = (drafter argmax == true). draft_acc ≈ 1 − hard_frac; a good conf head ⇒ conf_auc ≫ 0.5."""
    V = cfg.vocab_size
    y = torch.randint(0, V, (B, C), device=device)
    Ey = F.embedding(y, embed_weight)
    hard = (torch.rand(B, C, device=device) < hard_frac)
    easy = (~hard).float().unsqueeze(-1)                                 # [B,C,1]
    def nz(t, s): return t + s * torch.randn_like(t)
    def sig(view_noise):                                                 # easy: y+noise ; hard: pure junk
        return easy * nz(Ey, view_noise) + (1 - easy) * (hard_noise * torch.randn_like(Ey))
    H_sel_denoise = torch.cat([sig(noise * (1 + j)) for j in range(cfg.m)], dim=-1)
    hl = easy * (nz(Ey, noise) @ embed_weight.t()) \
         + (1 - easy) * ((hard_noise * torch.randn_like(Ey)) @ embed_weight.t())   # easy peaked / hard ~flat
    h_last_denoise = sig(1.2)
    o_denoise = torch.full((B, C), cfg.mask_token_id, device=device)
    yp = torch.randint(0, V, (B, P), device=device)                     # prefix context = clean (committed)
    Eyp = F.embedding(yp, embed_weight)
    H_sel_prefix = torch.cat([nz(Eyp, noise * (1 + j)) for j in range(cfg.m)], dim=-1)
    return dict(o_denoise=o_denoise, heavy_logits=hl, H_sel_denoise=H_sel_denoise,
                H_sel_prefix=H_sel_prefix, h_last_denoise=h_last_denoise), y


def toy_train(cfg: DraftConfig, steps=400, B=16, C=16, P=24, lr=3e-3, device="cpu", w_pos=1.0, w_neg=3.0,
              hard_frac=0.3, noise=0.6):
    torch.manual_seed(0)
    model = DraftModel(cfg).to(device)
    # tie lm_head to embedding (so the synthetic task is well-posed), both frozen
    with torch.no_grad():
        model.lm_head.weight.copy_(model.embed.weight)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    print(f"[toy] trainable params={model.num_trainable():,}  device={device}  hard_frac={hard_frac}")
    for step in range(steps):
        batch, y = make_toy_batch(cfg, B, C, P, device, model.embed.weight,
                                  noise=noise, hard_frac=hard_frac)
        out = model(**batch)
        L_tok = F.cross_entropy(out["logits"].reshape(-1, cfg.vocab_size), y.reshape(-1))
        L_delta = 1e-3 * out["delta"].pow(2).mean()
        a = (out["logits"].argmax(-1) == y).float()                     # accept label
        c = out["conf"].clamp(1e-5, 1 - 1e-5)
        L_conf = -(w_pos * a * c.log() + w_neg * (1 - a) * (1 - c).log()).mean()  # asymmetric (B1 fix)
        loss = L_tok + L_delta + L_conf
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0 or step == steps - 1:
            acc = a.mean().item()
            auc = _auc(out["conf"].detach().reshape(-1), a.reshape(-1))
            print(f"[toy] step {step:4d}  loss {loss.item():.3f}  tok {L_tok.item():.3f}  "
                  f"conf {L_conf.item():.3f}  draft_acc {acc:.3f}  conf_auc {auc:.3f}")
    return model


def _auc(scores, labels):
    s = scores.float().cpu().numpy(); y = labels.bool().cpu().numpy()
    import numpy as np
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg]); order = allv.argsort(kind="mergesort")
    r = np.empty(len(allv)); r[order] = np.arange(1, len(allv) + 1)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# ---------------------------------------------------------------- selftest / cli
def _selftest():
    cfg = DraftConfig(vocab_size=64, hidden_size=32, n_layers=3, n_heads=4, n_kv_heads=2,
                      intermediate_size=64, sel_layers=(1, 2, 3))
    m = DraftModel(cfg); B, C, P = 2, 5, 7
    batch, y = make_toy_batch(cfg, B, C, P, "cpu", m.embed.weight)
    out = m(**batch)
    assert out["logits"].shape == (B, C, cfg.vocab_size), out["logits"].shape
    assert out["conf"].shape == (B, C)
    assert out["h_draft"].shape == (B, C, cfg.hidden_size)
    # frozen pieces have no grad
    loss = out["logits"].sum() + out["conf"].sum(); loss.backward()
    assert m.embed.weight.grad is None and m.lm_head.weight.grad is None
    assert m.delta_head.weight.grad is not None and m.fuse_dn[0].weight.grad is not None
    # shared-fuse (DFlash) path also runs
    cfg2 = DraftConfig(vocab_size=64, hidden_size=32, n_layers=3, n_heads=4, n_kv_heads=2,
                       intermediate_size=64, sel_layers=(1, 2, 3), per_layer_prefix_fuse=False)
    m2 = DraftModel(cfg2)
    b2, _ = make_toy_batch(cfg2, 2, 4, 6, "cpu", m2.embed.weight)
    assert m2(**b2)["logits"].shape == (2, 4, cfg2.vocab_size)
    print("draft_model selftest OK")


def main():
    ap = argparse.ArgumentParser(description="Self-Conditioned Δh Drafter (Part I scaffold)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--toy_train", action="store_true")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--shared_fuse", action="store_true", help="use one shared prefix fuse (DFlash) instead of per-layer (D3)")
    ap.add_argument("--hard_frac", type=float, default=0.3, help="fraction of denoise positions with no usable signal (creates negatives for the conf head)")
    ap.add_argument("--noise", type=float, default=0.6)
    args = ap.parse_args()
    if args.selftest:
        _selftest(); return
    if args.toy_train:
        cfg = DraftConfig(hidden_size=args.hidden, n_layers=args.layers,
                          per_layer_prefix_fuse=not args.shared_fuse)
        toy_train(cfg, steps=args.steps, device=args.device,
                  hard_frac=args.hard_frac, noise=args.noise)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
