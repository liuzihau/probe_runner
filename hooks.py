"""Probe hooks: attention weights, value-projection norms, and per-layer hidden states.

For each model type ("llada" or "dream") this module:
  1. Monkey-patches the attention path so attention weights are exposed (SDPA otherwise hides them).
  2. Registers per-block forward hooks to capture residual-stream output at masked positions.
  3. Provides arm/disarm + per-block accumulation across the 8 firings expected per sample.

See ../T3_pruning_probe_step1to4.md §3, §5.2, §5.3, §5.4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Module-finding helpers
# ---------------------------------------------------------------------------

def _find_llada_blocks(model: nn.Module) -> list[nn.Module]:
    """Return list of LLaDABlock instances in document order."""
    out = []
    for m in model.modules():
        cls_name = type(m).__name__
        if cls_name in ("LLaDASequentialBlock", "LLaDALlamaBlock"):
            out.append(m)
    return out


def _find_dream_layers(model: nn.Module) -> list[nn.Module]:
    """Return list of DreamDecoderLayer instances in document order."""
    out = []
    for m in model.modules():
        if type(m).__name__ == "DreamDecoderLayer":
            out.append(m)
    return out


def _find_embedding(model: nn.Module) -> nn.Module:
    """Return the input token-embedding module."""
    # LLaDA: model.model.transformer.wte ; Dream: model.model.embed_tokens
    candidates = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Embedding) and ("wte" in name or "embed_tokens" in name):
            candidates.append((name, mod))
    if not candidates:
        # Fallback: any nn.Embedding at the top of the model.
        for mod in model.modules():
            if isinstance(mod, nn.Embedding):
                candidates.append(("", mod))
                break
    if not candidates:
        raise RuntimeError("Could not locate input token-embedding module.")
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Manual attention computation that exposes attn weights and v
# ---------------------------------------------------------------------------

def _manual_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      attn_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard SDPA computed in pure torch so we can keep `attn` for probing.

    q/k/v are already in [B, H, T, d_head] form, with GQA already broadcast (k/v repeat-interleaved
    to match q's head count). Returns (output [B, H, T_q, d_head], attn_weights [B, H, T_q, T_k]).
    """
    d_head = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_head)
    if attn_mask is not None:
        scores = scores + attn_mask
    attn = F.softmax(scores, dim=-1)
    out = torch.matmul(attn, v)
    return out, attn


# ---------------------------------------------------------------------------
# ProbeHooks
# ---------------------------------------------------------------------------

@dataclass
class _BlockBuffer:
    attn: list[torch.Tensor] = field(default_factory=list)         # per layer: [num_masked, H, S]
    v_norm: list[torch.Tensor] = field(default_factory=list)        # per layer: [H, S]
    h: list[torch.Tensor] = field(default_factory=list)             # per layer: [num_masked, d_model]
                                                                      # h[0] = embedding output, h[1..L] = block outputs


class ProbeHooks:
    """Install probe hooks for one (model, sample) recording session.

    Workflow:
        hooks = ProbeHooks(model, model_type="llada", n_layers=..., n_heads=...)
        for block_idx in range(num_blocks):
            hooks.set_block(block_idx, masked_positions_abs)
            hooks.armed = True
            model(x, ...)               # the step-0 forward of this block
            hooks.armed = False
            # ... rest of block's parallel-decoding steps run with armed=False ...
        data_per_block = hooks.collect()
        hooks.remove()
    """

    def __init__(
        self,
        model: nn.Module,
        model_type: str,
        n_layers: int,
        n_heads: int,
        d_head: int,
        record_v_norm: bool = True,
    ):
        self.model_type = model_type
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.record_v_norm = record_v_norm

        self.armed = False
        self.current_block: int | None = None
        self.masked_positions: list[int] | None = None
        self.expected_seq_len: int | None = None

        # Per-block accumulator
        self._buffers: dict[int, _BlockBuffer] = {}

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._patched_objs: list[tuple[Any, str, Any]] = []  # (obj, attr, original) for unpatching

        if model_type == "llada":
            self._install_llada(model)
        elif model_type == "dream":
            self._install_dream(model)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # ---- public API ----

    def set_block(self, block_idx: int, masked_positions: list[int]) -> None:
        self.current_block = block_idx
        self.masked_positions = list(masked_positions)
        self._buffers[block_idx] = _BlockBuffer()

    def collect(self) -> dict[int, dict[str, torch.Tensor]]:
        """Stack per-layer lists into [L, ...] tensors per block.

        Returns mapping: block_idx → {"attn": [num_masked, L, H, S], "v_norm": [L, H, S] or None,
                                       "h_masked": [L+1, num_masked, d_model]}
        """
        out: dict[int, dict[str, torch.Tensor]] = {}
        for block_idx, buf in self._buffers.items():
            # attn: each entry [num_masked, H, S]
            attn = torch.stack(buf.attn, dim=1) if buf.attn else None  # → [num_masked, L, H, S]
            v_norm = torch.stack(buf.v_norm, dim=0) if buf.v_norm else None  # → [L, H, S]
            h = torch.stack(buf.h, dim=0) if buf.h else None  # → [L+1, num_masked, d_model]
            out[block_idx] = {"attn": attn, "v_norm": v_norm, "h_masked": h}
        return out

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        for obj, attr, original in self._patched_objs:
            setattr(obj, attr, original)
        self._patched_objs.clear()

    # ---- LLaDA installation ----

    def _install_llada(self, model: nn.Module) -> None:
        blocks = _find_llada_blocks(model)
        if len(blocks) == 0:
            raise RuntimeError("No LLaDABlock instances found in model.")
        # Reconcile n_layers if the user passed a wrong number.
        self.n_layers = len(blocks)

        for layer_idx, block in enumerate(blocks):
            self._patch_llada_attention(block, layer_idx)
            self._handles.append(block.register_forward_hook(self._make_block_hook(layer_idx)))

        embed = _find_embedding(model)
        self._handles.append(embed.register_forward_hook(self._make_embed_hook()))

    def _patch_llada_attention(self, block: nn.Module, layer_idx: int) -> None:
        """Replace block._scaled_dot_product_attention with one that stashes attn + v_norm when armed."""
        original = block._scaled_dot_product_attention
        n_heads = self.n_heads
        d_head = self.d_head
        # W_O for this block: attn_out.weight is [d_model, n_heads * d_head] (out, in)
        # Reshape per-head for v_norm computation.
        W_O = block.attn_out.weight.detach()  # [d_model, n_heads * d_head]
        d_model = W_O.size(0)
        # [n_heads, d_head, d_model]: per head, the d_head→d_model projection.
        W_O_per_head = W_O.view(d_model, n_heads, d_head).permute(1, 2, 0).contiguous()

        hooks_self = self  # for closure

        def patched_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
            # Replicate LLaDA's GQA broadcast (already done inside its _scaled_dot_product_attention)
            num_q_heads = q.size(1)
            num_kv_heads = k.size(1)
            if num_q_heads != num_kv_heads:
                k_full = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
                v_full = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
            else:
                k_full = k
                v_full = v

            if hooks_self.armed and hooks_self.current_block is not None:
                # Manual softmax to expose attention weights and keep v
                out, attn = _manual_attention(q, k_full, v_full, attn_mask=attn_mask)
                # attn: [B=1, H, T_q, T_k] ; T_k is sequence length seen.
                B, H, T_q, T_k = attn.shape
                assert B == 1, "Probes are single-sample"
                masked_pos = hooks_self.masked_positions
                # attn at masked positions: [H, num_masked, T_k]
                attn_at_m = attn[0, :, masked_pos, :].to(torch.float16).cpu()
                # v_norm[h, j] = || W_O_h @ v_full[0, h, j, :] ||_2
                if hooks_self.record_v_norm:
                    # W_O_per_head: [H, d_head, d_model] ; v_full[0]: [H, T_k, d_head]
                    # einsum: per head h, per position j → vector w[h, j] = W_O_h @ v[0, h, j, :]
                    wov = torch.einsum("hkd,htk->htd", W_O_per_head.to(v_full.dtype).to(v_full.device), v_full[0])
                    v_norm = wov.norm(dim=-1).to(torch.float32).cpu()  # [H, T_k]
                else:
                    v_norm = None
                buf = hooks_self._buffers[hooks_self.current_block]
                buf.attn.append(attn_at_m)
                if v_norm is not None:
                    buf.v_norm.append(v_norm)
                return out
            else:
                # Disarmed: fall back to the original (uses flash/SDPA — fast).
                return original(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

        block._scaled_dot_product_attention = patched_sdpa
        self._patched_objs.append((block, "_scaled_dot_product_attention", original))

    # ---- Dream installation ----

    def _install_dream(self, model: nn.Module) -> None:
        layers = _find_dream_layers(model)
        if len(layers) == 0:
            raise RuntimeError("No DreamDecoderLayer instances found in model.")
        self.n_layers = len(layers)

        for layer_idx, layer in enumerate(layers):
            attn_module = layer.self_attn
            self._patch_dream_attention(attn_module, layer_idx)
            self._handles.append(layer.register_forward_hook(self._make_block_hook(layer_idx)))

        embed = _find_embedding(model)
        self._handles.append(embed.register_forward_hook(self._make_embed_hook()))

    def _patch_dream_attention(self, attn_module: nn.Module, layer_idx: int) -> None:
        """Wrap attn_module.forward to stash attn weights + v_norm when armed.

        We do this by replacing forward with a manual implementation only when armed; otherwise we
        delegate to the original.
        """
        original_forward = attn_module.forward
        n_heads = attn_module.num_heads
        n_kv_heads = attn_module.num_key_value_heads
        d_head = attn_module.head_dim
        # o_proj.weight: [hidden_size, n_heads * d_head]
        W_O = attn_module.o_proj.weight.detach()
        d_model = W_O.size(0)
        W_O_per_head = W_O.view(d_model, n_heads, d_head).permute(1, 2, 0).contiguous()  # [H, d_head, d_model]

        hooks_self = self

        def patched_forward(hidden_states, attention_mask=None, position_ids=None,
                            past_key_value=None, output_attentions=False, use_cache=False, **kwargs):
            if not hooks_self.armed or hooks_self.current_block is None:
                return original_forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    **kwargs,
                )

            # Manual path mirrors DreamSdpaAttention.forward but uses _manual_attention.
            bsz, q_len, _ = hidden_states.size()
            q = attn_module.q_proj(hidden_states).view(bsz, q_len, n_heads, d_head).transpose(1, 2)
            k = attn_module.k_proj(hidden_states).view(bsz, q_len, n_kv_heads, d_head).transpose(1, 2)
            v = attn_module.v_proj(hidden_states).view(bsz, q_len, n_kv_heads, d_head).transpose(1, 2)

            # Apply RoPE if the module has it
            cos, sin = attn_module.rotary_emb(v, position_ids) if hasattr(attn_module, "rotary_emb") else (None, None)
            if cos is not None:
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                q, k = apply_rotary_pos_emb(q, k, cos, sin)

            if past_key_value is not None:
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": kwargs.get("cache_position")}
                k, v = past_key_value.update(k, v, layer_idx, cache_kwargs)

            # GQA broadcast
            if n_heads != n_kv_heads:
                k_full = k.repeat_interleave(n_heads // n_kv_heads, dim=1, output_size=n_heads)
                v_full = v.repeat_interleave(n_heads // n_kv_heads, dim=1, output_size=n_heads)
            else:
                k_full = k
                v_full = v

            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask[:, :, :, : k_full.shape[-2]]

            out, attn_weights = _manual_attention(q, k_full, v_full, attn_mask=attn_mask)

            # Stash for the probe
            B, H, T_q, T_k = attn_weights.shape
            assert B == 1, "Probes are single-sample"
            masked_pos = hooks_self.masked_positions
            attn_at_m = attn_weights[0, :, masked_pos, :].to(torch.float16).cpu()
            if hooks_self.record_v_norm:
                wov = torch.einsum("hkd,htk->htd", W_O_per_head.to(v_full.dtype).to(v_full.device), v_full[0])
                v_norm = wov.norm(dim=-1).to(torch.float32).cpu()
            else:
                v_norm = None

            buf = hooks_self._buffers[hooks_self.current_block]
            buf.attn.append(attn_at_m)
            if v_norm is not None:
                buf.v_norm.append(v_norm)

            # Final output projection (matches DreamSdpaAttention.forward)
            attn_output = out.transpose(1, 2).contiguous().view(bsz, q_len, n_heads * d_head)
            attn_output = attn_module.o_proj(attn_output)

            present_kv = past_key_value if use_cache else None
            return attn_output, None, present_kv

        attn_module.forward = patched_forward
        self._patched_objs.append((attn_module, "forward", original_forward))

    # ---- shared block / embedding hooks ----

    def _make_block_hook(self, layer_idx: int):
        def hook(module, inputs, outputs):
            if not self.armed or self.current_block is None:
                return
            # Block forward returns (h, present_kv) for LLaDA; for Dream a longer tuple.
            if isinstance(outputs, tuple):
                h = outputs[0]
            else:
                h = outputs
            # h: [1, S, d_model]
            masked_pos = self.masked_positions
            h_at_m = h[0, masked_pos, :].to(torch.float16).cpu()
            buf = self._buffers[self.current_block]
            buf.h.append(h_at_m)
        return hook

    def _make_embed_hook(self):
        def hook(module, inputs, outputs):
            if not self.armed or self.current_block is None:
                return
            # outputs: [1, S, d_model]
            masked_pos = self.masked_positions
            h_at_m = outputs[0, masked_pos, :].to(torch.float16).cpu()
            buf = self._buffers[self.current_block]
            # Insert at index 0 so it represents ℓ=0 (input embedding)
            if len(buf.h) == 0 or len(buf.h) > 0 and buf.h[0].shape != h_at_m.shape:
                buf.h.insert(0, h_at_m)
            else:
                buf.h.insert(0, h_at_m)
        return hook
