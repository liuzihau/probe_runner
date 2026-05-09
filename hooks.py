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
    k_prompt: list[torch.Tensor] = field(default_factory=list)      # per layer: [n_kv_heads, prompt_len, d_head]
    v_prompt: list[torch.Tensor] = field(default_factory=list)      # per layer: [n_kv_heads, prompt_len, d_head]


@dataclass
class _IntraPassBuffer:
    """Per-(block, pass) recording for intra-block drift analysis.

    `h` accumulates one tensor per layer (incl. embed at index 0), each of shape
    [num_masked, d_model]. `token_state` is a [num_masked] int8 tensor saying
    whether each position is mask (0) or clean (1) at the *input* of this pass.
    `revealed_this_pass` is [num_masked] bool: was this pass the one that
    flipped the position from mask→clean? Filled in at pass-end.
    """
    h: list[torch.Tensor] = field(default_factory=list)
    token_state: torch.Tensor | None = None
    revealed_this_pass: torch.Tensor | None = None


class ProbeHooks:
    """Install probe hooks for one (model, sample) recording session.

    Two modes:

    Legacy (default, `intra_block=False`):
        hooks = ProbeHooks(model, model_type="llada", n_layers=..., n_heads=...)
        for block_idx in range(num_blocks):
            hooks.set_block(block_idx, masked_positions_abs)
            hooks.armed = True
            model(x, ...)               # the step-0 forward of this block
            hooks.armed = False
            # ... rest of block's parallel-decoding steps run with armed=False ...
        data_per_block = hooks.collect()
        hooks.remove()

    Intra-block (`intra_block=True`): hooks stay armed across every pass within
    a block. Per-pass `h` is captured; `attn` and `v_norm` are still pass-0 only
    (we already have those, and recording attn at every pass would force manual
    softmax across the whole inner loop and ~2× the wall time).

        hooks = ProbeHooks(..., intra_block=True)
        for block_idx in range(num_blocks):
            hooks.set_block(block_idx, masked_positions_abs)
            for pass_idx in range(steps_per_block):
                hooks.set_pass(block_idx, pass_idx, token_state=...)
                hooks.armed = True
                model(...)
                hooks.armed = False
                hooks.finalize_pass(revealed_this_pass=...)
        data_per_block, intra_per_block = hooks.collect()
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
        intra_block: bool = False,
        record_prompt_kv: bool = False,
    ):
        self.model_type = model_type
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.record_v_norm = record_v_norm
        self.intra_block = bool(intra_block)
        # When True, capture per-block per-layer post-RoPE K/V at prompt
        # positions during pass 0 of each block — for cross-block prefix
        # KV-drift analysis (does the cached prompt KV that pass-0 freshly
        # recomputes stay close from block to block?).
        self.record_prompt_kv = bool(record_prompt_kv)
        # Absolute positions to slice K/V at when record_prompt_kv is on.
        # Set once per sample via set_prompt_indices.
        self.prompt_indices: list[int] | None = None

        self.armed = False
        self.current_block: int | None = None
        self.current_pass: int | None = None
        self.masked_positions: list[int] | None = None  # absolute positions; legacy
        # `current_indices` is what hooks actually use to slice `outputs`. At
        # pass 0 it equals `masked_positions` (the model sees the full
        # sequence). At pass ≥ 1 the runner calls model(x[:, s:e]) so it sets
        # current_indices to the *local* range list(range(block_length)).
        self.current_indices: list[int] | None = None
        self.expected_seq_len: int | None = None

        # Per-block accumulator (legacy: pass-0 attn + v_norm + h_masked)
        self._buffers: dict[int, _BlockBuffer] = {}
        # Per-(block, pass) accumulator (intra_block mode only)
        self._intra_buffers: dict[int, dict[int, _IntraPassBuffer]] = {}

        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._patched_objs: list[tuple[Any, str, Any]] = []  # (obj, attr, original) for unpatching

        if model_type == "llada":
            self._install_llada(model)
        elif model_type == "dream":
            self._install_dream(model)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # ---- public API ----

    def set_prompt_indices(self, prompt_indices: list[int]) -> None:
        """Set the absolute prompt positions to capture K/V at (pass 0 only).

        Must be called once per sample before the first forward when
        `record_prompt_kv=True`. The same indices are reused for every block.
        """
        self.prompt_indices = list(prompt_indices)

    def set_block(self, block_idx: int, masked_positions: list[int]) -> None:
        self.current_block = block_idx
        self.current_pass = None
        self.masked_positions = list(masked_positions)
        self.current_indices = list(masked_positions)  # default: pass 0 uses absolute
        self._buffers[block_idx] = _BlockBuffer()
        if self.intra_block:
            self._intra_buffers[block_idx] = {}

    def set_pass(
        self,
        block_idx: int,
        pass_idx: int,
        *,
        token_state: torch.Tensor | None = None,
        indices: list[int] | None = None,
    ) -> None:
        """Tag the next forward as belonging to (block_idx, pass_idx).

        `token_state` is a [num_masked] int8/bool tensor of pre-pass token
        states. Stored on the per-pass buffer so analyses can group positions
        by mask/clean transitions across consecutive passes.

        `indices` overrides which positions of `outputs` the hooks slice for
        this pass. Required at pass ≥ 1 because the model is called with
        `x[:, s:e]` and the hidden-state tensors are *block-local* (length
        `block_length`), not full-sequence — so absolute positions would
        index out of bounds. At pass 0 the model gets the full sequence and
        the absolute `masked_positions` set by `set_block` works.
        """
        self.current_block = block_idx
        self.current_pass = pass_idx
        if indices is not None:
            self.current_indices = list(indices)
        if self.intra_block:
            buf = _IntraPassBuffer()
            if token_state is not None:
                buf.token_state = token_state.detach().to(torch.int8).cpu()
            self._intra_buffers.setdefault(block_idx, {})[pass_idx] = buf

    def finalize_pass(self, revealed_this_pass: torch.Tensor | None = None) -> None:
        """Attach the post-pass reveal mask to the current pass buffer."""
        if not self.intra_block or self.current_block is None or self.current_pass is None:
            return
        buf = self._intra_buffers[self.current_block].get(self.current_pass)
        if buf is not None and revealed_this_pass is not None:
            buf.revealed_this_pass = revealed_this_pass.detach().to(torch.bool).cpu()

    def collect(self) -> dict[int, dict[str, torch.Tensor]]:
        """Stack per-layer lists into [L, ...] tensors per block.

        Returns mapping: block_idx → {"attn": [num_masked, L, H, S], "v_norm": [L, H, S] or None,
                                       "h_masked": [L+1, num_masked, d_model],
                                       "k_prompt": [L, n_kv_heads, prompt_len, d_head] or None,
                                       "v_prompt": [L, n_kv_heads, prompt_len, d_head] or None}
        """
        out: dict[int, dict[str, torch.Tensor]] = {}
        for block_idx, buf in self._buffers.items():
            # attn: each entry [num_masked, H, S]
            attn = torch.stack(buf.attn, dim=1) if buf.attn else None  # → [num_masked, L, H, S]
            v_norm = torch.stack(buf.v_norm, dim=0) if buf.v_norm else None  # → [L, H, S]
            h = torch.stack(buf.h, dim=0) if buf.h else None  # → [L+1, num_masked, d_model]
            k_prompt = torch.stack(buf.k_prompt, dim=0) if buf.k_prompt else None  # → [L, n_kv_heads, prompt_len, d_head]
            v_prompt = torch.stack(buf.v_prompt, dim=0) if buf.v_prompt else None
            out[block_idx] = {
                "attn": attn, "v_norm": v_norm, "h_masked": h,
                "k_prompt": k_prompt, "v_prompt": v_prompt,
            }
        return out

    def collect_intra_block(self) -> dict[int, dict[str, torch.Tensor]]:
        """Stack per-pass `h` lists into per-block tensors.

        Returns mapping: block_idx → {
            "h_per_pass":            [n_passes, L+1, num_masked, d_model] float16,
            "token_state_per_pass":  [n_passes, num_masked] int8,
            "revealed_per_pass":     [n_passes, num_masked] bool,
        }
        Empty dict when `intra_block=False`. Blocks where all passes ran
        get a uniform first dimension; ragged-by-block (different blocks
        can have different `n_passes` due to early-exit when block fully
        decoded).
        """
        out: dict[int, dict[str, torch.Tensor]] = {}
        if not self.intra_block:
            return out
        for block_idx, pass_map in self._intra_buffers.items():
            if not pass_map:
                continue
            sorted_passes = sorted(pass_map.keys())
            # Drop trailing passes that have no h recorded (e.g. armed=False
            # because the runner detected early-exit before firing the model).
            sorted_passes = [p for p in sorted_passes if pass_map[p].h]
            if not sorted_passes:
                continue
            h_stacks = []
            ts_list, rv_list = [], []
            for p in sorted_passes:
                buf = pass_map[p]
                h_stacks.append(torch.stack(buf.h, dim=0))  # [L+1, num_masked, d_model]
                ts_list.append(buf.token_state if buf.token_state is not None
                               else torch.zeros(len(self.masked_positions or []), dtype=torch.int8))
                rv_list.append(buf.revealed_this_pass if buf.revealed_this_pass is not None
                               else torch.zeros(len(self.masked_positions or []), dtype=torch.bool))
            out[block_idx] = {
                "h_per_pass":           torch.stack(h_stacks, dim=0),
                "token_state_per_pass": torch.stack(ts_list, dim=0),
                "revealed_per_pass":    torch.stack(rv_list, dim=0),
                "pass_indices":         torch.tensor(sorted_passes, dtype=torch.int32),
            }
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

            # Skip manual-softmax capture on intra-block passes ≥ 1: we
            # already have attn/v_norm at pass 0, and forcing manual softmax
            # for the rest of the block ~doubles inference time.
            capture_attn = (
                hooks_self.armed
                and hooks_self.current_block is not None
                and (hooks_self.current_pass is None or hooks_self.current_pass == 0)
            )
            if capture_attn:
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
                # Capture pre-GQA-broadcast post-RoPE K/V at prompt positions.
                # k/v shapes here are [B=1, n_kv_heads, T_k, d_head] — what
                # actually lands in the cache at pass 0 of every block.
                if hooks_self.record_prompt_kv and hooks_self.prompt_indices is not None:
                    p_idx = hooks_self.prompt_indices
                    k_at_prompt = k[0, :, p_idx, :].to(torch.float16).cpu()  # [n_kv_heads, prompt_len, d_head]
                    v_at_prompt = v[0, :, p_idx, :].to(torch.float16).cpu()
                    buf.k_prompt.append(k_at_prompt)
                    buf.v_prompt.append(v_at_prompt)
                return out
            else:
                # Disarmed OR intra-block pass ≥ 1: fall back to the
                # original (uses flash/SDPA — fast). The block-output
                # forward hook still records `h` per pass.
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

        def patched_forward(
            hidden_states, attention_mask=None, position_ids=None,
            past_key_value=None, output_attentions=False, use_cache=False,
            cache_position=None, position_embeddings=None,
            dual_cache=False, replace_position=None, **kwargs,
        ):
            # Skip manual-softmax capture on intra-block passes ≥ 1 for the
            # same reason as the LLaDA path (avoid 2× inference cost).
            capture_attn = (
                hooks_self.armed
                and hooks_self.current_block is not None
                and (hooks_self.current_pass is None or hooks_self.current_pass == 0)
            )
            if not capture_attn:
                return original_forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    dual_cache=dual_cache,
                    replace_position=replace_position,
                    **kwargs,
                )

            # Manual path mirrors DreamSdpaAttention.forward exactly.
            # Crucially: Dream's cache is List[Tuple[K, V]] with PRE-RoPE
            # K/V; cache update happens BEFORE the reshape + RoPE; and
            # `dual_cache=True` does in-place replacement at
            # `replace_position` indices instead of concatenating.
            bsz, q_len, _ = hidden_states.size()
            query_states = attn_module.q_proj(hidden_states)
            key_states   = attn_module.k_proj(hidden_states)
            value_states = attn_module.v_proj(hidden_states)

            # ---- Dream cache update (pre-RoPE, [B, T, hidden] shape) ----
            if past_key_value is not None:
                if dual_cache:
                    past_key, past_value = past_key_value
                    replace_idx = replace_position.nonzero(as_tuple=True)[1]
                    past_key[:, replace_idx] = key_states
                    key_states = past_key
                    past_value[:, replace_idx] = value_states
                    value_states = past_value
                else:
                    past_key, past_value = past_key_value
                    key_states   = torch.cat([past_key,   key_states],   dim=-2)
                    value_states = torch.cat([past_value, value_states], dim=-2)

            new_past_key_value = (key_states, value_states) if use_cache else None

            # ---- Reshape for attention ----
            query_states = query_states.view(bsz, q_len, n_heads, d_head).transpose(1, 2)
            key_states   = key_states.view(  bsz, -1,  n_kv_heads, d_head).transpose(1, 2)
            value_states = value_states.view(bsz, -1,  n_kv_heads, d_head).transpose(1, 2)

            # ---- RoPE (apply to full key sequence, including cached) ----
            if position_embeddings is None and hasattr(attn_module, "rotary_emb"):
                cos, sin = attn_module.rotary_emb(value_states, position_ids)
            elif position_embeddings is not None:
                cos, sin = position_embeddings
            else:
                cos, sin = None, None
            if cos is not None:
                from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
                query_states, key_states = apply_rotary_pos_emb(
                    query_states, key_states, cos, sin,
                )

            # Capture pre-GQA-broadcast post-RoPE K/V at prompt positions
            # before broadcasting. key_states / value_states shapes here are
            # [B=1, n_kv_heads, T_k, d_head]. Pass-0 only (capture_attn gate
            # above ensures this), so T_k = full sequence length.
            if hooks_self.record_prompt_kv and hooks_self.prompt_indices is not None:
                p_idx = hooks_self.prompt_indices
                k_at_prompt = key_states[0, :, p_idx, :].to(torch.float16).cpu()
                v_at_prompt = value_states[0, :, p_idx, :].to(torch.float16).cpu()
                buf_kv = hooks_self._buffers[hooks_self.current_block]
                buf_kv.k_prompt.append(k_at_prompt)
                buf_kv.v_prompt.append(v_at_prompt)

            # ---- GQA broadcast for the attention math + capture ----
            if n_heads != n_kv_heads:
                k_full = key_states.repeat_interleave(
                    n_heads // n_kv_heads, dim=1, output_size=n_heads,
                )
                v_full = value_states.repeat_interleave(
                    n_heads // n_kv_heads, dim=1, output_size=n_heads,
                )
            else:
                k_full = key_states
                v_full = value_states

            # Dream's runner passes attention_mask="full" (literal string)
            # for prompts without padding; upstream Dream attention keys off
            # `isinstance(attention_mask, torch.Tensor)`. Mirror that —
            # non-tensor → no mask (full attention).
            attn_mask = None
            if isinstance(attention_mask, torch.Tensor):
                attn_mask = attention_mask[:, :, :, : k_full.shape[-2]]

            out, attn_weights = _manual_attention(
                query_states, k_full, v_full, attn_mask=attn_mask,
            )

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

            return attn_output, None, new_past_key_value

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
            # h: [1, S, d_model] at pass 0; [1, block_length, d_model] at pass ≥ 1
            indices = self.current_indices if self.current_indices is not None else self.masked_positions
            h_at_m = h[0, indices, :].to(torch.float16).cpu()

            # Legacy buffer: only at pass 0 (or when set_pass was never called).
            if self.current_pass is None or self.current_pass == 0:
                self._buffers[self.current_block].h.append(h_at_m)

            # Intra-block buffer: every pass.
            if self.intra_block and self.current_pass is not None:
                pass_buf = self._intra_buffers[self.current_block].get(self.current_pass)
                if pass_buf is not None:
                    pass_buf.h.append(h_at_m)
        return hook

    def _make_embed_hook(self):
        def hook(module, inputs, outputs):
            if not self.armed or self.current_block is None:
                return
            # outputs: [1, S, d_model] at pass 0; [1, block_length, d_model] at pass ≥ 1
            indices = self.current_indices if self.current_indices is not None else self.masked_positions
            h_at_m = outputs[0, indices, :].to(torch.float16).cpu()

            # Legacy buffer: only at pass 0.
            if self.current_pass is None or self.current_pass == 0:
                buf = self._buffers[self.current_block]
                # Insert at index 0 so it represents ℓ=0 (input embedding)
                buf.h.insert(0, h_at_m)

            # Intra-block buffer: prepend embed for every pass.
            if self.intra_block and self.current_pass is not None:
                pass_buf = self._intra_buffers[self.current_block].get(self.current_pass)
                if pass_buf is not None:
                    pass_buf.h.insert(0, h_at_m)
        return hook
