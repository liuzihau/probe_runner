"""LLaDA-2.0-MoE / DMax-Math-16B generation with probe arming.

Replicates DMax's ``decode_uniform`` block-diffusion decode — NOT the Fast-dLLM
dual-cache loop the ``llada`` / ``dream`` runners use. We study the heavy
model's per-iteration refinement trajectory, so we mirror dInfer's
``generate_uniform`` / the canonical T3-D ``generate_t3d`` decode rule exactly,
with two deliberate simplifications for a probe:

  * **Full ``[0, block_end]`` forward every pass (no KV cache).** Simplest
    faithful decode; captures clean hidden at *all* positions each pass and
    keeps the probe-hook indices ABSOLUTE for every pass (the dream/llada
    runners switch to block-local indices at pass ≥ 1 because of the cache).
  * **Prompt-relative blocks** (``[Lp + b·B, Lp + (b+1)·B)``) instead of the
    production grid-aligned blocks — matches probe_runner's existing storage /
    plot conventions. The decode *rule* (threshold-0.3 left-to-right prefix
    commit, soft-embedding mix, block-causal mask) is identical, which is what
    the trajectory depends on.

Two decode modes (see T3-DMax handoff §5 / the design doc §1):

  * ``decode_mode="soft"`` (**DMax** ``decode_uniform``): committed positions
    are fed back as a soft embedding mix ``p·e(top1) + (1-p)·e(mask)`` and are
    **re-argmax'd every pass** → a commit can flip → the **CR** domain is live.
  * ``decode_mode="hard"`` (**LLaDA-2.0-mini** threshold decode): committed
    positions are hard token embeddings and **fixed** once committed → no CR.

Loads ``LLaDA2MoeModelLM`` from the T3-DMax checkout (resolved via
``configs.add_llada2_to_path``).
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F

from probe_runner import configs


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------

def load_llada2(
    model_path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    attn_implementation: str = "sdpa",
    t3dmax_root: str | Path | None = None,
    device: str = "cuda",
):
    """Load LLaDA2MoeModelLM (LLaDA-2.0-mini or DMax-Math-16B — same arch).

    `attn_implementation`: use "eager" to let the probe capture attention
    weights / v_norm (the info-flow plots); "sdpa" (default) is faster and
    fine for the hidden-state trajectory metrics, which need no attn weights.
    """
    configs.add_llada2_to_path(t3dmax_root)
    from dinfer.model import LLaDA2MoeModelLM  # type: ignore  # noqa: WPS433
    from transformers import AutoConfig, AutoTokenizer  # noqa: WPS433

    model_path = os.path.abspath(str(model_path))
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cfg._attn_implementation = attn_implementation
    model = LLaDA2MoeModelLM(config=cfg).eval()
    # Custom sharded-safetensors loader (NOT from_pretrained); see digest §8.
    model.load_weights(model_path, torch_dtype=dtype, device=device)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def _embedding_layer(model):
    """Input token-embedding module (LLaDA-2.0 names it word_embeddings)."""
    try:
        emb = model.get_input_embeddings()
        if emb is not None:
            return emb
    except Exception:
        pass
    base = getattr(model, "model", model)
    return getattr(base, "word_embeddings", None) or base.get_input_embeddings()


# ----------------------------------------------------------------------
# decode_uniform primitives (ported from dInfer parallel_strategy.py)
# ----------------------------------------------------------------------

def _build_block_causal_mask(seq_len: int, prompt_len: int, block_length: int,
                             dtype: torch.dtype, device) -> torch.Tensor:
    """Additive block-causal mask [1, 1, L, L].

    The prompt [0, prompt_len) is block 0 (fully visible to everyone, and
    bidirectional within itself — LLaDA is a masked-diffusion model, not
    autoregressive). Decode positions form blocks 1, 2, … A query in block b
    attends to all keys in blocks ≤ b (full bidirectional within a block,
    causal across blocks). Blocked entries get finfo.min so they vanish after
    softmax (the model adds this mask to QKᵀ/√d).
    """
    idx = torch.arange(seq_len, device=device)
    blk = torch.where(idx < prompt_len,
                      torch.zeros_like(idx),
                      1 + (idx - prompt_len) // block_length)            # [L]
    allowed = blk.unsqueeze(0) <= blk.unsqueeze(1)                       # [L, L] (key, query) -> [q, k]
    mask = torch.zeros(seq_len, seq_len, dtype=dtype, device=device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.unsqueeze(0).unsqueeze(0)                                # [1, 1, L, L]


def _commit_uniform(block_logits: torch.Tensor, block_x: torch.Tensor,
                    mask_id: int, threshold: float):
    """DMax left-to-right contiguous-prefix commit (get_transfer_index_uniform).

    Args:
        block_logits: [1, B, V] logits for the current block.
        block_x:      [1, B] current token ids (mask_id where masked).
    Returns:
        x0:             [1, B] argmax token ids (at every position).
        high_conf_index:[1, B] bool — the committable prefix among masked.
        max_probs:      [1, B] argmax probability (confidence).
    """
    probs = torch.softmax(block_logits.float(), dim=-1)
    max_probs, x0 = probs.max(dim=-1)                                   # [1, B]
    mask_index = (block_x == mask_id)
    neg_inf = torch.full_like(max_probs, float("-inf"))
    confidence = torch.where(mask_index, max_probs, neg_inf)
    # Left-to-right: after the first low-confidence masked position, fail the rest.
    is_low = mask_index & (confidence < threshold)
    has_failure = torch.cumsum(is_low.long(), dim=1) > 0
    candidates = mask_index & (~has_failure)
    has_selection = candidates.any(dim=-1, keepdim=True)
    # Fallback: if nothing qualifies, commit the first (leftmost) masked token.
    mask_cumsum = torch.cumsum(mask_index.long(), dim=1)
    first_mask = (mask_cumsum == 1) & mask_index
    high_conf_index = torch.where(has_selection, candidates, first_mask)
    return x0, high_conf_index, max_probs


def _build_block_soft_embeds(block_ids: torch.Tensor, max_probs: torch.Tensor,
                             x0: torch.Tensor, soft_cond: torch.Tensor,
                             embedding_layer, mask_id: int) -> torch.Tensor:
    """Soft-embedding mix for committed positions (top_k=1, the standard).

    For each committed position: ``p·e(argmax) + (1-p)·e(mask)`` then
    L2-renormalized to the probability-weighted expected norm (matches
    dInfer's ThresholdParallelDecoder.decode_uniform). Masked positions keep
    their hard mask embedding.

    Args:
        block_ids: [1, B] post-commit token ids (mask_id where masked).
        max_probs/x0: [1, B] from `_commit_uniform`.
        soft_cond: [B] bool — committed positions to soften.
    Returns [1, B, D].
    """
    base = embedding_layer(block_ids)                                   # [1, B, D]
    if not bool(soft_cond.any()):
        return base
    device = base.device
    topk_probs = max_probs.unsqueeze(-1)                                # [1, B, 1]
    topk_idx = x0.unsqueeze(-1)                                         # [1, B, 1]
    residual = (1.0 - topk_probs.sum(-1, keepdim=True)).clamp_min(0.0)  # [1, B, 1]
    topk_embeds = embedding_layer(topk_idx)                            # [1, B, 1, D]
    mask_embed = embedding_layer(
        torch.tensor([[mask_id]], device=device, dtype=torch.long))     # [1, 1, D]
    mask_norm = mask_embed.norm(dim=-1, keepdim=True)                   # [1, 1, 1]

    topk_weighted = (topk_embeds * topk_probs.unsqueeze(-1)).sum(dim=2)  # [1, B, D]
    mask_weighted = mask_embed * residual                              # [1, B, D]
    soft = topk_weighted + mask_weighted
    cur_norm = soft.norm(dim=-1, keepdim=True)                         # [1, B, 1]
    topk_norms = topk_embeds.norm(dim=-1)                              # [1, B, 1]
    expected_topk_norm = (topk_norms * topk_probs).sum(-1, keepdim=True)
    expected_mask_norm = mask_norm * residual
    target_norm = expected_topk_norm + expected_mask_norm
    soft = soft * (target_norm / (cur_norm + 1e-6))
    soft = soft.to(base.dtype)

    out = base.clone()
    out[:, soft_cond, :] = soft[:, soft_cond, :]
    return out


# ----------------------------------------------------------------------
# Generation loop with probe callbacks
# ----------------------------------------------------------------------

@torch.no_grad()
def generate_with_probes(
    model,
    prompt: torch.Tensor,
    *,
    on_block_start,
    on_block_end,
    on_pass_start=None,
    on_pass_end=None,
    intra_block: bool = False,
    mask_id: int,
    eos_id: int | None = None,
    gen_length: int = 256,
    block_length: int = 32,
    decode_mode: str = "soft",
    commit_threshold: float = 0.3,
    break_threshold: float = 0.9,
    max_iter_per_block: int | None = None,
    # Accept (and ignore) the Fast-dLLM-style kwargs run_probes passes uniformly
    # to all runners (steps / threshold / temperature / top_p / top_k).
    **_ignored,
):
    """Run LLaDA-2.0 / DMax decode_uniform with probe callbacks.

    Block-level callbacks (always):
        on_block_start(block_idx, masked_positions_abs)
        on_block_end(block_idx)

    Per-pass callbacks (intra_block=True; fired around every forward):
        on_pass_start(block_idx, pass_idx, token_state_pre, indices_in_forward,
                      committed_tokens_pre)
            token_state_pre: int8 [block_length] (0=mask, 1=committed) at pass start.
            indices_in_forward: absolute positions s..e (full-seq forward every pass).
            committed_tokens_pre: long [block_length] — the block's token ids fed
                INTO this pass (mask_id where masked). Stored as
                committed_tokens_per_pass → enables the CR domain + converged-token
                + rank trajectory.
        on_pass_end(block_idx, pass_idx, revealed_this_pass)
            revealed_this_pass: bool [block_length] — positions mask→committed this pass.

    Returns (x, nfe).
    """
    assert decode_mode in ("soft", "hard")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if prompt.device != device:
        prompt = prompt.to(device)
    Lp = int(prompt.shape[1])
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    if max_iter_per_block is None:
        max_iter_per_block = block_length

    embedding_layer = _embedding_layer(model)
    x = F.pad(prompt, (0, gen_length), value=mask_id)                  # [1, Lp+gen_length]
    nfe = 0

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length
        block_end = e
        on_block_start(nb, list(range(s, e)))

        attn_mask = _build_block_causal_mask(block_end, Lp, block_length, dtype, device)
        pos_ids = torch.arange(block_end, device=device).unsqueeze(0)
        block_soft = None  # [1, B, D] soft embeds for the current block (None until first commit)

        for i in range(max_iter_per_block):
            # Probe arming (pre-pass state).
            pre_state = (x[0, s:e] != mask_id).to(torch.int8)
            pre_tokens = x[0, s:e].clone().to(torch.long)
            if intra_block and on_pass_start is not None:
                on_pass_start(nb, i, pre_state, list(range(s, e)), pre_tokens)

            # Build inputs_embeds over [0, block_end]: hard for prompt + earlier
            # blocks, soft (or hard mask) for the current block.
            inputs_embeds = embedding_layer(x[:, :block_end])
            if block_soft is not None:
                inputs_embeds[:, s:e, :] = block_soft

            out = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attn_mask,
                position_ids=pos_ids,
                use_cache=False,
                output_hidden_states=False,
                output_router_logits=False,
                return_dict=True,
            )
            nfe += 1
            block_logits = out.logits[:, s:e, :]                       # [1, B, V]

            block_x = x[:, s:e]
            mask_index = (block_x == mask_id)
            x0, high_conf_index, max_probs = _commit_uniform(
                block_logits, block_x, mask_id, commit_threshold)

            # Update rule. soft (DMax): also re-argmax already-committed
            # positions (active & ~mask) → commits can be revised (CR).
            active_committed = (~mask_index)
            if decode_mode == "soft":
                update_mask = high_conf_index | active_committed
            else:
                update_mask = high_conf_index
            changed = update_mask & (x0 != block_x)
            new_block = block_x.clone()
            new_block[update_mask] = x0[update_mask]
            x[:, s:e] = new_block

            revealed_blk = (mask_index & high_conf_index)[0].to(torch.bool)  # mask→committed this pass
            if intra_block and on_pass_end is not None:
                on_pass_end(nb, i, revealed_blk)
            elif not intra_block and i == 0:
                # Legacy contract: capture pass 0 only, then disarm.
                on_block_end(nb)

            # Early stop (decode_uniform Breakflag): all positions confident, or
            # nothing changed this pass.
            conf_ok = bool((max_probs >= break_threshold).all())
            no_change = not bool(changed.any())
            if conf_ok or no_change:
                break

            # Build next pass's soft embeds (soft mode only).
            if decode_mode == "soft":
                soft_cond = (x[0, s:e] != mask_id)                     # committed after this pass
                block_soft = _build_block_soft_embeds(
                    x[:, s:e], max_probs, x0, soft_cond, embedding_layer, mask_id)

        if intra_block:
            on_block_end(nb)

    return x, nfe
