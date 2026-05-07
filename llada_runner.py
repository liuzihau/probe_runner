"""LLaDA generation with probe arming, mirroring Fast-dLLM v1's `generate_with_dual_cache`.

We copy the dual-cache loop from `related_repos/Fast-dLLM/v1/llada/generate.py` and inject
arm/disarm calls around the **first forward of each block** — the moment the block's 32 positions
are still all masked. The 7 (parallel) or 31 (sequential) follow-up steps run with hooks disarmed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np

from probe_runner.configs import resolve_fast_dllm_path


def _add_fast_dllm_to_path(fast_dllm_path: str | Path | None = None) -> None:
    """Add Fast-dLLM v1 LLaDA module dir to sys.path so its `model.modeling_llada` resolves."""
    root = resolve_fast_dllm_path(fast_dllm_path)
    fast_dllm_llada = root / "llada"
    if str(fast_dllm_llada) not in sys.path:
        sys.path.insert(0, str(fast_dllm_llada))


def load_llada(
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
    dtype: torch.dtype = torch.bfloat16,
    fast_dllm_path: str | Path | None = None,
):
    """Load LLaDA via Fast-dLLM v1's LLaDAModelLM class."""
    _add_fast_dllm_to_path(fast_dllm_path)
    from model.modeling_llada import LLaDAModelLM  # noqa: WPS433
    from transformers import AutoTokenizer  # noqa: WPS433

    model = LLaDAModelLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


# ----------------------------------------------------------------------
# Generation utilities (verbatim from Fast-dLLM v1, kept here so we can
# inject arm/disarm callbacks around the first forward of each block)
# ----------------------------------------------------------------------

def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    device = block_mask_index.device
    dtype = torch.long
    total = block_mask_index.sum(dim=1)
    base = torch.div(total, steps, rounding_mode="floor")
    rem = total - base * steps
    num = base.unsqueeze(1).expand(-1, steps).to(dtype)
    cols = torch.arange(steps, device=device).unsqueeze(0)
    add_mask = cols < rem.unsqueeze(1)
    return num + add_mask.to(dtype)


def _get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)
    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)
    elif remasking == "random":
        x0_p = torch.rand(x0.shape, device=x0.device, dtype=torch.float64)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    neg_inf = torch.tensor(torch.finfo(x0_p.dtype).min, device=x0_p.device, dtype=x0_p.dtype)
    confidence = torch.where(mask_index, x0_p, neg_inf)

    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)
        max_conf_indices = torch.argmax(confidence, dim=1, keepdim=True)
        force_mask = torch.zeros_like(transfer_index).scatter_(1, max_conf_indices, True)
        transfer_index = (transfer_index | force_mask) & mask_index
        return x0, transfer_index

    if num_transfer_tokens.dim() == 2 and num_transfer_tokens.size(1) == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    num_transfer_tokens = num_transfer_tokens.to(dtype=torch.long, device=confidence.device).clamp(min=0)
    _, idx = torch.sort(confidence, dim=1, descending=True)
    B, L = confidence.shape
    cols = torch.arange(L, device=confidence.device).unsqueeze(0).expand(B, L)
    k_expanded = num_transfer_tokens.unsqueeze(1).expand(B, L)
    select_sorted = cols < k_expanded
    transfer_int = torch.zeros(B, L, device=confidence.device, dtype=torch.int8)
    transfer_int = transfer_int.scatter(1, idx, select_sorted.to(torch.int8))
    transfer_index = transfer_int.bool() & mask_index
    return x0, transfer_index


# ----------------------------------------------------------------------
# Generation loop with probe arming
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
    steps: int = 256,
    gen_length: int = 256,
    block_length: int = 32,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: float | None = 0.9,
):
    """Mirror of `generate_with_dual_cache` from Fast-dLLM v1, with probe callbacks.

    Block-level callbacks (always called):
        on_block_start(block_idx, masked_positions_abs)  — before the block's first forward.
        on_block_end(block_idx)                          — after the block's last forward.

    Per-pass callbacks (intra_block=True only — fired around every forward):
        on_pass_start(block_idx, pass_idx, token_state_pre, indices_in_forward)
            token_state_pre: int8 [block_length] — 0=mask, 1=clean at pass start.
            indices_in_forward: list[int] — positions of the forward's hidden-
                state tensor to record at. Absolute (s..e) at pass 0 (full
                sequence), local (0..block_length) at pass ≥ 1 (block-only).
        on_pass_end(block_idx, pass_idx, revealed_this_pass)
            revealed_this_pass: bool [block_length] — positions revealed by this pass.

    In legacy mode (`intra_block=False`) on_block_end fires immediately after
    the block's first forward (matching the old contract). In intra-block
    mode it fires *after the last pass of the block* — same total firings,
    but now the inner-loop forwards are also armed.

    Returns (x, nfe).
    """
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    x = torch.full((B, Lp + gen_length), mask_id, dtype=torch.long, device=model.device)
    x[:, :Lp] = prompt

    nfe = 0

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length

        block_mask_index = (x[:, s:e] == mask_id)
        num_transfer_tokens = _get_num_transfer_tokens(block_mask_index, steps_per_block)

        masked_positions_abs = list(range(s, e))
        on_block_start(nb, masked_positions_abs)

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True
        past_key_values = None

        for i in range(steps_per_block):
            # Block fully decoded — early exit. Also true at the top of the
            # very first iter of pass 0 only when block_length == 0.
            if i > 0 and (x[:, s:e] == mask_id).sum() == 0:
                break

            # Per-pass start callback (intra-block mode only).
            if intra_block and on_pass_start is not None:
                pre_state = (x[0, s:e] != mask_id).to(torch.int8)
                # Pass 0: model sees full x → absolute positions [s, e).
                # Pass ≥ 1: model sees x[:, s:e] → local positions [0, B).
                indices_in_forward = (
                    list(range(s, e)) if i == 0 else list(range(block_length))
                )
                on_pass_start(nb, i, pre_state, indices_in_forward)

            # In legacy mode, only pass 0 is armed; the runner relies on
            # on_block_start having armed and on_block_end about to disarm.
            # In intra-block mode every pass is armed and the (un)arming is
            # done by the per-pass callbacks instead.

            if i == 0:
                out_full = model(x, use_cache=True)
                past_key_values = out_full.past_key_values
                if not intra_block:
                    # Legacy contract: disarm right after the first forward.
                    on_block_end(nb)

                global_mask_index = (x == mask_id)
                global_mask_index[:, e:] = False
                quota_i = None if threshold is not None else num_transfer_tokens[:, 0]
                x0, transfer_index = _get_transfer_index(
                    out_full.logits, temperature, remasking, global_mask_index, x, quota_i, threshold,
                )
                new_x = torch.where(transfer_index, x0, x)
                revealed_blk = transfer_index[0, s:e].to(torch.bool)
                x = new_x
            else:
                logits_blk = model(
                    x[:, s:e],
                    past_key_values=past_key_values,
                    use_cache=True,
                    replace_position=replace_position,
                ).logits
                mask_blk = (x[:, s:e] == mask_id)
                quota_i = None if threshold is not None else num_transfer_tokens[:, i]
                x0_blk, transfer_idx_blk = _get_transfer_index(
                    logits_blk, temperature, remasking, mask_blk, x[:, s:e], quota_i, threshold,
                )
                blk_old = x[:, s:e]
                blk_new = torch.where(transfer_idx_blk, x0_blk, blk_old)
                x = torch.cat([x[:, :s], blk_new, x[:, e:]], dim=1)
                revealed_blk = transfer_idx_blk[0].to(torch.bool)

            nfe += 1

            if intra_block and on_pass_end is not None:
                on_pass_end(nb, i, revealed_blk)

        # In intra-block mode the block-end callback fires after the inner
        # loop finishes (or breaks out early). In legacy mode it already
        # fired right after pass 0.
        if intra_block:
            on_block_end(nb)

    return x, nfe
