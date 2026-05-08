"""Dream generation with probe arming.

Mirrors Fast-dLLM v1 Dream's `_sample` (in `related_repos/Fast-dLLM/v1/dream/model/generation_utils_block.py`),
restricted to the dual_cache + confidence_threshold path (matches LLaDA's dual-cache + parallel decoding).

We replicate the loop here so we can inject arm/disarm callbacks around the first forward of each block
(line ~451 in the upstream `_sample`).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from probe_runner.configs import resolve_fast_dllm_path


def _add_fast_dllm_to_path(fast_dllm_path: str | Path | None = None) -> None:
    root = resolve_fast_dllm_path(fast_dllm_path)
    fast_dllm_dream = root / "dream"
    if str(fast_dllm_dream) not in sys.path:
        sys.path.insert(0, str(fast_dllm_dream))


def load_dream(
    model_name: str = "Dream-org/Dream-v0-Instruct-7B",
    dtype: torch.dtype = torch.bfloat16,
    fast_dllm_path: str | Path | None = None,
):
    """Load Dream via the Fast-dLLM v1 wrapper (which adds the block-aware generation utils)."""
    _add_fast_dllm_to_path(fast_dllm_path)
    from transformers import AutoTokenizer  # noqa: WPS433
    # Fast-dLLM v1 ships a custom DreamModel + tokenization in its `model/` folder; importing
    # it via trust_remote_code on the HF id pulls the upstream version, so prefer the local one.
    try:
        from model.modeling_dream import DreamModel  # noqa: WPS433
        model = DreamModel.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True).cuda().eval()
    except Exception:
        from transformers import AutoModel  # noqa: WPS433
        model = AutoModel.from_pretrained(model_name, torch_dtype=dtype, trust_remote_code=True).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    return model, tokenizer


# ----------------------------------------------------------------------
# Sampling helpers (copied from Fast-dLLM v1 Dream generation_utils_block)
# ----------------------------------------------------------------------

def _sample_tokens(logits, temperature: float = 0.0, top_p: float | None = None, top_k: int | None = None,
                   neg_entropy: bool = False):
    if temperature > 0:
        logits = logits / temperature
        if top_p is not None and top_p < 1:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, -float("inf"))
        if top_k is not None:
            kth = torch.topk(logits, top_k, dim=-1).values[..., -1:].clone()
            logits = logits.masked_fill(logits < kth, -float("inf"))
        probs = F.softmax(logits, dim=-1)
        x0 = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(*probs.shape[:-1])
    else:
        probs = F.softmax(logits, dim=-1)
        confidence, x0 = probs.max(dim=-1)
        return confidence, x0
    confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
    if neg_entropy:
        confidence = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1) * -1  # entropy as confidence proxy
    return confidence, x0


# ----------------------------------------------------------------------
# Generation loop (dual_cache + confidence_threshold) with probe arming
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
    mask_token_id: int,
    steps: int = 256,
    gen_length: int = 256,
    block_length: int = 32,
    threshold: float = 0.9,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
):
    """Run Dream block-diffusion generation with probe callbacks.

    Block-level callbacks (always called):
        on_block_start(block_idx, masked_positions_abs)  — before the block's first forward.
        on_block_end(block_idx)                          — after the block's last forward.

    Per-pass callbacks (intra_block=True only — fired around every forward):
        on_pass_start(block_idx, pass_idx, token_state_pre, indices_in_forward)
            token_state_pre: int8 [block_length] — 0=mask, 1=clean at pass start.
            indices_in_forward: list[int] — positions in the forward's hidden
                state tensor to record at. Absolute (s..e) at pass 0
                (full sequence), local (0..block_length) at pass ≥ 1
                (block-only via x[:, s:e]).
        on_pass_end(block_idx, pass_idx, revealed_this_pass)
            revealed_this_pass: bool [block_length] — positions revealed by this pass.

    Dream-specific quirks vs LLaDA:
    - **Logit shift.** Output logits are rotated by one position
      (`[logits[:, :1], logits[:, :-1]]`) — Dream's convention; see line
      ~454 of upstream `_sample`.
    - **Pass 0 reveals only position s.** The first forward computes
      logits over the whole sequence, but only `x[:, s]` (the block's
      first position) is sampled. Passes ≥ 1 do the parallel
      confidence-thresholded reveal over remaining mask positions.
    - **`dual_cache=True` + `replace_position`** at every block-only
      forward (pass ≥ 1) — Fast-dLLM's specific cache update interface.

    Returns (x, nfe).
    """
    B = prompt.shape[0]
    Lp = int(prompt.shape[1])
    max_length = Lp + gen_length

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    x = F.pad(prompt, (0, max_length - prompt.shape[1]), value=mask_token_id)

    # Dream uses "full" attention as default. tok_idx unused unless the prompt has padding.
    tok_idx = None
    attention_mask = "full"

    past_key_values = None
    nfe = 0

    for nb in range(num_blocks):
        s = Lp + nb * block_length
        e = s + block_length
        masked_positions_abs = list(range(s, e))

        on_block_start(nb, masked_positions_abs)

        # ---- Pass 0: full-sequence prefill, samples only position s ----
        if intra_block and on_pass_start is not None:
            pre_state = (x[0, s:e] != mask_token_id).to(torch.int8)
            on_pass_start(nb, 0, pre_state, list(range(s, e)))

        out = model(x, attention_mask, tok_idx, use_cache=True)
        nfe += 1
        past_key_values = out.past_key_values
        logits = out.logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        confidence, x0 = _sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)

        was_mask_at_s = (x[0, s] == mask_token_id).item()
        x[:, s] = x0[:, s]
        revealed_p0 = torch.zeros(block_length, dtype=torch.bool, device=x.device)
        if was_mask_at_s and (x[0, s] != mask_token_id).item():
            revealed_p0[0] = True   # position s = local index 0 in [s..e)

        if intra_block:
            if on_pass_end is not None:
                on_pass_end(nb, 0, revealed_p0)
        else:
            # Legacy contract: disarm right after the first forward.
            on_block_end(nb)

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, s:e] = True

        # ---- Passes ≥ 1: block-only forwards with cache + replace_position ----
        i = 1
        while True:
            if i >= steps_per_block:
                break
            if (x[:, s:e] == mask_token_id).sum() == 0:
                break

            if intra_block and on_pass_start is not None:
                pre_state = (x[0, s:e] != mask_token_id).to(torch.int8)
                on_pass_start(nb, i, pre_state, list(range(block_length)))

            mask_index = (x[:, s:e] == mask_token_id)
            current_attention_mask = (
                attention_mask if attention_mask == "full"
                else attention_mask[:, :, :, s:]
            )
            out = model(
                x[:, s:e],
                current_attention_mask,
                None,
                past_key_values=past_key_values,
                use_cache=True,
                dual_cache=True,
                replace_position=replace_position,
            )
            logits = out.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            mask_logits = logits[mask_index]
            confidence, x0 = _sample_tokens(
                mask_logits, temperature=temperature, top_p=top_p, top_k=top_k,
            )

            x_ = torch.full_like(x[:, s:e], mask_token_id, device=model.device, dtype=torch.long)
            full_confidence = torch.full_like(
                x[:, s:e], -float("inf"), device=model.device, dtype=logits.dtype,
            )
            x_[mask_index] = x0.clone()
            full_confidence[mask_index] = confidence

            current_transfer_tokens = int((x[:, s:e] == mask_token_id).sum().item())
            selected_confidence, select_index = torch.topk(full_confidence, current_transfer_tokens)
            transfer_index = torch.zeros_like(x_, dtype=torch.bool)
            transfer_index[0, select_index[0]] = True
            for k in range(1, current_transfer_tokens):
                if selected_confidence[0, k] < threshold:
                    transfer_index[0, select_index[0, k]] = False

            x[:, s:e][transfer_index] = x_[transfer_index]
            revealed_blk = transfer_index[0].to(torch.bool)
            nfe += 1

            if intra_block and on_pass_end is not None:
                on_pass_end(nb, i, revealed_blk)

            i += 1

        if intra_block:
            on_block_end(nb)

    return x, nfe
