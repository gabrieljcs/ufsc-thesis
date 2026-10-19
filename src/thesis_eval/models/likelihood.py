"""Conditional log-probability scoring of fixed continuations.

Used by the BELEBELE option-likelihood path (AGENTS.md, 2026-10-15 addendum).
The index arithmetic and the gather are kept as pure functions so they can be
tested without torch; only :func:`build_option_likelihood_runner` needs a model.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from thesis_eval.progress import info, step


def common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    """Number of shared leading token ids.

    Concatenating a continuation onto a prompt can re-tokenize the boundary, so
    the continuation span is measured against the joint encoding instead of
    assuming the prompt encodes identically on its own.
    """
    length = 0
    for a, b in zip(left, right, strict=False):
        if int(a) != int(b):
            break
        length += 1
    return length


def plan_sequence_batches(
    item_sequence_counts: Sequence[int], batch_size: int
) -> list[list[tuple[int, int]]]:
    """Group ``(item_index, sequence_index)`` pairs into forward-pass batches.

    Batches span item boundaries. Each item contributes only four short
    continuations, so batching per item would leave the GPU badly underfed and
    make ``--gpu-batch-size`` above four a no-op.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1; got {batch_size}")
    flat = [
        (item_index, sequence_index)
        for item_index, count in enumerate(item_sequence_counts)
        for sequence_index in range(int(count))
    ]
    return [flat[start : start + batch_size] for start in range(0, len(flat), batch_size)]


def continuation_logit_positions(*, prefix_len: int, total_len: int) -> list[int]:
    """Logit positions predicting the continuation tokens.

    A causal model predicts the token at position ``t`` from the logits at
    ``t - 1``, so the continuation at ``[prefix_len, total_len)`` is predicted
    from ``[prefix_len - 1, total_len - 1)``.
    """
    if prefix_len < 1:
        raise ValueError(
            f"prefix_len must be at least 1 to predict from; got {prefix_len}"
        )
    if total_len <= prefix_len:
        raise ValueError(
            f"Empty continuation: total_len={total_len} must exceed prefix_len={prefix_len}"
        )
    return list(range(prefix_len - 1, total_len - 1))


def score_continuation(
    log_probs: Any, target_token_ids: Sequence[int]
) -> tuple[float, int]:
    """Sum the log-probability of each target token.

    ``log_probs`` holds one already-log-softmaxed row per target token, aligned
    so that row ``i`` predicts ``target_token_ids[i]``.
    """
    rows = np.asarray(log_probs)
    if rows.ndim != 2:
        raise ValueError(f"log_probs must be 2-D [tokens, vocab]; got shape {rows.shape}")
    targets = [int(token) for token in target_token_ids]
    if rows.shape[0] != len(targets):
        raise ValueError(
            f"log_probs has {rows.shape[0]} rows but {len(targets)} target tokens"
        )
    if not targets:
        raise ValueError("Cannot score an empty continuation")
    vocab_size = int(rows.shape[1])
    for token in targets:
        if not 0 <= token < vocab_size:
            raise ValueError(f"Target token {token} outside vocabulary of {vocab_size}")
    total = float(sum(float(rows[index, token]) for index, token in enumerate(targets)))
    return total, len(targets)


def build_option_likelihood_runner(
    model_ref: str,
    *,
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    gpu_batch_size: int = 16,
) -> Callable[[list[tuple[str, list[tuple[str, str]]]]], list[dict[str, dict[str, Any]]]]:
    """Return a callable scoring each item's option continuations.

    The returned function takes ``[(prompt, [(label, continuation), ...]), ...]``
    and returns one ``{label: {"logprob_sum": float, "token_count": int}}`` per
    item. Prompts are rendered with the same native chat template used for
    generation, so the two scoring paths see identical model inputs.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Option-likelihood scoring requires torch and transformers. "
            "Install the mac or rtx extras."
        ) from exc

    from thesis_eval.models.generation import (
        _is_local_model_ref,
        _resolve_device,
        _resolve_dtype,
        format_generation_prompt,
        prompt_format_metadata_for_tokenizer,
    )

    if gpu_batch_size < 1:
        raise ValueError(f"gpu_batch_size must be >= 1; got {gpu_batch_size}")

    resolved_device = _resolve_device(device)
    model_dtype = _resolve_dtype(dtype, resolved_device)
    local_only = _is_local_model_ref(model_ref)
    tokenizer = AutoTokenizer.from_pretrained(
        model_ref, trust_remote_code=trust_remote_code, local_files_only=local_only
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        local_files_only=local_only,
    )
    if resolved_device != "cpu":
        with step(f"move model to {resolved_device}"):
            model = model.to(resolved_device)
    model.eval()
    metadata = prompt_format_metadata_for_tokenizer(tokenizer)
    info(
        "Option-likelihood scoring ready "
        f"device={resolved_device} "
        f"strategy={metadata['strategy']} "
        f"chat_template={metadata['uses_chat_template']} "
        f"system_prompt={metadata['system_prompt_id'] or 'none'} "
        f"gpu_batch_size={gpu_batch_size}"
    )
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        raise RuntimeError(
            f"Tokenizer for {model_ref} exposes neither pad_token nor eos_token; cannot batch."
        )
    pad_id = int(pad_id)

    def _encode_item(prompt: str, continuations: list[tuple[str, str]]):
        rendered = format_generation_prompt(tokenizer, prompt)
        prefix_ids = tokenizer(rendered, add_special_tokens=True)["input_ids"]
        encoded: list[tuple[str, list[int], int]] = []
        for label, continuation in continuations:
            full_ids = tokenizer(rendered + continuation, add_special_tokens=True)["input_ids"]
            prefix_len = common_prefix_length(prefix_ids, full_ids)
            if prefix_len < 1 or len(full_ids) <= prefix_len:
                raise RuntimeError(
                    f"Continuation {label!r} did not extend the prompt encoding for {model_ref}; "
                    "the tokenizer re-encoded the boundary beyond recovery."
                )
            encoded.append((label, [int(token) for token in full_ids], prefix_len))
        return encoded

    def _score_sequences(
        sequences: list[tuple[str, list[int], int]]
    ) -> list[tuple[str, float, int]]:
        # Padding is on the right. Causal attention means a position can only
        # attend to earlier ones, so trailing pad never reaches a scored
        # position; the mask keeps it out of the attention normalization too.
        max_len = max(len(ids) for _, ids, _ in sequences)
        input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for index, (_, ids, _) in enumerate(sequences):
            input_ids[index, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[index, : len(ids)] = 1
        if resolved_device != "cpu":
            input_ids = input_ids.to(resolved_device)
            attention_mask = attention_mask.to(resolved_device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        scored: list[tuple[str, float, int]] = []
        for index, (label, ids, prefix_len) in enumerate(sequences):
            positions = continuation_logit_positions(
                prefix_len=prefix_len, total_len=len(ids)
            )
            # Slice the sequence dimension before the softmax: only a handful of
            # positions matter, so the full [seq, vocab] tensor never leaves the
            # device.
            selected = logits[index, positions, :]
            row_log_probs = (
                torch.log_softmax(selected.float(), dim=-1).cpu().numpy()
            )
            total, count = score_continuation(row_log_probs, ids[prefix_len : len(ids)])
            scored.append((label, total, count))
        return scored

    def _run(
        items: list[tuple[str, list[tuple[str, str]]]]
    ) -> list[dict[str, dict[str, Any]]]:
        encoded = [_encode_item(prompt, continuations) for prompt, continuations in items]
        results: list[dict[str, dict[str, Any]]] = [{} for _ in encoded]
        # Sort each batch's members by length so a long sequence does not pad out
        # a batch of short ones. Grouping is by global position, so batches still
        # span items and every sequence is scheduled exactly once.
        plan = plan_sequence_batches([len(group) for group in encoded], gpu_batch_size)
        for batch in plan:
            ordered = sorted(batch, key=lambda ref: len(encoded[ref[0]][ref[1]][1]))
            sequences = [encoded[item_index][seq_index] for item_index, seq_index in ordered]
            for (item_index, _), (label, total, count) in zip(
                ordered, _score_sequences(sequences), strict=True
            ):
                results[item_index][label] = {
                    "logprob_sum": total,
                    "token_count": count,
                }
        return results

    return _run
