"""Option-likelihood scoring for BELEBELE.

The generation-based path asks the model to write an answer and then parses the
text. That makes the measure depend on whether the model emits a parseable
string within the output budget, which is a property of its answer style rather
than of its reading comprehension. See the 2026-10-15 addendum in AGENTS.md.

Here the model never generates. Each of the four options is scored as a
continuation of the same rendered prompt, and the highest-scoring option is the
answer. Every item therefore yields a choice by construction.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from thesis_eval.benchmarks.belebele import parse_choice, prediction_question_id


OPTION_ORDER = ("A", "B", "C", "D")
CONTINUATION_MODES = ("option_text", "letter")
NORMALIZATIONS = ("mean", "sum")

_OPTION_FIELDS = {
    "A": "choice_a",
    "B": "choice_b",
    "C": "choice_c",
    "D": "choice_d",
}


def build_option_continuations(
    row: dict[str, Any], *, mode: str = "option_text"
) -> list[tuple[str, str]]:
    """Return the four scored continuations as (label, text) in A--D order.

    ``option_text`` scores the answer itself, which is what BELEBELE's own
    non-chat protocol does and what actually probes comprehension.
    ``letter`` scores the bare label and is retained only as a diagnostic.
    """
    if mode not in CONTINUATION_MODES:
        raise ValueError(
            f"Unknown continuation mode {mode!r}; expected one of {list(CONTINUATION_MODES)}"
        )
    continuations: list[tuple[str, str]] = []
    for label in OPTION_ORDER:
        if mode == "letter":
            continuations.append((label, f" {label}"))
            continue
        text = str(row[_OPTION_FIELDS[label]]).strip()
        if not text:
            raise ValueError(f"Empty BELEBELE option {label} for item {row.get('item_id')!r}")
        continuations.append((label, f" {text}"))
    return continuations


def _validated_stats(option_stats: dict[str, Any]) -> dict[str, tuple[float, int]]:
    missing = [label for label in OPTION_ORDER if label not in option_stats]
    if missing:
        raise ValueError(f"Missing option log-probabilities for {missing}")
    validated: dict[str, tuple[float, int]] = {}
    for label in OPTION_ORDER:
        entry = option_stats[label]
        count = int(entry["token_count"])
        # A zero-token continuation sums to 0.0, which beats every real
        # (negative) log-probability. Refuse it rather than silently winning.
        if count < 1:
            raise ValueError(
                f"Option {label} has {count} continuation tokens; cannot be scored"
            )
        validated[label] = (float(entry["logprob_sum"]), count)
    return validated


def select_choice(option_stats: dict[str, Any], *, normalization: str = "mean") -> str:
    """Pick the best-scoring option label.

    ``mean`` divides by the continuation token count. Without it, shorter
    options win on total log-probability alone, and BELEBELE distractors differ
    in length. Ties resolve to the earliest option label.
    """
    if normalization not in NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalization {normalization!r}; expected one of {list(NORMALIZATIONS)}"
        )
    validated = _validated_stats(option_stats)
    if normalization == "mean":
        scores = {label: total / count for label, (total, count) in validated.items()}
    else:
        scores = {label: total for label, (total, _) in validated.items()}
    return max(OPTION_ORDER, key=lambda label: scores[label])


def attach_option_likelihood_predictions(
    rows: list[dict[str, Any]],
    option_scores: list[dict[str, Any]],
    *,
    normalization: str = "mean",
    mode: str = "option_text",
) -> list[dict[str, Any]]:
    """Attach option log-probabilities and the selected answer to prediction rows.

    Both normalizations are stored alongside the token counts so the choice can
    be recomputed offline without a GPU.
    """
    if len(rows) != len(option_scores):
        raise ValueError(
            f"BELEBELE row count ({len(rows)}) and option-score count "
            f"({len(option_scores)}) differ"
        )
    if normalization not in NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalization {normalization!r}; expected one of {list(NORMALIZATIONS)}"
        )
    if mode not in CONTINUATION_MODES:
        raise ValueError(
            f"Unknown continuation mode {mode!r}; expected one of {list(CONTINUATION_MODES)}"
        )
    enriched: list[dict[str, Any]] = []
    for row, stats in zip(rows, option_scores, strict=True):
        validated = _validated_stats(stats)
        sums = {label: total for label, (total, _) in validated.items()}
        counts = {label: count for label, (_, count) in validated.items()}
        means = {label: sums[label] / counts[label] for label in OPTION_ORDER}
        choice_mean = select_choice(stats, normalization="mean")
        choice_sum = select_choice(stats, normalization="sum")
        enriched_row = dict(row)
        enriched_row.update(
            {
                "scoring_method": "option_likelihood",
                "likelihood_normalization": normalization,
                "continuation_mode": mode,
                "option_logprob_sum": sums,
                "option_logprob_mean": means,
                "option_token_counts": counts,
                "predicted_choice_sum": choice_sum,
                "predicted_choice_mean": choice_mean,
                "predicted_choice": choice_mean if normalization == "mean" else choice_sum,
                # Nothing was generated, so there is no model text to retain.
                "raw_output": None,
                "generation_completed": True,
            }
        )
        enriched_row["prediction"] = enriched_row["predicted_choice"]
        enriched.append(enriched_row)
    return enriched


def _choice_of(row: dict[str, Any]) -> str | None:
    value = row.get("predicted_choice")
    if value is None:
        return None
    return parse_choice(str(value))


def compare_scoring_paths(
    generated: list[dict[str, Any]], likelihood: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Summarize agreement between the generation-parsed and likelihood paths.

    Agreement is computed only on items the generation path actually parsed,
    because the unparsed rows are exactly what the likelihood path exists to
    replace. It is the only available calibration for retaining a
    generation-parsed number anywhere (AGENTS.md, 2026-10-15 addendum), so it is
    reported per model--language cell rather than pooled.
    """
    indexed: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in likelihood:
        key = (str(row["model"]), str(row["language"]))
        indexed[key][prediction_question_id(row)] = row

    grouped: dict[tuple[str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row in generated:
        key = (str(row["model"]), str(row["language"]))
        match = indexed.get(key, {}).get(prediction_question_id(row))
        if match is not None:
            grouped[key].append((row, match))

    summaries: list[dict[str, Any]] = []
    for (model, language), pairs in sorted(grouped.items()):
        joined = len(pairs)
        generated_parsed = 0
        likelihood_parsed = 0
        agreed = 0
        correct_generated = 0
        correct_likelihood = 0
        for generated_row, likelihood_row in pairs:
            gold = parse_choice(str(generated_row.get("gold", "")))
            generated_choice = _choice_of(generated_row)
            likelihood_choice = _choice_of(likelihood_row)
            if generated_choice is not None:
                generated_parsed += 1
                if generated_choice == likelihood_choice:
                    agreed += 1
                if gold is not None and generated_choice == gold:
                    correct_generated += 1
            if likelihood_choice is not None:
                likelihood_parsed += 1
                if gold is not None and likelihood_choice == gold:
                    correct_likelihood += 1
        summaries.append(
            {
                "model": model,
                "attack_language": language,
                "n_joined": joined,
                "n_generated_parsed": generated_parsed,
                "n_likelihood_parsed": likelihood_parsed,
                "n_agreed": agreed,
                "agreement_on_generated_parsed": (
                    agreed / generated_parsed if generated_parsed else None
                ),
                "parse_rate_generated": generated_parsed / joined if joined else None,
                "parse_rate_likelihood": likelihood_parsed / joined if joined else None,
                "if_generated": correct_generated / joined if joined else None,
                "if_likelihood": correct_likelihood / joined if joined else None,
            }
        )
    return summaries
