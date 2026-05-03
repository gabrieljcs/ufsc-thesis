from __future__ import annotations

from collections import defaultdict
import math
from typing import Any

from thesis_eval.metrics.asr import wilson_ci


def coverage_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model"])].append(row)
    output: list[dict[str, Any]] = []
    for model, model_rows in sorted(grouped.items()):
        total = len(model_rows)
        output.append(
            {
                "model": model,
                "total_attempts": total,
                "api_failure_rate": _mean_bool(model_rows, "api_failure"),
                "provider_block_rate": _mean_bool(model_rows, "provider_block"),
                "scoring_failure_rate": _mean_bool(model_rows, "scoring_failed"),
                "refusal_rate": _mean_numeric(model_rows, "refusal_binary"),
            }
        )
    return output


def asr_by_model_language(rows: list[dict[str, Any]], exclude_aligned: bool = False) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("excluded") or row.get("unsafe_binary") is None:
            continue
        if exclude_aligned and row.get("attack_language") == row.get("aligned_language"):
            continue
        grouped[(str(row["model"]), str(row["attack_language"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model, language), group_rows in sorted(grouped.items()):
        total = len(group_rows)
        successes = sum(1 for row in group_rows if int(row["unsafe_binary"]) == 1)
        low, high = wilson_ci(successes, total)
        output.append(
            {
                "model": model,
                "attack_language": language,
                "n": total,
                "unsafe": successes,
                "asr": successes / total,
                "ci_low": low,
                "ci_high": high,
                "mean_strongreject_score": _mean_numeric(group_rows, "strongreject_score"),
                "refusal_rate": _mean_numeric(group_rows, "refusal_binary"),
            }
        )
    return output


def closest_farthest_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries = asr_by_model_language(rows)
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    distances: dict[tuple[str, str], float] = {}
    for row in rows:
        if row.get("distance") is not None:
            distances[(str(row["model"]), str(row["attack_language"]))] = float(row["distance"])
    for row in summaries:
        by_model[str(row["model"])].append(row)
    output: list[dict[str, Any]] = []
    for model, model_rows in sorted(by_model.items()):
        with_distance = [row for row in model_rows if (model, str(row["attack_language"])) in distances]
        if not with_distance:
            continue
        closest = min(with_distance, key=lambda row: distances[(model, str(row["attack_language"]))])
        farthest = max(with_distance, key=lambda row: distances[(model, str(row["attack_language"]))])
        output.append(
            {
                "model": model,
                "closest_language": closest["attack_language"],
                "closest_distance": distances[(model, str(closest["attack_language"]))],
                "closest_asr": closest["asr"],
                "farthest_language": farthest["attack_language"],
                "farthest_distance": distances[(model, str(farthest["attack_language"]))],
                "farthest_asr": farthest["asr"],
                "gap": farthest["asr"] - closest["asr"],
            }
        )
    return output


def spearman_tables(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries = asr_by_model_language(rows)
    distance_rows: list[dict[str, Any]] = []
    spec_rows: list[dict[str, Any]] = []
    by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    predictors: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (str(row["model"]), str(row["attack_language"]))
        if row.get("distance") is not None:
            predictors[key]["distance"] = float(row["distance"])
        if row.get("spec_score") is not None:
            predictors[key]["spec_score"] = float(row["spec_score"])
    for row in summaries:
        by_model[str(row["model"])].append(row)
    for model, model_rows in sorted(by_model.items()):
        distance_pairs = [
            (predictors[(model, str(row["attack_language"]))]["distance"], float(row["asr"]))
            for row in model_rows
            if "distance" in predictors[(model, str(row["attack_language"]))]
        ]
        spec_pairs = [
            (predictors[(model, str(row["attack_language"]))]["spec_score"], float(row["asr"]))
            for row in model_rows
            if "spec_score" in predictors[(model, str(row["attack_language"]))]
        ]
        distance_rows.append(_correlation_row(model, "distance", distance_pairs))
        spec_rows.append(_correlation_row(model, "spec_score", spec_pairs))
    return distance_rows, spec_rows


def tokenizer_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("excluded"):
            continue
        grouped[(str(row["model"]), str(row["attack_language"]))].append(row)
    output: list[dict[str, Any]] = []
    for (model, language), group_rows in sorted(grouped.items()):
        output.append(
            {
                "model": model,
                "attack_language": language,
                "n": len(group_rows),
                "mean_token_inflation": _mean_numeric(group_rows, "token_inflation"),
                "mean_tokens_per_char": _mean_numeric(group_rows, "tokens_per_char"),
                "truncation_risk_rate": _mean_bool(group_rows, "truncation_risk"),
            }
        )
    return output


def collinearity_diagnostics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = [
        (float(row["distance"]), float(row["spec_score"]))
        for row in rows
        if not row.get("excluded") and row.get("distance") is not None and row.get("spec_score") is not None
    ]
    if len(pairs) < 2:
        return [
            {"diagnostic": "vif_distance", "value": None},
            {"diagnostic": "vif_spec", "value": None},
            {"diagnostic": "condition_number", "value": None},
            {"diagnostic": "single_predictor_rerun_triggered", "value": None},
        ]
    r = _pearson([pair[0] for pair in pairs], [pair[1] for pair in pairs])
    vif = 1 / max(1e-12, 1 - r * r)
    condition = math.sqrt((1 + abs(r)) / max(1e-12, 1 - abs(r)))
    return [
        {"diagnostic": "vif_distance", "value": vif},
        {"diagnostic": "vif_spec", "value": vif},
        {"diagnostic": "condition_number", "value": condition},
        {"diagnostic": "single_predictor_rerun_triggered", "value": bool(vif > 5)},
    ]


def _mean_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1 for row in rows if row.get(key)) / len(rows)


def _mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def _correlation_row(model: str, predictor: str, pairs: list[tuple[float, float]]) -> dict[str, Any]:
    return {
        "model": model,
        "predictor": predictor,
        "n": len(pairs),
        "spearman_rho": _spearman([pair[0] for pair in pairs], [pair[1] for pair in pairs]) if len(pairs) >= 2 else None,
        "p_value": None,
        "interpretation": "descriptive_only",
    }


def _spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_ranks(xs), _ranks(ys))


def _pearson(xs: list[float], ys: list[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def _ranks(values: list[float]) -> list[float]:
    sorted_values = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0 for _ in values]
    position = 0
    while position < len(sorted_values):
        end = position
        while end + 1 < len(sorted_values) and sorted_values[end + 1][0] == sorted_values[position][0]:
            end += 1
        rank = (position + end + 2) / 2
        for _, index in sorted_values[position : end + 1]:
            ranks[index] = rank
        position = end + 1
    return ranks
