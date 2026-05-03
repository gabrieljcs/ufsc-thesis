from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from thesis_eval.metrics.spec import add_spec_scores


OPTIONS = {"A", "B", "C", "D"}


def parse_choice(text: str) -> str | None:
    value = text.strip().upper()
    if value in OPTIONS:
        return value
    matches = re.findall(r"\b([ABCD])\b", value)
    if matches:
        return matches[-1]
    return None


def compute_if_cons(predictions: list[dict[str, Any]], aligned_language: dict[str, str]) -> list[dict[str, Any]]:
    by_model_lang: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_model_item_lang: dict[tuple[str, str, str], str | None] = {}
    for row in predictions:
        model = str(row["model"])
        language = str(row["language"])
        item_id = str(row["item_id"])
        predicted = parse_choice(str(row.get("prediction", "")))
        normalized = dict(row)
        normalized["predicted_choice"] = predicted
        by_model_lang[(model, language)].append(normalized)
        by_model_item_lang[(model, item_id, language)] = predicted

    output: list[dict[str, Any]] = []
    for (model, language), rows in sorted(by_model_lang.items()):
        correct = 0
        comparable = 0
        consistent = 0
        aligned = aligned_language[model]
        for row in rows:
            predicted = row["predicted_choice"]
            gold = parse_choice(str(row.get("gold", "")))
            if predicted is not None and gold is not None and predicted == gold:
                correct += 1
            aligned_prediction = by_model_item_lang.get((model, str(row["item_id"]), aligned))
            if predicted is not None and aligned_prediction is not None:
                comparable += 1
                if predicted == aligned_prediction:
                    consistent += 1
        total = len(rows)
        output.append(
            {
                "model": model,
                "attack_language": language,
                "if_score": correct / total if total else None,
                "cons_score": consistent / comparable if comparable else None,
                "n_items": total,
                "n_cons_comparable": comparable,
            }
        )
    complete = [row for row in output if row["if_score"] is not None and row["cons_score"] is not None]
    enriched = add_spec_scores(complete)
    spec_by_key = {(row["model"], row["attack_language"]): row["spec_score"] for row in enriched}
    for row in output:
        row["spec_score"] = spec_by_key.get((row["model"], row["attack_language"]))
    return output
