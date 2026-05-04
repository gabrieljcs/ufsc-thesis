from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re
from typing import Any

from thesis_eval.io import read_jsonl
from thesis_eval.metrics.spec import add_spec_scores


OPTIONS = {"A", "B", "C", "D"}
BELEBELE_LANGUAGE_CODES = {
    "ara": "arb_Arab",
    "bul": "bul_Cyrl",
    "dan": "dan_Latn",
    "eng": "eng_Latn",
    "fin": "fin_Latn",
    "ita": "ita_Latn",
    "nor": "nob_Latn",
    "por": "por_Latn",
    "rus": "rus_Cyrl",
    "spa": "spa_Latn",
    "swa": "swh_Latn",
    "swe": "swe_Latn",
    "ukr": "ukr_Cyrl",
}
ANSWER_BY_NUMBER = {
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
}


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


def resolve_belebele_language_code(language: str) -> str:
    if language not in BELEBELE_LANGUAGE_CODES:
        raise ValueError(
            f"Unsupported BELEBELE language {language!r}; expected one of {sorted(BELEBELE_LANGUAGE_CODES)}"
        )
    return BELEBELE_LANGUAGE_CODES[language]


def load_belebele_rows(dataset_dir: Path, language: str, limit: int | None = None) -> list[dict[str, Any]]:
    data_dir = dataset_dir / "data" if (dataset_dir / "data").is_dir() else dataset_dir
    dialect = resolve_belebele_language_code(language)
    path = data_dir / f"{dialect}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing BELEBELE file for {language}: {path}")
    rows = read_jsonl(path)
    if limit is not None:
        return rows[:limit]
    return rows


def build_belebele_prompt(row: dict[str, Any]) -> str:
    return (
        "Passage:\n"
        f"{row['flores_passage']}\n\n"
        "Question:\n"
        f"{row['question']}\n\n"
        "Choices:\n"
        f"A. {row['mc_answer1']}\n"
        f"B. {row['mc_answer2']}\n"
        f"C. {row['mc_answer3']}\n"
        f"D. {row['mc_answer4']}\n\n"
        "Reply with only A, B, C, or D."
    )


def build_belebele_prediction_rows(
    dataset_rows: list[dict[str, Any]],
    *,
    model: str,
    language: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in dataset_rows:
        answer_number = str(row["correct_answer_num"])
        if answer_number not in ANSWER_BY_NUMBER:
            raise ValueError(f"Unexpected BELEBELE correct_answer_num={answer_number!r}")
        item_id = f"belebele_{int(row['question_number']):04d}"
        output.append(
            {
                "model": model,
                "language": language,
                "dataset_dialect": str(row["dialect"]),
                "item_id": item_id,
                "question_number": int(row["question_number"]),
                "gold": ANSWER_BY_NUMBER[answer_number],
            }
        )
    return output


def attach_belebele_predictions(rows: list[dict[str, Any]], outputs: list[Any]) -> list[dict[str, Any]]:
    if len(rows) != len(outputs):
        raise ValueError("BELEBELE row count and output count differ")
    enriched: list[dict[str, Any]] = []
    for row, output in zip(rows, outputs, strict=True):
        text = str(output.get("text", "")) if isinstance(output, dict) else str(output)
        predicted = parse_choice(text)
        enriched_row = dict(row)
        enriched_row["raw_output"] = text
        enriched_row["prediction"] = predicted if predicted is not None else text.strip()
        enriched_row["predicted_choice"] = predicted
        if isinstance(output, dict):
            for key in (
                "generation_completed",
                "api_http_status",
                "api_failure",
                "api_failure_reason",
                "provider_block",
                "provider_block_reason",
            ):
                if key in output:
                    enriched_row[key] = output[key]
        else:
            enriched_row["generation_completed"] = True
        enriched.append(enriched_row)
    return enriched
