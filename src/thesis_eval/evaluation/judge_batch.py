from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from thesis_eval.evaluation.majority_vote import JUDGE_COLUMNS, normalize_raw_label, vote_from_row
from thesis_eval.evaluation.prompts import build_judge_messages


@dataclass(frozen=True)
class JudgeRequest:
    request_id: str
    judge: str
    model: str
    aligned_language: str
    attack_language: str
    prompt_id: str
    run_id: str
    harmful_prompt: str
    model_response: str
    api_messages: list[dict[str, str]]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def build_judge_requests(generations: list[dict[str, Any]], judges: list[str]) -> list[dict[str, str]]:
    requests: list[dict[str, str]] = []
    for generation in generations:
        for judge in judges:
            request_id = f"{generation['run_id']}::{judge}"
            requests.append(
                JudgeRequest(
                    request_id=request_id,
                    judge=judge,
                    model=str(generation["model"]),
                    aligned_language=str(generation["aligned_language"]),
                    attack_language=str(generation["attack_language"]),
                    prompt_id=str(generation["prompt_id"]),
                    run_id=str(generation["run_id"]),
                    harmful_prompt=str(generation["prompt_text"]),
                    model_response=str(generation["model_output"]),
                    api_messages=build_judge_messages(str(generation["prompt_text"]), str(generation["model_output"])),
                ).to_dict()
            )
    return requests


def normalize_judge_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        raw = str(row.get("label") or row.get("raw_label") or row.get("raw_response") or "")
        label = normalize_raw_label(raw)
        normalized.append(
            {
                "request_id": row["request_id"],
                "run_id": row["run_id"],
                "judge": row["judge"],
                "label": label,
                "raw_response": row.get("raw_response", raw),
                "failure": row.get("failure"),
            }
        )
    return normalized


def combine_judge_results(generations: list[dict[str, Any]], judge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_run: dict[str, dict[str, str]] = defaultdict(dict)
    for result in normalize_judge_results(judge_rows):
        by_run[str(result["run_id"])][str(result["judge"])] = str(result["label"])

    combined: list[dict[str, Any]] = []
    for generation in generations:
        row = dict(generation)
        run_id = str(generation["run_id"])
        labels = by_run.get(run_id, {})
        missing = [judge for judge in JUDGE_COLUMNS if judge not in labels]
        if missing:
            raise ValueError(f"Run {run_id} is missing judge results: {missing}")
        for judge, column in JUDGE_COLUMNS.items():
            row[column] = labels[judge]
        combined.append(vote_from_row(row))
    return combined
