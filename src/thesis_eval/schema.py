from __future__ import annotations

from collections.abc import Iterable, Mapping


LABELS = {"safe", "unsafe", "invalid"}

ANALYSIS_COLUMNS = [
    "model",
    "aligned_language",
    "attack_language",
    "prompt_id",
    "run_id",
    "judge_llama",
    "judge_gpt4o",
    "judge_claude",
    "majority_label",
    "unsafe_binary",
    "invalid_binary",
    "judge_disagreement",
    "distance",
    "if_score",
    "cons_score",
    "spec_score",
    "input_tokens",
    "tokens_per_char",
    "token_inflation",
    "benchmark",
    "excluded",
    "exclusion_reason",
]

TRANSLATION_LOG_FIELDS = [
    "prompt_id",
    "source_language",
    "target_language",
    "harmful_goal",
    "expected_output_form",
    "fixed_constraints",
    "allowed_adaptation",
    "translator_config",
    "blaser_qe_score",
    "blaser_status",
    "human_audit_status",
    "reference_subset_score",
    "harm_preservation",
    "revision_note",
    "source_text",
    "translated_text",
]


def validate_label(label: str) -> str:
    normalized = label.strip().lower()
    if normalized not in LABELS:
        raise ValueError(f"Unknown label {label!r}; expected one of {sorted(LABELS)}")
    return normalized


def validate_analysis_rows(rows: Iterable[Mapping[str, object]]) -> None:
    required = set(ANALYSIS_COLUMNS)
    for index, row in enumerate(rows, start=1):
        missing = required.difference(row)
        if missing:
            raise ValueError(f"Row {index} missing columns: {sorted(missing)}")
        for key in ("judge_llama", "judge_gpt4o", "judge_claude", "majority_label"):
            if row[key] is not None:
                validate_label(str(row[key]))


def validate_translation_log(record: Mapping[str, object]) -> None:
    missing = set(TRANSLATION_LOG_FIELDS).difference(record)
    if missing:
        raise ValueError(f"Translation log missing fields: {sorted(missing)}")
    if record["source_language"] != "eng":
        raise ValueError("StrongREJECT translation logs must use source_language='eng'")
    if record["blaser_status"] not in {"pass", "flag", "fail", "pending"}:
        raise ValueError("blaser_status must be pass, flag, fail, or pending")
    if record["human_audit_status"] not in {"not_audited", "audited_pass", "audited_revised"}:
        raise ValueError("human_audit_status has an unexpected value")
    if record["harm_preservation"] not in {"preserved", "attenuated", "amplified", "pending"}:
        raise ValueError("harm_preservation has an unexpected value")
