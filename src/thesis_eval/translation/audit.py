from __future__ import annotations

from typing import Any


AUDIT_COLUMNS = [
    "prompt_id",
    "target_language",
    "source_text",
    "translated_text",
    "roundtrip_text",
    "roundtrip_drift_score",
    "roundtrip_status",
    "blaser_qe_score",
    "blaser_status",
    "harm_preservation",
    "xsts_semantic_equivalence",
    "xsts_attack_strength",
    "xsts_constraint_preservation",
    "xsts_naturalness",
    "xsts_no_harmfulness_drift",
    "audit_decision",
    "xsts_note",
    "revision_note",
]


def build_audit_queue(logs: list[dict[str, Any]], include_passed: bool = False) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    for log in logs:
        needs_audit = include_passed or log.get("blaser_status") in {"flag", "fail"} or log.get("harm_preservation") != "preserved"
        if not needs_audit:
            continue
        queue.append(
            {
                "prompt_id": log["prompt_id"],
                "target_language": log["target_language"],
                "source_text": log["source_text"],
                "translated_text": log["translated_text"],
                "roundtrip_text": log.get("roundtrip_text"),
                "roundtrip_drift_score": log.get("roundtrip_drift_score"),
                "roundtrip_status": log.get("roundtrip_status"),
                "blaser_qe_score": log.get("blaser_qe_score"),
                "blaser_status": log.get("blaser_status"),
                "harm_preservation": log.get("harm_preservation"),
                "xsts_semantic_equivalence": log.get("xsts_semantic_equivalence", "pending"),
                "xsts_attack_strength": log.get("xsts_attack_strength", "pending"),
                "xsts_constraint_preservation": log.get("xsts_constraint_preservation", "pending"),
                "xsts_naturalness": log.get("xsts_naturalness", "pending"),
                "xsts_no_harmfulness_drift": log.get("xsts_no_harmfulness_drift", "pending"),
                "audit_decision": "",
                "xsts_note": "",
                "revision_note": "",
            }
        )
    return queue


def apply_audit_decisions(logs: list[dict[str, Any]], decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decision_by_key = {(row["prompt_id"], row["target_language"]): row for row in decisions}
    output: list[dict[str, Any]] = []
    for log in logs:
        row = dict(log)
        decision = decision_by_key.get((row["prompt_id"], row["target_language"]))
        if decision:
            audit_decision = str(decision.get("audit_decision", "")).strip().lower()
            if audit_decision not in {"pass", "revise"}:
                raise ValueError(f"audit_decision must be pass or revise for {row['prompt_id']}:{row['target_language']}")
            row["human_audit_status"] = "audited_pass" if audit_decision == "pass" else "audited_revised"
            if decision.get("revision_note"):
                row["revision_note"] = decision["revision_note"]
            if decision.get("harm_preservation") in {"preserved", "attenuated", "amplified"}:
                row["harm_preservation"] = decision["harm_preservation"]
            if decision.get("translated_text"):
                row["translated_text"] = decision["translated_text"]
                row["translation_revised"] = True
            for key in (
                "xsts_semantic_equivalence",
                "xsts_attack_strength",
                "xsts_constraint_preservation",
                "xsts_naturalness",
                "xsts_no_harmfulness_drift",
            ):
                if decision.get(key):
                    row[key] = decision[key]
            if audit_decision == "revise":
                row["translation_revised"] = True
        output.append(row)
    return output
