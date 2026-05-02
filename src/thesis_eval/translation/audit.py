from __future__ import annotations

from typing import Any


AUDIT_COLUMNS = [
    "prompt_id",
    "target_language",
    "source_text",
    "translated_text",
    "blaser_qe_score",
    "blaser_status",
    "harm_preservation",
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
                "blaser_qe_score": log.get("blaser_qe_score"),
                "blaser_status": log.get("blaser_status"),
                "harm_preservation": log.get("harm_preservation"),
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
        output.append(row)
    return output
