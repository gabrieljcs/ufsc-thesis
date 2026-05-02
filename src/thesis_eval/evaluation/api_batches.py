from __future__ import annotations

from typing import Any

from thesis_eval.evaluation.majority_vote import normalize_raw_label


def split_request_id(request_id: str) -> tuple[str, str]:
    if "::" not in request_id:
        raise ValueError(f"request_id must contain '::' separator: {request_id}")
    run_id, judge = request_id.rsplit("::", 1)
    return run_id, judge


def openai_batch_rows(requests: list[dict[str, Any]], model_id: str, max_tokens: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for request in requests:
        rows.append(
            {
                "custom_id": request["request_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model_id,
                    "messages": request["api_messages"],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                },
            }
        )
    return rows


def anthropic_batch_body(requests: list[dict[str, Any]], model_id: str, max_tokens: int = 8) -> dict[str, Any]:
    batch_requests: list[dict[str, Any]] = []
    for request in requests:
        messages = request["api_messages"]
        system_parts = [message["content"] for message in messages if message["role"] == "system"]
        user_messages = [message for message in messages if message["role"] != "system"]
        params: dict[str, Any] = {
            "model": model_id,
            "max_tokens": max_tokens,
            "temperature": 0,
            "messages": user_messages,
        }
        if system_parts:
            params["system"] = "\n\n".join(system_parts)
        batch_requests.append({"custom_id": request["request_id"], "params": params})
    return {"requests": batch_requests}


def hosted_judge_rows(requests: list[dict[str, Any]], model_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for request in requests:
        rows.append(
            {
                "custom_id": request["request_id"],
                "model": model_id,
                "messages": request["api_messages"],
                "metadata": {
                    "judge": request["judge"],
                    "run_id": request["run_id"],
                    "prompt_id": request["prompt_id"],
                    "attack_language": request["attack_language"],
                },
            }
        )
    return rows


def parse_openai_batch_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        request_id = row["custom_id"]
        run_id, judge = split_request_id(request_id)
        error = row.get("error")
        raw_response = ""
        label = None
        if error is None:
            try:
                raw_response = row["response"]["body"]["choices"][0]["message"]["content"]
                label = normalize_raw_label(str(raw_response))
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                error = {"code": "parse_error", "message": str(exc)}
        output.append(
            {
                "request_id": request_id,
                "run_id": run_id,
                "judge": judge,
                "label": label,
                "raw_response": raw_response,
                "failure": error,
            }
        )
    return output


def parse_anthropic_batch_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        request_id = row.get("custom_id") or row.get("id")
        if not request_id:
            raise ValueError(f"Anthropic result missing custom_id/id: {row}")
        run_id, judge = split_request_id(str(request_id))
        result = row.get("result", {})
        failure = None
        raw_response = ""
        label = None
        if result.get("type") == "succeeded":
            content = result.get("message", {}).get("content", [])
            raw_response = "\n".join(part.get("text", "") for part in content if part.get("type") == "text")
            try:
                label = normalize_raw_label(raw_response)
            except ValueError as exc:
                failure = {"code": "parse_error", "message": str(exc)}
        else:
            failure = result or {"code": "missing_result", "message": "Anthropic result did not succeed"}
        output.append(
            {
                "request_id": str(request_id),
                "run_id": run_id,
                "judge": judge,
                "label": label,
                "raw_response": raw_response,
                "failure": failure,
            }
        )
    return output


def normalize_hosted_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        request_id = row.get("request_id") or row.get("custom_id")
        if not request_id:
            raise ValueError(f"Hosted judge result missing request_id/custom_id: {row}")
        run_id, judge = split_request_id(str(request_id))
        raw_response = str(row.get("label") or row.get("raw_label") or row.get("raw_response") or "")
        label = None
        failure = row.get("failure")
        try:
            label = normalize_raw_label(raw_response)
        except ValueError as exc:
            failure = failure or {"code": "parse_error", "message": str(exc)}
        output.append(
            {
                "request_id": str(request_id),
                "run_id": run_id,
                "judge": judge,
                "label": label,
                "raw_response": raw_response,
                "failure": failure,
            }
        )
    return output
