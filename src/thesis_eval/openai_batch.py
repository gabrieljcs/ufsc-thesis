from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:  # pragma: no cover
    import requests

OPENAI_API_BASE = "https://api.openai.com/v1"


def _resolve_api_key(api_key: str | None = None) -> str:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OpenAI Batch requires an API key. Set OPENAI_API_KEY or pass api_key explicitly."
        )
    return key


def _normalize_api_base(api_base: str) -> str:
    return api_base.rstrip("/")


def _json_or_raise(response: requests.Response, context: str) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:  # pragma: no cover - defensive: API should return JSON
        raise RuntimeError(
            f"{context} failed with HTTP {response.status_code}: {response.text[:500]}"
        ) from exc
    if not response.ok:
        raise RuntimeError(f"{context} failed with HTTP {response.status_code}: {payload}")
    return dict(payload)


def upload_batch_input_file(
    path: Path,
    *,
    api_key: str | None = None,
    api_base: str = OPENAI_API_BASE,
) -> dict[str, Any]:
    import requests

    key = _resolve_api_key(api_key)
    base = _normalize_api_base(api_base)
    with path.open("rb") as handle:
        response = requests.post(
            f"{base}/files",
            headers={"Authorization": f"Bearer {key}"},
            data={"purpose": "batch"},
            files={"file": (path.name, handle, "application/jsonl")},
            timeout=300,
        )
    return _json_or_raise(response, f"Upload batch input file {path}")


def create_batch(
    *,
    input_file_id: str,
    endpoint: str,
    completion_window: str = "24h",
    metadata: dict[str, str] | None = None,
    api_key: str | None = None,
    api_base: str = OPENAI_API_BASE,
) -> dict[str, Any]:
    import requests

    key = _resolve_api_key(api_key)
    base = _normalize_api_base(api_base)
    payload: dict[str, Any] = {
        "input_file_id": input_file_id,
        "endpoint": endpoint,
        "completion_window": completion_window,
    }
    if metadata:
        payload["metadata"] = metadata
    response = requests.post(
        f"{base}/batches",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=300,
    )
    return _json_or_raise(response, "Create batch")


def retrieve_batch(
    batch_id: str,
    *,
    api_key: str | None = None,
    api_base: str = OPENAI_API_BASE,
) -> dict[str, Any]:
    import requests

    key = _resolve_api_key(api_key)
    base = _normalize_api_base(api_base)
    response = requests.get(
        f"{base}/batches/{batch_id}",
        headers={"Authorization": f"Bearer {key}"},
        timeout=120,
    )
    return _json_or_raise(response, f"Retrieve batch {batch_id}")


def download_file_content(
    file_id: str,
    output_path: Path,
    *,
    api_key: str | None = None,
    api_base: str = OPENAI_API_BASE,
) -> Path:
    import requests

    key = _resolve_api_key(api_key)
    base = _normalize_api_base(api_base)
    response = requests.get(
        f"{base}/files/{file_id}/content",
        headers={"Authorization": f"Bearer {key}"},
        timeout=300,
    )
    if not response.ok:
        raise RuntimeError(
            f"Download file content {file_id} failed with HTTP {response.status_code}: "
            f"{response.text[:500]}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    return output_path
