from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from thesis_eval.progress import info, step


def build_generation_rows(translations: list[dict[str, Any]], model: str, aligned_language: str, benchmark: str = "strongreject") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, translation in enumerate(translations, start=1):
        prompt_id = str(translation["prompt_id"])
        attack_language = str(translation["target_language"])
        rows.append(
            {
                "model": model,
                "aligned_language": aligned_language,
                "attack_language": attack_language,
                "prompt_id": prompt_id,
                "run_id": f"{model}:{attack_language}:{prompt_id}:pilot-{index}",
                "prompt_text": translation["translated_text"],
                "source_prompt_text": translation["source_text"],
                "model_output": "",
                "benchmark": benchmark,
            }
        )
    return rows


def generate_outputs(
    prompts: list[str],
    model_ref: str,
    backend: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    stream: bool = False,
) -> list[str]:
    if backend == "mock":
        return ["[mock target response: replace with local model output before research use.]" for _ in prompts]
    if backend == "vllm":
        return generate_with_vllm(prompts, model_ref, max_tokens=max_tokens, temperature=temperature, trust_remote_code=trust_remote_code)
    if backend == "transformers":
        return generate_with_transformers(
            prompts,
            model_ref,
            max_tokens=max_tokens,
            temperature=temperature,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            stream=stream,
        )
    if backend == "auto":
        if _cuda_available():
            try:
                return generate_with_vllm(prompts, model_ref, max_tokens=max_tokens, temperature=temperature, trust_remote_code=trust_remote_code)
            except RuntimeError:
                pass
        return generate_with_transformers(
            prompts,
            model_ref,
            max_tokens=max_tokens,
            temperature=temperature,
            device=device,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            stream=stream,
        )
    raise ValueError(f"Unknown generation backend {backend!r}")


def estimate_model_asset_state(hf_id: str, local_dir: str | None = None) -> dict[str, object]:
    if local_dir is None:
        return {
            "hf_id": hf_id,
            "local_dir": None,
            "exists": None,
            "files": None,
            "bytes": None,
            "has_config": None,
            "has_chat_template": None,
            "model_type": None,
            "architectures": None,
        }
    path = Path(local_dir)
    files = [child for child in path.rglob("*") if child.is_file()] if path.exists() else []
    config = _read_json(path / "config.json")
    tokenizer_config = _read_json(path / "tokenizer_config.json")
    return {
        "hf_id": hf_id,
        "local_dir": str(path),
        "exists": path.exists(),
        "files": len(files),
        "bytes": sum(child.stat().st_size for child in files),
        "has_config": (path / "config.json").exists(),
        "has_chat_template": bool(tokenizer_config.get("chat_template")),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
    }


def generate_with_vllm(
    prompts: list[str],
    model_ref: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    trust_remote_code: bool = False,
) -> list[str]:
    try:
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("vLLM is not installed. Use the Docker GPU profile or run with --mock-output.") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=trust_remote_code, local_files_only=_is_local_model_ref(model_ref))
    formatted_prompts = [_format_prompt(tokenizer, prompt) for prompt in prompts]
    if getattr(tokenizer, "chat_template", None):
        info("Applying tokenizer chat template before vLLM generation")
    else:
        info("Tokenizer has no chat_template; vLLM will use raw continuation prompts")
    llm = LLM(
        model=model_ref,
        trust_remote_code=trust_remote_code,
        max_model_len=1280,
        gpu_memory_utilization=0.90,
    )
    sampling = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    outputs = llm.generate(formatted_prompts, sampling)
    return [output.outputs[0].text for output in outputs]


def generate_with_transformers(
    prompts: list[str],
    model_ref: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = False,
    stream: bool = False,
) -> list[str]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
        from threading import Thread
    except ImportError as exc:
        raise RuntimeError("Transformers generation requires torch and transformers. Install the mac or rtx extras.") from exc

    resolved_device = _resolve_device(device)
    model_dtype = _resolve_dtype(dtype, resolved_device)
    tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=trust_remote_code, local_files_only=_is_local_model_ref(model_ref))
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        dtype=model_dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
        local_files_only=_is_local_model_ref(model_ref),
    )
    if resolved_device != "cpu":
        with step(f"move model to {resolved_device}"):
            model = model.to(resolved_device)
    else:
        info("Keeping model on CPU")
    model.eval()
    info(f"Model ready on device={resolved_device}; starting generation loop")

    outputs: list[str] = []
    do_sample = temperature > 0
    for index, prompt in enumerate(prompts, start=1):
        info(f"Generating prompt {index}/{len(prompts)} with max_tokens={max_tokens}")
        input_text = _format_prompt(tokenizer, prompt)
        encoded = tokenizer(input_text, return_tensors="pt")
        if resolved_device != "cpu":
            encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
        generation_kwargs = {
            **encoded,
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "temperature": temperature if do_sample else None,
            "pad_token_id": tokenizer.eos_token_id,
        }
        if stream:
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs["streamer"] = streamer
            pieces: list[str] = []

            def _run_generation() -> None:
                with torch.no_grad():
                    model.generate(**generation_kwargs)

            thread = Thread(target=_run_generation)
            thread.start()
            sys.stderr.write(f"\n[thesis-eval] STREAM prompt {index}/{len(prompts)}\n")
            sys.stderr.flush()
            for piece in streamer:
                pieces.append(str(piece))
                sys.stderr.write(str(piece))
                sys.stderr.flush()
            thread.join()
            sys.stderr.write("\n[thesis-eval] END STREAM\n")
            sys.stderr.flush()
            outputs.append("".join(pieces).strip())
        else:
            with step(f"generate prompt {index}/{len(prompts)}"):
                with torch.no_grad():
                    generated = model.generate(**generation_kwargs)
            new_tokens = generated[0][encoded["input_ids"].shape[-1] :]
            outputs.append(str(tokenizer.decode(new_tokens, skip_special_tokens=True)).strip())
    return outputs


def _format_prompt(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return str(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompt


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype: str, device: str) -> Any:
    if dtype == "auto":
        if device == "cpu":
            return None
        try:
            import torch
        except ImportError:
            return None
        return torch.float16
    try:
        import torch
    except ImportError:
        return None
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype not in mapping:
        raise ValueError(f"Unknown dtype {dtype!r}; expected auto, float16, bfloat16, or float32")
    return mapping[dtype]


def _is_local_model_ref(model_ref: str) -> bool:
    return Path(model_ref).exists()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def attach_outputs(rows: list[dict[str, Any]], outputs: list[str]) -> list[dict[str, Any]]:
    if len(rows) != len(outputs):
        raise ValueError("Generation row count and output count differ")
    enriched: list[dict[str, Any]] = []
    for row, output in zip(rows, outputs, strict=True):
        new_row = dict(row)
        new_row["model_output"] = output
        enriched.append(new_row)
    return enriched
