from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from thesis_eval.paths import CONFIG_DIR, read_yaml


@dataclass(frozen=True)
class ConfigBundle:
    attack_languages: list[str]
    aligned_language: dict[str, str]
    models: dict[str, dict[str, str]]
    judges: dict[str, dict[str, str]]


def load_config(config_dir: Path = CONFIG_DIR) -> ConfigBundle:
    languages = read_yaml(config_dir / "languages.yaml")
    models = read_yaml(config_dir / "models.yaml")
    judges = read_yaml(config_dir / "judges.yaml")

    attack_languages = list(languages.get("attack_languages", []))
    aligned_language = dict(languages.get("aligned_language", {}))
    if not attack_languages:
        raise ValueError("configs/languages.yaml must define attack_languages")
    if len(set(attack_languages)) != len(attack_languages):
        raise ValueError("attack_languages contains duplicates")
    if not models:
        raise ValueError("configs/models.yaml must define at least one model")
    if set(models) != set(aligned_language):
        raise ValueError("Model keys must match aligned_language keys")
    for model_name, model_cfg in models.items():
        model_lang = model_cfg.get("aligned_language")
        if model_lang != aligned_language[model_name]:
            raise ValueError(f"{model_name} aligned_language mismatch")
        if model_lang not in attack_languages:
            raise ValueError(f"{model_name} aligned language {model_lang} missing from inventory")
    for judge_name in ("llama_guard_4", "gpt_4o", "claude_opus_4"):
        if judge_name not in judges:
            raise ValueError(f"Missing required judge config: {judge_name}")
    return ConfigBundle(
        attack_languages=attack_languages,
        aligned_language=aligned_language,
        models=models,
        judges=judges,
    )
