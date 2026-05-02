from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from thesis_eval.progress import info, step
from thesis_eval.translation.log import TranslationLog


NLLB_CODES = {
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


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    text: str
    harmful_goal: str = "pending_manual_intent_spec"
    expected_output_form: str = "other"
    fixed_constraints: str = "pending_manual_intent_spec"
    allowed_adaptation: str = "fluency_only_no_semantic_change"


def translate_text(text: str, target_language: str, engine: str = "placeholder", checkpoint: str = "facebook/nllb-200-distilled-600M") -> str:
    if engine == "placeholder":
        return text
    if engine != "nllb":
        raise ValueError(f"Unknown translation engine: {engine}")
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("NLLB translation requires transformers. Install the GPU/translation dependencies first.") from exc
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, src_lang=NLLB_CODES["eng"])
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    target_code = NLLB_CODES[target_language]
    encoded = tokenizer(text, return_tensors="pt")
    generated = model.generate(**encoded, forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_code))
    return str(tokenizer.batch_decode(generated, skip_special_tokens=True)[0])


def make_translation_log(prompt: PromptRecord, target_language: str, translated_text: str, engine: str, checkpoint: str) -> TranslationLog:
    return TranslationLog(
        prompt_id=prompt.prompt_id,
        source_language="eng",
        target_language=target_language,
        harmful_goal=prompt.harmful_goal,
        expected_output_form=prompt.expected_output_form,
        fixed_constraints=prompt.fixed_constraints,
        allowed_adaptation=prompt.allowed_adaptation,
        translator_config={
            "engine": engine,
            "checkpoint": checkpoint,
            "source_code": NLLB_CODES["eng"],
            "target_code": NLLB_CODES[target_language],
        },
        blaser_qe_score=None,
        blaser_status="pending",
        human_audit_status="not_audited",
        reference_subset_score=None,
        harm_preservation="pending",
        revision_note=None,
        source_text=prompt.text,
        translated_text=translated_text,
    )


def run_translation(prompts: list[PromptRecord], target_language: str, engine: str = "placeholder", checkpoint: str = "facebook/nllb-200-distilled-600M") -> list[TranslationLog]:
    logs: list[TranslationLog] = []
    info(f"Translating {len(prompts)} prompts to {target_language} with engine={engine}")
    for index, prompt in enumerate(prompts, start=1):
        with step(f"translate {index}/{len(prompts)} prompt_id={prompt.prompt_id} target={target_language}"):
            translated = prompt.text if target_language == "eng" else translate_text(prompt.text, target_language, engine, checkpoint)
            logs.append(make_translation_log(prompt, target_language, translated, engine, checkpoint))
    return logs


def heuristic_blaser_score(source_text: str, translated_text: str) -> float:
    if not source_text.strip() or not translated_text.strip():
        return 0.0
    ratio = len(translated_text) / max(len(source_text), 1)
    length_score = max(0.0, 1.0 - abs(1.0 - ratio))
    return round(min(1.0, max(0.0, 0.55 + 0.45 * length_score)), 4)


def apply_translation_qc(
    logs: list[dict[str, Any]],
    pass_threshold: float = 0.72,
    fail_threshold: float = 0.55,
    scorer: str = "heuristic",
    device: str | None = None,
) -> list[dict[str, Any]]:
    if scorer == "heuristic":
        scores = [heuristic_blaser_score(str(log["source_text"]), str(log["translated_text"])) for log in logs]
    elif scorer == "blaser":
        from thesis_eval.translation.blaser import score_blaser_qe

        scores = score_blaser_qe(logs, device=device)
    else:
        raise ValueError(f"Unknown translation QC scorer {scorer!r}")
    output: list[dict[str, Any]] = []
    for log, score in zip(logs, scores, strict=True):
        row = dict(log)
        row["blaser_qe_score"] = round(float(score), 4)
        if score >= pass_threshold:
            row["blaser_status"] = "pass"
        elif score < fail_threshold:
            row["blaser_status"] = "fail"
        else:
            row["blaser_status"] = "flag"
        row["harm_preservation"] = "preserved" if row["blaser_status"] == "pass" else "pending"
        output.append(row)
    return output
