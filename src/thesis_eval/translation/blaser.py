from __future__ import annotations

from dataclasses import dataclass

from thesis_eval.progress import info, step
from thesis_eval.translation.pipeline import NLLB_CODES


@dataclass
class BlaserScorer:
    device: str | None = None

    def __post_init__(self) -> None:
        try:
            import torch
            from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
            from sonar.models.blaser.loader import load_blaser_model
        except ImportError as exc:
            raise RuntimeError(
                "BLASER QC requires SONAR + fairseq2 + PyTorch. Use Python 3.12, then run "
                "`uv sync --extra qc`. On macOS, install libsndfile first if fairseq2 complains: "
                "`brew install libsndfile`."
            ) from exc

        self._torch = torch
        with step("load BLASER 2.0 QE model"):
            self._blaser = load_blaser_model("blaser_2_0_qe").eval()
        info("Loading SONAR text encoder/tokenizer. First run may download fairseq2/SONAR assets.")
        with step("load SONAR text encoder"):
            self._embedder = TextToEmbeddingModelPipeline(
                encoder="text_sonar_basic_encoder",
                tokenizer="text_sonar_basic_encoder",
            )
        if self.device:
            self._blaser = self._blaser.to(self.device)

    def score(self, source_texts: list[str], translated_texts: list[str], target_language: str) -> list[float]:
        if len(source_texts) != len(translated_texts):
            raise ValueError("source_texts and translated_texts must have the same length")
        source_lang = NLLB_CODES["eng"]
        target_lang = NLLB_CODES[target_language]
        with step(f"embed {len(source_texts)} source texts ({source_lang})"):
            src_embs = self._embedder.predict(source_texts, source_lang=source_lang)
        with step(f"embed {len(translated_texts)} translated texts ({target_lang})"):
            mt_embs = self._embedder.predict(translated_texts, source_lang=target_lang)
        if self.device:
            src_embs = src_embs.to(self.device)
            mt_embs = mt_embs.to(self.device)
        with step("score translations with BLASER"):
            with self._torch.inference_mode():
                scores = self._blaser(src=src_embs, mt=mt_embs)
        return [float(score) for score in scores.detach().cpu().reshape(-1)]


def score_blaser_qe(logs: list[dict[str, object]], device: str | None = None) -> list[float]:
    if not logs:
        return []
    scorer = BlaserScorer(device=device)
    scores: list[float] = []
    by_language: dict[str, list[dict[str, object]]] = {}
    for log in logs:
        by_language.setdefault(str(log["target_language"]), []).append(log)
    score_by_key: dict[tuple[str, str], float] = {}
    for target_language, rows in by_language.items():
        source_texts = [str(row["source_text"]) for row in rows]
        translated_texts = [str(row["translated_text"]) for row in rows]
        language_scores = scorer.score(source_texts, translated_texts, target_language=target_language)
        for row, score in zip(rows, language_scores, strict=True):
            score_by_key[(str(row["prompt_id"]), str(row["target_language"]))] = score
    for log in logs:
        scores.append(score_by_key[(str(log["prompt_id"]), str(log["target_language"]))])
    return scores
