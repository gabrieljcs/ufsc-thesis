from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from thesis_eval.assets import AssetSpec, asset_status, download_asset, load_assets, select_assets, verify_asset
from thesis_eval.benchmarks.strongreject import import_strongreject, load_prompt_records, write_prompt_records
from thesis_eval.benchmarks.uriel import load_distance_matrix, subset_matrix, write_distance_matrix
from thesis_eval.cli import main
from thesis_eval.config import load_config
from thesis_eval.analysis.dataset import build_frozen_rows
from thesis_eval.benchmarks.belebele import compute_if_cons, parse_choice
from thesis_eval.evaluation.strongreject import score_generation_rows
from thesis_eval.io import write_jsonl
from thesis_eval.metrics.asr import summarize_labels
from thesis_eval.metrics.spec import add_spec_scores, z_scores
from thesis_eval.metrics.tokenizer import attach_tokenizer_metrics
from thesis_eval.models.generation import estimate_model_asset_state, generate_outputs
from thesis_eval.runtime import load_runtime_profiles
from thesis_eval.schema import validate_analysis_rows, validate_translation_log
from thesis_eval.translation.audit import apply_audit_decisions, build_audit_queue
from thesis_eval.translation.calibration import calibrate_blaser_thresholds
from thesis_eval.translation.log import TranslationLog
from thesis_eval.translation.pipeline import PromptRecord, apply_translation_qc, run_translation


class ConfigTests(unittest.TestCase):
    def test_load_config(self) -> None:
        cfg = load_config()
        self.assertEqual(len(cfg.models), 8)
        self.assertEqual(len(cfg.attack_languages), 13)
        self.assertEqual(cfg.aligned_language["sagui_7b"], "por")
        self.assertEqual(cfg.models["sabia_3"]["access_mode"], "api")

    def test_runtime_profiles(self) -> None:
        profiles = load_runtime_profiles()
        self.assertEqual(profiles["macbook"].target_backend, "transformers")
        self.assertEqual(profiles["rtx_5070_ti"].target_backend, "vllm")

    def test_assets_manifest(self) -> None:
        assets = load_assets()
        self.assertIn("strongreject", assets)
        self.assertIn("nllb_200", assets)
        self.assertIn("sagui_7b", assets)
        self.assertIn("sabia_3", assets)
        self.assertEqual(select_assets(groups=["translation"])[0].group, "translation")
        self.assertEqual({asset.name for asset in select_assets(groups=["jailbreaks"])}, {"strongreject", "multijail"})
        status = asset_status(assets["nllb_200"])
        self.assertEqual(status["name"], "nllb_200")

    def test_incomplete_hf_asset_resumes_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp) / "partial-model"
            local_dir.mkdir()
            asset = AssetSpec(
                name="partial",
                kind="hf_model",
                group="targets",
                groups=("targets",),
                description="partial fixture",
                repo_id="org/model",
                local_dir=local_dir,
                required_files=(local_dir / "config.json",),
            )
            with patch("thesis_eval.assets._snapshot_download") as snapshot_download:
                result = download_asset(asset)
            snapshot_download.assert_called_once()
            self.assertEqual(result["action"], "downloaded")


class MetricsTests(unittest.TestCase):
    def test_asr_summary_maps_invalid_out_of_success(self) -> None:
        summary = summarize_labels(["unsafe", "safe", "invalid", "unsafe"])
        self.assertEqual(summary["unsafe"], 2)
        self.assertEqual(summary["invalid"], 1)
        self.assertAlmostEqual(summary["asr"], 0.5)

    def test_spec_scores(self) -> None:
        rows = [
            {"if_score": 0.8, "cons_score": 0.7},
            {"if_score": 0.6, "cons_score": 0.9},
        ]
        enriched = add_spec_scores(rows)
        self.assertIn("spec_score", enriched[0])
        self.assertEqual(z_scores([1.0, 1.0]), [0.0, 0.0])

    def test_tokenizer_metrics_use_aligned_language_baseline(self) -> None:
        class FakeTokenizer:
            chat_template = "fake"

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                assert tokenize
                assert add_generation_prompt
                return [0 for _ in messages[0]["content"].split()]

        rows = [
            {
                "model": "sagui_7b",
                "benchmark": "strongreject",
                "prompt_id": "p1",
                "aligned_language": "por",
                "attack_language": "por",
                "prompt_text": "um dois",
            },
            {
                "model": "sagui_7b",
                "benchmark": "strongreject",
                "prompt_id": "p1",
                "aligned_language": "por",
                "attack_language": "ara",
                "prompt_text": "um dois tres quatro",
            },
        ]
        enriched = attach_tokenizer_metrics(rows, FakeTokenizer())
        self.assertEqual(enriched[0]["input_tokens"], 2)
        self.assertEqual(enriched[0]["token_inflation"], 1.0)
        self.assertEqual(enriched[1]["input_tokens"], 4)
        self.assertEqual(enriched[1]["token_inflation"], 2.0)


class GenerationTests(unittest.TestCase):
    def test_mock_backend_does_not_load_model(self) -> None:
        outputs = generate_outputs(["prompt"], "unused/model", backend="mock")
        self.assertEqual(len(outputs), 1)
        self.assertIn("mock target response", outputs[0])

    def test_model_status_reports_chat_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text('{"model_type":"llama","architectures":["LlamaForCausalLM"]}', encoding="utf-8")
            (path / "tokenizer_config.json").write_text('{"chat_template":"{{ messages }}"}', encoding="utf-8")
            result = estimate_model_asset_state("fixture/model", local_dir=str(path))
            self.assertTrue(result["has_chat_template"])
            self.assertEqual(result["model_type"], "llama")


class StrongRejectScoringTests(unittest.TestCase):
    def test_mock_strongreject_scoring_preserves_subscores(self) -> None:
        rows = [
            {
                "model": "sagui_7b",
                "aligned_language": "por",
                "attack_language": "eng",
                "prompt_id": "p1",
                "run_id": "r1",
                "prompt_text": "prompt",
                "source_prompt_text": "source prompt",
                "model_output": "unsafe harmful answer",
                "model_output_backtranslated": None,
                "generation_completed": True,
                "api_http_status": None,
                "api_failure": False,
                "api_failure_reason": None,
                "provider_block": False,
                "provider_block_reason": None,
                "benchmark": "strongreject",
                "distance": 0.0,
                "if_score": None,
                "cons_score": None,
                "spec_score": None,
                "input_tokens": None,
                "tokens_per_char": None,
                "token_inflation": None,
                "excluded": False,
                "exclusion_reason": None,
            }
        ]
        scored = score_generation_rows(rows, evaluator="mock")
        self.assertEqual(scored[0]["unsafe_binary"], 1)
        self.assertEqual(scored[0]["strongreject_score"], 1.0)
        self.assertEqual(scored[0]["specificity_score"], 5.0)
        self.assertFalse(scored[0]["scoring_failed"])

    def test_scoring_failure_does_not_become_safe(self) -> None:
        row = {
            "model": "sabia_3",
            "aligned_language": "por",
            "attack_language": "ara",
            "prompt_id": "p1",
            "run_id": "r1",
            "prompt_text": "prompt",
            "source_prompt_text": "source prompt",
            "model_output": "",
            "generation_completed": False,
            "api_failure": True,
            "api_failure_reason": "timeout",
            "provider_block": False,
            "provider_block_reason": None,
            "benchmark": "strongreject",
        }
        scored = score_generation_rows([row], evaluator="mock")
        self.assertIsNone(scored[0]["unsafe_binary"])
        self.assertTrue(scored[0]["scoring_failed"])
        self.assertTrue(scored[0]["excluded"])


class DatasetBuilderTests(unittest.TestCase):
    def test_build_frozen_rows_from_scored_generation(self) -> None:
        cfg = load_config()
        scored = score_generation_rows(
            [
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "por",
                    "prompt_id": "p1",
                    "run_id": "r1",
                    "prompt_text": "prompt",
                    "source_prompt_text": "source",
                    "model_output": "safe refusal",
                    "model_output_backtranslated": None,
                    "generation_completed": True,
                    "api_http_status": None,
                    "api_failure": False,
                    "api_failure_reason": None,
                    "provider_block": False,
                    "provider_block_reason": None,
                    "benchmark": "strongreject",
                    "distance": 0.0,
                    "input_tokens": 10,
                    "tokens_per_char": 1.0,
                    "token_inflation": 1.0,
                    "excluded": False,
                    "exclusion_reason": None,
                }
            ],
            evaluator="mock",
        )
        translations = [
            TranslationLog(
                prompt_id="p1",
                source_language="eng",
                target_language="por",
                harmful_goal="goal",
                expected_output_form="other",
                fixed_constraints="none",
                allowed_adaptation="fluency",
                translator_config={"engine": "placeholder"},
                blaser_qe_score=None,
                blaser_status="pending",
                human_audit_status="not_audited",
                reference_subset_score=None,
                harm_preservation="pending",
                revision_note=None,
                source_text="source",
                translated_text="prompt",
            ).to_dict()
        ]
        rows = build_frozen_rows(scored, cfg, translations=translations)
        validate_analysis_rows(rows)
        self.assertEqual(rows[0]["model_alignment_pole"], "weak")
        self.assertEqual(rows[0]["translation_blaser_status"], "pending")


class BelebeleTests(unittest.TestCase):
    def test_parse_choice_and_compute_spec(self) -> None:
        self.assertEqual(parse_choice("Answer: B"), "B")
        predictions = [
            {"model": "sagui_7b", "language": "por", "item_id": "i1", "prediction": "A", "gold": "A"},
            {"model": "sagui_7b", "language": "ara", "item_id": "i1", "prediction": "B", "gold": "A"},
            {"model": "sagui_7b", "language": "por", "item_id": "i2", "prediction": "C", "gold": "C"},
            {"model": "sagui_7b", "language": "ara", "item_id": "i2", "prediction": "C", "gold": "C"},
        ]
        rows = compute_if_cons(predictions, {"sagui_7b": "por"})
        by_lang = {row["attack_language"]: row for row in rows}
        self.assertEqual(by_lang["por"]["if_score"], 1.0)
        self.assertEqual(by_lang["ara"]["cons_score"], 0.5)
        self.assertIn("spec_score", by_lang["ara"])


class UrielTests(unittest.TestCase):
    def test_matrix_round_trip_and_subset(self) -> None:
        matrix = {
            "por": {"por": 0.0, "ara": 0.5},
            "ara": {"por": 0.5, "ara": 0.0},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "distance.csv"
            write_distance_matrix(path, matrix, ["por", "ara"])
            loaded = load_distance_matrix(path)
            self.assertEqual(loaded["por"]["ara"], 0.5)
            self.assertEqual(set(subset_matrix(loaded, ["ara", "por"])), {"ara", "por"})


class StrongRejectImportTests(unittest.TestCase):
    def test_import_local_csv_and_pilot_records(self) -> None:
        csv_text = "category,source,forbidden_prompt\ncat,src,benign fixture prompt\n"
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "strongreject.csv"
            out_path = Path(tmp) / "prompts.jsonl"
            csv_path.write_text(csv_text, encoding="utf-8")
            records = import_strongreject(source="local", input_path=csv_path)
            self.assertEqual(records[0]["prompt_id"], "strongreject_0001")
            self.assertEqual(records[0]["text"], "benign fixture prompt")
            write_prompt_records(out_path, records)
            loaded = load_prompt_records(out_path)
            self.assertEqual(loaded[0]["benchmark"], "strongreject")

    def test_verify_strongreject_asset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "prompts.jsonl"
            pilot = Path(tmp) / "pilot_prompts.jsonl"
            records = [
                {
                    "prompt_id": "strongreject_0001",
                    "text": "benign fixture prompt",
                    "harmful_goal": "pending",
                    "expected_output_form": "other",
                    "fixed_constraints": "pending",
                    "allowed_adaptation": "fluency",
                    "benchmark": "strongreject",
                }
            ]
            write_prompt_records(output, records)
            write_prompt_records(pilot, records)
            spec = AssetSpec(
                name="strongreject_fixture",
                kind="strongreject",
                group="datasets",
                groups=("datasets", "jailbreaks"),
                description="fixture",
                output=output,
                pilot_output=pilot,
                required_files=(output, pilot),
                required_any_files=((output,),),
                required_columns=("prompt_id", "text", "benchmark"),
            )
            result = verify_asset(spec)
            self.assertTrue(result["ok"])


class TranslationTests(unittest.TestCase):
    def test_translation_log_is_complete(self) -> None:
        prompt = PromptRecord(prompt_id="p1", text="placeholder")
        logs = run_translation([prompt], "por", engine="placeholder")
        record = logs[0].to_dict()
        validate_translation_log(record)
        self.assertEqual(record["target_language"], "por")

    def test_translation_qc_sets_blaser_fields(self) -> None:
        log = TranslationLog(
            prompt_id="p1",
            source_language="eng",
            target_language="por",
            harmful_goal="goal",
            expected_output_form="other",
            fixed_constraints="none",
            allowed_adaptation="fluency",
            translator_config={"engine": "placeholder"},
            blaser_qe_score=None,
            blaser_status="pending",
            human_audit_status="not_audited",
            reference_subset_score=None,
            harm_preservation="pending",
            revision_note=None,
            source_text="same text",
            translated_text="same text",
        ).to_dict()
        qc = apply_translation_qc([log])
        self.assertEqual(qc[0]["blaser_status"], "pass")
        self.assertEqual(qc[0]["harm_preservation"], "preserved")

    def test_translation_qc_accepts_blaser_scores(self) -> None:
        log = TranslationLog(
            prompt_id="p1",
            source_language="eng",
            target_language="por",
            harmful_goal="goal",
            expected_output_form="other",
            fixed_constraints="none",
            allowed_adaptation="fluency",
            translator_config={"engine": "placeholder"},
            blaser_qe_score=None,
            blaser_status="pending",
            human_audit_status="not_audited",
            reference_subset_score=None,
            harm_preservation="pending",
            revision_note=None,
            source_text="source",
            translated_text="translation",
        ).to_dict()
        with patch("thesis_eval.translation.blaser.score_blaser_qe", return_value=[4.2]):
            qc = apply_translation_qc([log], scorer="blaser", pass_threshold=4.0, fail_threshold=3.0)
        self.assertEqual(qc[0]["blaser_qe_score"], 4.2)
        self.assertEqual(qc[0]["blaser_status"], "pass")

    def test_unknown_qc_scorer_fails(self) -> None:
        with self.assertRaises(ValueError):
            apply_translation_qc([], scorer="not-real")

    def test_cli_translate_writes_multiple_language_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            prompts = tmp_path / "prompts.jsonl"
            outputs = tmp_path / "translations"
            prompts.write_text('{"prompt_id":"p1","text":"hello"}\n', encoding="utf-8")
            main(
                [
                    "translate",
                    "--prompts",
                    str(prompts),
                    "--target-language",
                    "por",
                    "--target-language",
                    "ara",
                    "--engine",
                    "placeholder",
                    "--output-dir",
                    str(outputs),
                    "--output-template",
                    "{lang}.jsonl",
                ]
            )
            self.assertTrue((outputs / "por.jsonl").exists())
            self.assertTrue((outputs / "ara.jsonl").exists())

    def test_cli_qc_writes_multiple_language_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            por = tmp_path / "por.jsonl"
            ara = tmp_path / "ara.jsonl"
            outputs = tmp_path / "qc"
            write_jsonl(por, [self._translation_log("p1", "por")])
            write_jsonl(ara, [self._translation_log("p1", "ara")])
            main(
                [
                    "translation-qc",
                    "--translations",
                    str(por),
                    "--translations",
                    str(ara),
                    "--scorer",
                    "heuristic",
                    "--output-dir",
                    str(outputs),
                    "--output-template",
                    "{lang}.qc.jsonl",
                ]
            )
            self.assertTrue((outputs / "por.qc.jsonl").exists())
            self.assertTrue((outputs / "ara.qc.jsonl").exists())

    def test_cli_generate_targets_batches_multiple_translation_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            por = tmp_path / "por.jsonl"
            ara = tmp_path / "ara.jsonl"
            output = tmp_path / "generations.jsonl"
            write_jsonl(por, [self._translation_log("p1", "por")])
            write_jsonl(ara, [self._translation_log("p1", "ara")])
            main(
                [
                    "generate-targets",
                    "--translations",
                    str(por),
                    "--translations",
                    str(ara),
                    "--model",
                    "sagui_7b",
                    "--backend",
                    "mock",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(len(output.read_text(encoding="utf-8").strip().splitlines()), 2)

    def _translation_log(self, prompt_id: str, target_language: str) -> dict[str, object]:
        return TranslationLog(
            prompt_id=prompt_id,
            source_language="eng",
            target_language=target_language,
            harmful_goal="goal",
            expected_output_form="other",
            fixed_constraints="none",
            allowed_adaptation="fluency",
            translator_config={"engine": "placeholder"},
            blaser_qe_score=None,
            blaser_status="pending",
            human_audit_status="not_audited",
            reference_subset_score=None,
            harm_preservation="pending",
            revision_note=None,
            source_text="same text",
            translated_text="same text",
        ).to_dict()


class AuditTests(unittest.TestCase):
    def test_audit_queue_and_import(self) -> None:
        log = TranslationLog(
            prompt_id="p1",
            source_language="eng",
            target_language="ara",
            harmful_goal="goal",
            expected_output_form="other",
            fixed_constraints="none",
            allowed_adaptation="fluency",
            translator_config={"engine": "placeholder"},
            blaser_qe_score=0.5,
            blaser_status="flag",
            human_audit_status="not_audited",
            reference_subset_score=None,
            harm_preservation="pending",
            revision_note=None,
            source_text="source",
            translated_text="translation",
        ).to_dict()
        queue = build_audit_queue([log])
        self.assertEqual(len(queue), 1)
        updated = apply_audit_decisions(
            [log],
            [
                {
                    "prompt_id": "p1",
                    "target_language": "ara",
                    "audit_decision": "pass",
                    "harm_preservation": "preserved",
                }
            ],
        )
        self.assertEqual(updated[0]["human_audit_status"], "audited_pass")
        self.assertEqual(updated[0]["harm_preservation"], "preserved")

    def test_blaser_calibration(self) -> None:
        result = calibrate_blaser_thresholds(
            [
                {"blaser_qe_score": "4.2", "audit_decision": "pass"},
                {"blaser_qe_score": "3.1", "audit_decision": "revise"},
            ]
        )
        self.assertEqual(result["audited_pass"], 1)
        self.assertLessEqual(result["fail_threshold"], result["pass_threshold"])


if __name__ == "__main__":
    unittest.main()
