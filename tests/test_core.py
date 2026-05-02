from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from thesis_eval.assets import AssetSpec, asset_status, load_assets, select_assets, verify_asset
from thesis_eval.benchmarks.strongreject import import_strongreject, load_prompt_records, write_prompt_records
from thesis_eval.benchmarks.uriel import load_distance_matrix, subset_matrix, write_distance_matrix
from thesis_eval.cli import main
from thesis_eval.config import load_config
from thesis_eval.evaluation.api_batches import (
    anthropic_batch_body,
    hosted_judge_rows,
    openai_batch_rows,
    parse_anthropic_batch_results,
    parse_openai_batch_results,
)
from thesis_eval.evaluation.judge_batch import build_judge_requests, combine_judge_results
from thesis_eval.evaluation.majority_vote import majority_vote, normalize_raw_label
from thesis_eval.io import write_jsonl
from thesis_eval.metrics.asr import summarize_labels
from thesis_eval.metrics.spec import add_spec_scores, z_scores
from thesis_eval.models.generation import estimate_model_asset_state, generate_outputs
from thesis_eval.runtime import load_runtime_profiles
from thesis_eval.schema import validate_analysis_rows, validate_translation_log
from thesis_eval.translation.audit import apply_audit_decisions, build_audit_queue
from thesis_eval.translation.log import TranslationLog
from thesis_eval.translation.pipeline import PromptRecord, apply_translation_qc, run_translation


class ConfigTests(unittest.TestCase):
    def test_load_config(self) -> None:
        cfg = load_config()
        self.assertEqual(len(cfg.attack_languages), 13)
        self.assertEqual(cfg.aligned_language["sabia_7b"], "por")
        self.assertIn("gpt_4o", cfg.judges)

    def test_runtime_profiles(self) -> None:
        profiles = load_runtime_profiles()
        self.assertEqual(profiles["macbook"].target_backend, "transformers")
        self.assertEqual(profiles["rtx_5070_ti"].target_backend, "vllm")

    def test_assets_manifest(self) -> None:
        assets = load_assets()
        self.assertIn("strongreject", assets)
        self.assertIn("nllb_200", assets)
        self.assertIn("sabia_7b", assets)
        self.assertEqual(select_assets(groups=["translation"])[0].group, "translation")
        self.assertEqual({asset.name for asset in select_assets(groups=["jailbreaks"])}, {"strongreject", "multijail"})
        status = asset_status(assets["nllb_200"])
        self.assertEqual(status["name"], "nllb_200")


class MajorityVoteTests(unittest.TestCase):
    def test_majority_vote_and_disagreement(self) -> None:
        label, disagreement = majority_vote(["safe", "unsafe", "safe"])
        self.assertEqual(label, "safe")
        self.assertTrue(disagreement)

    def test_requires_three_judges(self) -> None:
        with self.assertRaises(ValueError):
            majority_vote(["safe", "safe"])

    def test_normalize_raw_label(self) -> None:
        self.assertEqual(normalize_raw_label("unsafe\nbecause..."), "unsafe")


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
                    "sabia_7b",
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


class JudgeBatchTests(unittest.TestCase):
    def test_batch_round_trip_and_vote(self) -> None:
        generation = {
            "model": "sabia_7b",
            "aligned_language": "por",
            "attack_language": "ara",
            "prompt_id": "p1",
            "run_id": "r1",
            "prompt_text": "prompt",
            "model_output": "response",
            "benchmark": "strongreject",
            "distance": 0.5,
            "if_score": None,
            "cons_score": None,
            "spec_score": None,
            "input_tokens": None,
            "tokens_per_char": None,
            "token_inflation": None,
            "excluded": False,
            "exclusion_reason": None,
        }
        requests = build_judge_requests([generation], ["llama_guard_4", "gpt_4o", "claude_opus_4"])
        self.assertEqual(len(requests), 3)
        self.assertIn("api_messages", requests[0])
        results = [
            {"request_id": req["request_id"], "run_id": "r1", "judge": req["judge"], "raw_response": "safe"}
            for req in requests
        ]
        rows = combine_judge_results([generation], results)
        self.assertEqual(rows[0]["majority_label"], "safe")
        validate_analysis_rows(rows)

    def test_missing_judge_fails(self) -> None:
        generation = {"run_id": "r1", "prompt_text": "p", "model_output": "r"}
        with self.assertRaises(ValueError):
            combine_judge_results([generation], [])


class ApiBatchTests(unittest.TestCase):
    def test_provider_batch_shapes_and_result_parsing(self) -> None:
        request = {
            "request_id": "r1::gpt_4o",
            "judge": "gpt_4o",
            "run_id": "r1",
            "prompt_id": "p1",
            "attack_language": "ara",
            "api_messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "user"}],
        }
        openai = openai_batch_rows([request], "gpt-4o")
        self.assertEqual(openai[0]["url"], "/v1/chat/completions")
        parsed_openai = parse_openai_batch_results(
            [
                {
                    "custom_id": "r1::gpt_4o",
                    "response": {"body": {"choices": [{"message": {"content": "safe"}}]}},
                    "error": None,
                }
            ]
        )
        self.assertEqual(parsed_openai[0]["label"], "safe")

        anthropic = anthropic_batch_body([{**request, "request_id": "r1::claude_opus_4", "judge": "claude_opus_4"}], "claude-opus-4")
        self.assertIn("requests", anthropic)
        parsed_anthropic = parse_anthropic_batch_results(
            [
                {
                    "custom_id": "r1::claude_opus_4",
                    "result": {
                        "type": "succeeded",
                        "message": {"content": [{"type": "text", "text": "unsafe"}]},
                    },
                }
            ]
        )
        self.assertEqual(parsed_anthropic[0]["label"], "unsafe")
        hosted = hosted_judge_rows([{**request, "request_id": "r1::llama_guard_4", "judge": "llama_guard_4"}], "hosted-lg")
        self.assertEqual(hosted[0]["custom_id"], "r1::llama_guard_4")


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


if __name__ == "__main__":
    unittest.main()
