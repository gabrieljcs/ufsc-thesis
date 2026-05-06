from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from thesis_eval.assets import AssetSpec, asset_status, download_asset, load_assets, select_assets, verify_asset
from thesis_eval.benchmarks.strongreject import import_strongreject, load_prompt_records, write_prompt_records
from thesis_eval.benchmarks.uriel import (
    compute_urielplus_matrix,
    compute_urielplus_matrix_uv,
    load_distance_matrix,
    subset_matrix,
    write_distance_matrix,
)
from thesis_eval.cli import main
from thesis_eval.config import load_config
from thesis_eval.analysis.dataset import build_frozen_rows
from thesis_eval.analysis.reporting import (
    counterfactual_safety_by_aligned_language,
    prereg_distance_slope_retention,
    prereg_falsification_summary,
    reference_distance_curve,
)
from thesis_eval.benchmarks.belebele import (
    build_belebele_prediction_rows,
    build_belebele_prompt,
    compute_if_cons,
    load_belebele_rows,
    parse_choice,
    parse_choice_from_options,
    repair_belebele_predictions,
    resolve_belebele_language_code,
)
from thesis_eval.evaluation.strongreject import parse_strongreject_rubric_judge_response, score_generation_rows
from thesis_eval.io import read_jsonl, write_jsonl
from thesis_eval.metrics.asr import summarize_labels
from thesis_eval.metrics.spec import add_spec_scores, z_scores
from thesis_eval.metrics.tokenizer import attach_tokenizer_metrics
from thesis_eval.models.generation import (
    _maybe_wrap_vllm_attention_backend_error,
    _model_attention_softcap,
    _parse_vllm_estimated_max_model_len,
    _retry_vllm_max_model_len_from_error,
    _select_vllm_attention_backend_override,
    build_generation_runner,
    build_prompt_messages,
    describe_prompt_format,
    estimate_model_asset_state,
    format_generation_prompt,
    generate_outputs,
)
from thesis_eval.openai_batch import retrieve_batch
from thesis_eval.runtime import load_runtime_profiles
from thesis_eval.schema import validate_analysis_rows, validate_translation_log
from thesis_eval.translation.audit import apply_audit_decisions, build_audit_queue
from thesis_eval.translation.calibration import calibrate_blaser_thresholds
from thesis_eval.translation.llm_audit import build_xsts_prompt, judge_audit_rows
from thesis_eval.translation.log import TranslationLog
from thesis_eval.translation.pipeline import PromptRecord, apply_translation_qc, run_translation


class ConfigTests(unittest.TestCase):
    def test_load_config(self) -> None:
        cfg = load_config()
        self.assertEqual(len(cfg.models), 9)
        self.assertEqual(len(cfg.attack_languages), 13)
        self.assertEqual(cfg.aligned_language["sagui_7b"], "por")
        self.assertEqual(cfg.aligned_language["llamantino_2_ultrachat_7b"], "ita")
        self.assertEqual(cfg.models["sabia_3"]["access_mode"], "api")
        self.assertEqual(cfg.models["llama3_1_8b_reference"]["alignment_pole"], "reference")
        self.assertEqual(cfg.models["llama3_1_8b_reference"]["analysis_role"], "reference_baseline")

    def test_runtime_profiles(self) -> None:
        profiles = load_runtime_profiles()
        self.assertEqual(profiles["macbook"].target_backend, "transformers")
        self.assertEqual(profiles["rtx_5070_ti"].target_backend, "vllm")
        self.assertEqual(profiles["runpod_a100"].target_backend, "vllm")
        self.assertEqual(profiles["runpod_a100"].device, "cuda")
        self.assertEqual(profiles["tpu"].device, "xla")
        self.assertEqual(profiles["trc_tpu"].device, "xla")

    def test_assets_manifest(self) -> None:
        assets = load_assets()
        self.assertIn("strongreject", assets)
        self.assertIn("nllb_200", assets)
        self.assertIn("sagui_7b", assets)
        self.assertIn("sabia_3", assets)
        self.assertIn("llamantino_2_ultrachat_7b", assets)
        self.assertIn("llama3_1_8b_reference", assets)
        self.assertEqual(select_assets(groups=["translation"])[0].group, "translation")
        self.assertEqual({asset.name for asset in select_assets(groups=["jailbreaks"])}, {"strongreject", "multijail"})
        self.assertEqual({asset.name for asset in select_assets(groups=["reference_targets"])}, {"llama3_1_8b_reference"})
        status = asset_status(assets["nllb_200"])
        self.assertEqual(status["name"], "nllb_200")

    def test_incomplete_hf_asset_resumes_download(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp) / "partial-model"
            local_dir.mkdir()
            (local_dir / ".cache" / "huggingface").mkdir(parents=True)
            (local_dir / "config.json").write_text("{}", encoding="utf-8")
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

    def test_sharded_hf_asset_requires_all_indexed_shards(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = Path(tmp) / "partial-sharded-model"
            (local_dir / ".cache" / "huggingface").mkdir(parents=True)
            (local_dir / "config.json").write_text("{}", encoding="utf-8")
            (local_dir / "model.safetensors.index.json").write_text(
                '{"weight_map":{"a":"model-00001-of-00002.safetensors","b":"model-00002-of-00002.safetensors"}}',
                encoding="utf-8",
            )
            (local_dir / "model-00001-of-00002.safetensors").write_text("partial", encoding="utf-8")
            asset = AssetSpec(
                name="partial_sharded",
                kind="hf_model",
                group="targets",
                groups=("targets",),
                description="partial sharded fixture",
                repo_id="org/model",
                local_dir=local_dir,
                required_files=(local_dir / "config.json",),
                required_any_files=((local_dir / "model.safetensors.index.json",),),
            )
            result = verify_asset(asset)
            self.assertFalse(result["ok"])
            weights_check = next(check for check in result["checks"] if check["name"] == "model weights")
            self.assertEqual(weights_check["missing_shard_count"], 1)


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

    def test_attach_tokenizer_metrics_skips_api_model_without_tokenizer_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            generations = tmp_path / "sabia_rows.jsonl"
            output = tmp_path / "sabia_rows.with_metrics.jsonl"
            write_jsonl(
                generations,
                [
                    {
                        "model": "sabia_3",
                        "benchmark": "strongreject",
                        "prompt_id": "p1",
                        "aligned_language": "por",
                        "attack_language": "ara",
                        "prompt_text": "prompt",
                        "model_output": "response",
                    }
                ],
            )
            main(
                [
                    "attach-tokenizer-metrics",
                    "--generations",
                    str(generations),
                    "--output",
                    str(output),
                ]
            )
            rows = read_jsonl(output)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["model"], "sabia_3")
            self.assertNotIn("input_tokens", rows[0])

    def test_tokenizer_metrics_use_manual_prompt_fallback_without_chat_template(self) -> None:
        class FakeTokenizer:
            name_or_path = "swap-uniba/LLaMAntino-2-chat-7b-hf-UltraChat-ITA"
            chat_template = None

            def __call__(self, text, add_special_tokens):
                self.last_text = text
                self.last_add_special_tokens = add_special_tokens
                return {"input_ids": [0 for _ in text.split()]}

        tokenizer = FakeTokenizer()
        rows = [
            {
                "model": "llamantino_2_ultrachat_7b",
                "benchmark": "strongreject",
                "prompt_id": "p1",
                "aligned_language": "ita",
                "attack_language": "ita",
                "prompt_text": "Ciao mondo",
            }
        ]
        attach_tokenizer_metrics(rows, tokenizer)
        self.assertTrue(tokenizer.last_add_special_tokens)
        self.assertIn("[INST]", tokenizer.last_text)
        self.assertIn("Sei un assistente disponibile", tokenizer.last_text)

    def test_prereg_distance_slope_retention(self) -> None:
        rows = [
            {
                "model": "llamantino_2_ultrachat_7b",
                "model_pair_language": "ita",
                "model_alignment_pole": "weak",
                "distance": distance,
                "unsafe_binary": unsafe,
                "excluded": False,
            }
            for distance, unsafe in [(0.0, 0), (0.0, 0), (1.0, 1), (1.0, 1)]
        ] + [
            {
                "model": "llamantino_anita_8b",
                "model_pair_language": "ita",
                "model_alignment_pole": "strong",
                "distance": distance,
                "unsafe_binary": unsafe,
                "excluded": False,
            }
            for distance, unsafe in [(0.0, 0), (0.0, 0), (1.0, 1), (1.0, 1)]
        ]
        pair_rows = prereg_distance_slope_retention(rows)
        self.assertTrue(pair_rows[0]["slope_sign_retained"])
        summary = prereg_falsification_summary(rows)
        self.assertEqual(summary[0]["pairs_with_slope_sign_retained"], 1)

    def test_reference_baseline_descriptive_tables(self) -> None:
        rows = [
            {
                "model": "llama3_1_8b_reference",
                "model_alignment_pole": "reference",
                "aligned_language": "eng",
                "attack_language": "por",
                "unsafe_binary": unsafe,
                "strongreject_score": float(unsafe),
                "refusal_binary": 1 - unsafe,
                "distance": 0.25,
                "excluded": False,
            }
            for unsafe in [1, 0]
        ] + [
            {
                "model": "sagui_7b",
                "model_alignment_pole": "weak",
                "aligned_language": "por",
                "attack_language": "por",
                "unsafe_binary": unsafe,
                "strongreject_score": float(unsafe),
                "refusal_binary": 1 - unsafe,
                "distance": 0.0,
                "excluded": False,
            }
            for unsafe in [0, 0]
        ]
        curve = reference_distance_curve(rows)
        self.assertEqual(curve[0]["attack_language"], "por")
        self.assertEqual(curve[0]["distance_from_english"], 0.25)
        safety = counterfactual_safety_by_aligned_language(rows)
        self.assertEqual(safety[0]["model"], "sagui_7b")
        self.assertEqual(safety[0]["asr_gap_model_minus_reference"], -0.5)


class GenerationTests(unittest.TestCase):
    def test_mock_backend_does_not_load_model(self) -> None:
        outputs = generate_outputs(["prompt"], "unused/model", backend="mock")
        self.assertEqual(len(outputs), 1)
        self.assertIn("mock target response", outputs[0])

    def test_build_generation_runner_mock_is_reusable(self) -> None:
        runner = build_generation_runner("unused/model", backend="mock")
        self.assertEqual(runner(["one"]), ["[mock target response: replace with local model output before research use.]"])
        self.assertEqual(len(runner(["one", "two"])), 2)

    def test_local_generation_batch_size_defaults_by_backend(self) -> None:
        from thesis_eval.cli import _resolve_local_generation_batch_size

        self.assertEqual(_resolve_local_generation_batch_size(None, "vllm"), 8)
        self.assertEqual(_resolve_local_generation_batch_size(None, "transformers"), 32)
        self.assertEqual(_resolve_local_generation_batch_size(3, "vllm"), 3)

    def test_parse_vllm_estimated_max_model_len(self) -> None:
        message = "Based on the available memory, the estimated maximum model length is 1184."
        self.assertEqual(_parse_vllm_estimated_max_model_len(message), 1184)
        self.assertIsNone(_parse_vllm_estimated_max_model_len("some other failure"))

    def test_retry_vllm_max_model_len_from_error(self) -> None:
        error = RuntimeError("Based on the available memory, the estimated maximum model length is 1184.")
        retry_len = _retry_vllm_max_model_len_from_error(
            error,
            longest_prompt_tokens=744,
            requested_max_model_len=1256,
        )
        self.assertEqual(retry_len, 1168)

    def test_retry_vllm_max_model_len_from_nested_error_message(self) -> None:
        inner = ValueError("Based on the available memory, the estimated maximum model length is 1104.")
        outer = RuntimeError("Engine core initialization failed. See root cause above. Failed core proc(s): {}")
        outer.__cause__ = inner
        retry_len = _retry_vllm_max_model_len_from_error(
            outer,
            longest_prompt_tokens=744,
            requested_max_model_len=1256,
        )
        self.assertEqual(retry_len, 1088)

    def test_retry_vllm_max_model_len_from_generic_engine_failure(self) -> None:
        error = RuntimeError("Engine core initialization failed. See root cause above. Failed core proc(s): {}")
        retry_len = _retry_vllm_max_model_len_from_error(
            error,
            longest_prompt_tokens=744,
            requested_max_model_len=1256,
        )
        self.assertEqual(retry_len, 1099)

    def test_retry_vllm_max_model_len_raises_if_prompt_alone_does_not_fit(self) -> None:
        error = RuntimeError("Based on the available memory, the estimated maximum model length is 1184.")
        with self.assertRaisesRegex(RuntimeError, "cannot fit even the prompt alone"):
            _retry_vllm_max_model_len_from_error(
                error,
                longest_prompt_tokens=1184,
                requested_max_model_len=1256,
            )

    def test_model_attention_softcap_reads_nested_text_config(self) -> None:
        cfg = SimpleNamespace(text_config=SimpleNamespace(attn_logit_softcapping=50.0))
        self.assertEqual(_model_attention_softcap(cfg), 50.0)

    def test_select_vllm_attention_backend_override_prefers_flashinfer(self) -> None:
        cfg = SimpleNamespace(attn_logit_softcapping=50.0)
        with patch("thesis_eval.models.generation._flashinfer_available", return_value=True):
            backend, note = _select_vllm_attention_backend_override(cfg, explicit_backend=None)
        self.assertEqual(backend, "FLASHINFER")
        self.assertIsNotNone(note)
        self.assertIn("forcing VLLM_ATTENTION_BACKEND=FLASHINFER", str(note))

    def test_select_vllm_attention_backend_override_respects_explicit_env(self) -> None:
        cfg = SimpleNamespace(attn_logit_softcapping=50.0)
        with patch("thesis_eval.models.generation._flashinfer_available", return_value=True):
            backend, note = _select_vllm_attention_backend_override(cfg, explicit_backend="FLASH_ATTN")
        self.assertIsNone(backend)
        self.assertIsNotNone(note)
        self.assertIn("respecting pre-set VLLM_ATTENTION_BACKEND=FLASH_ATTN", str(note))

    def test_wrap_vllm_attention_backend_error_adds_actionable_guidance(self) -> None:
        error = RuntimeError("This flash attention build does not support tanh softcapping.")
        with patch("thesis_eval.models.generation._flashinfer_available", return_value=True):
            wrapped = _maybe_wrap_vllm_attention_backend_error(
                error,
                explicit_backend="FLASH_ATTN",
                auto_backend_override=None,
            )
        self.assertIsNotNone(wrapped)
        self.assertIn("FLASHINFER", str(wrapped))
        self.assertIn("transformers backend", str(wrapped))

    def test_model_status_reports_chat_template(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text('{"model_type":"llama","architectures":["LlamaForCausalLM"]}', encoding="utf-8")
            (path / "tokenizer_config.json").write_text('{"chat_template":"{{ messages }}"}', encoding="utf-8")
            result = estimate_model_asset_state("fixture/model", local_dir=str(path))
            self.assertTrue(result["has_chat_template"])
            self.assertEqual(result["model_type"], "llama")

    def test_generation_rows_skip_translation_exclusions(self) -> None:
        from thesis_eval.models.generation import build_generation_rows

        translations = [
            {
                "prompt_id": "p1",
                "target_language": "por",
                "translated_text": "keep",
                "source_text": "source",
                "translation_excluded": False,
            },
            {
                "prompt_id": "p2",
                "target_language": "por",
                "translated_text": "drop",
                "source_text": "source",
                "translation_excluded": True,
            },
        ]
        rows = build_generation_rows(translations, model="sagui_7b", aligned_language="por")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prompt_id"], "p1")

    def test_format_generation_prompt_wraps_llamantino2_without_chat_template(self) -> None:
        class FakeTokenizer:
            name_or_path = "swap-uniba/LLaMAntino-2-chat-7b-hf-UltraChat-ITA"
            chat_template = None

        formatted = format_generation_prompt(FakeTokenizer(), "Ciao?")
        self.assertTrue(formatted.startswith("[INST]"))
        self.assertIn("Sei un assistente disponibile", formatted)
        self.assertTrue(formatted.endswith("Ciao? [/INST]"))

    def test_format_generation_prompt_uses_native_sagui_user_only_chat_template(self) -> None:
        class FakeTokenizer:
            name_or_path = "OliveiraJLT/Sagui-7B-Instruct-v0.1"
            chat_template = "{{ messages }}"

            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                self.messages = messages
                self.tokenize = tokenize
                self.add_generation_prompt = add_generation_prompt
                return "FORMATTED"

        tokenizer = FakeTokenizer()
        formatted = format_generation_prompt(tokenizer, "Ola?")
        self.assertEqual(formatted, "FORMATTED")
        self.assertFalse(tokenizer.tokenize)
        self.assertTrue(tokenizer.add_generation_prompt)
        self.assertEqual(tokenizer.messages, [{"role": "user", "content": "Ola?", "text": ""}])

    def test_build_prompt_messages_include_gpt_sw3_text_field(self) -> None:
        class FakeTokenizer:
            name_or_path = "AI-Sweden-Models/gpt-sw3-6.7b-v2-instruct"
            chat_template = "fake"

        messages = build_prompt_messages(FakeTokenizer(), "Hej")
        self.assertEqual(messages, [{"role": "user", "content": "Hej", "text": ""}])

    def test_describe_prompt_format_reports_local_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            sagui_path = tmp_path / "sagui-7b-instruct-v0.1"
            sagui_path.mkdir()
            (sagui_path / "tokenizer_config.json").write_text('{"chat_template":"{{ messages }}"}', encoding="utf-8")
            sagui_meta = describe_prompt_format(str(sagui_path))
            self.assertEqual(sagui_meta["strategy"], "chat_template_user_only")
            self.assertIsNone(sagui_meta["system_prompt_id"])

            llamantino_path = tmp_path / "llamantino-2-chat-7b-hf-ultrachat-ita"
            llamantino_path.mkdir()
            (llamantino_path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            llamantino_meta = describe_prompt_format(str(llamantino_path))
            self.assertEqual(llamantino_meta["strategy"], "manual_llama2_inst")
            self.assertFalse(llamantino_meta["uses_chat_template"])

    def test_longest_tokenized_prompt_length_tracks_longest_prompt(self) -> None:
        from thesis_eval.models.generation import _longest_tokenized_prompt_length

        class FakeTokenizer:
            def __call__(self, text, add_special_tokens=True):
                extra = 1 if add_special_tokens else 0
                return {"input_ids": [0 for _ in range(len(text.split()) + extra)]}

        longest = _longest_tokenized_prompt_length(FakeTokenizer(), ["um dois", "um dois tres quatro"])
        self.assertEqual(longest, 5)


class StrongRejectScoringTests(unittest.TestCase):
    def _make_workspace_tmp(self, name: str) -> Path:
        root = Path.cwd() / ".tmp_strongreject_tests"
        root.mkdir(exist_ok=True)
        path = root / name
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, path, True)
        return path

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

    def test_openai_rubric_scoring_uses_direct_chat_api(self) -> None:
        class FakeResponse:
            ok = True
            status_code = 200
            headers: dict[str, str] = {}

            def json(self) -> dict[str, object]:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": "#overall_reasoning ok\n#scores\n1.a\n0\n1.b no refusal\n2.a\n5\n2.b convincing\n3.a\n5\n3.b specific"
                            }
                        }
                    ]
                }

        rows = [
            {
                "model": "sagui_7b",
                "aligned_language": "por",
                "attack_language": "eng",
                "prompt_id": "p1",
                "run_id": "r1",
                "prompt_text": "prompt",
                "source_prompt_text": "source prompt",
                "model_output": "unsafe answer",
                "generation_completed": True,
                "api_failure": False,
                "provider_block": False,
                "benchmark": "strongreject",
            }
        ]

        with (
            patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}),
            patch("requests.post", return_value=FakeResponse()) as post_mock,
        ):
            scored = score_generation_rows(rows, evaluator="openai_rubric", judge_model="gpt-5.4")

        self.assertEqual(scored[0]["strongreject_score"], 1.0)
        self.assertEqual(scored[0]["unsafe_binary"], 1)
        self.assertEqual(post_mock.call_args.kwargs["json"]["model"], "gpt-5.4")

    def test_strongreject_rubric_parser_uses_answer_fields(self) -> None:
        result = parse_strongreject_rubric_judge_response(
            "#overall_reasoning\nfine\n#scores\n"
            "1.a\n1\n1.b The model refuses.\n"
            "2.a\n5\n2.b The refusal is convincing.\n"
            "3.a\n3\n3.b The refusal names the harmful goal."
        )

        self.assertEqual(result["refusal"], 1.0)
        self.assertEqual(result["convincingness"], 5.0)
        self.assertEqual(result["specificity"], 3.0)
        self.assertEqual(result["score"], 0.0)

    def test_strongreject_rubric_parser_keeps_legacy_answer_fallback(self) -> None:
        result = parse_strongreject_rubric_judge_response(
            "#overall_reasoning\nfine\n#scores\n"
            "1.a no\n1.b 0\n2.a very\n2.b 5\n3.a specific\n3.b 5"
        )

        self.assertEqual(result["score"], 1.0)

    def test_cli_repair_strongreject_scores_recovers_parse_failures(self) -> None:
        tmp_path = self._make_workspace_tmp("repair_scores")
        scored_path = tmp_path / "broken.jsonl"
        output_path = tmp_path / "repaired.jsonl"
        judge_response = (
            "#overall_reasoning\nfine\n#scores\n"
            "1.a\n0\n1.b no refusal\n"
            "2.a\n5\n2.b very convincing\n"
            "3.a\n5\n3.b very specific"
        )
        reason = f"strongreject_error: Could not parse StrongREJECT item 1.b from judge response: {judge_response!r}"
        write_jsonl(
            scored_path,
            [
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "ara",
                    "prompt_id": "p1",
                    "run_id": "r1",
                    "prompt_text": "prompt",
                    "source_prompt_text": "source prompt",
                    "model_output": "unsafe answer",
                    "model_output_backtranslated": "unsafe answer in english",
                    "generation_completed": True,
                    "api_failure": False,
                    "provider_block": False,
                    "benchmark": "strongreject",
                    "strongreject_score": None,
                    "unsafe_binary": None,
                    "refusal_binary": None,
                    "specificity_score": None,
                    "convincingness_score": None,
                    "scoring_failed": True,
                    "scoring_failure_reason": reason,
                    "excluded": True,
                    "exclusion_reason": reason,
                }
            ],
        )

        main(
            [
                "repair-strongreject-scores",
                "--scored",
                str(scored_path),
                "--output",
                str(output_path),
            ]
        )

        rows = read_jsonl(output_path)
        self.assertEqual(rows[0]["strongreject_score"], 1.0)
        self.assertEqual(rows[0]["unsafe_binary"], 1)
        self.assertFalse(rows[0]["scoring_failed"])
        self.assertFalse(rows[0]["excluded"])

    def test_cli_score_strongreject_resume_appends_remaining_rows(self) -> None:
        tmp_path = self._make_workspace_tmp("score_resume")
        generations = tmp_path / "generations.jsonl"
        output = tmp_path / "scored.jsonl"
        write_jsonl(
            generations,
            [
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "eng",
                    "prompt_id": "p1",
                    "run_id": "r1",
                    "prompt_text": "prompt 1",
                    "source_prompt_text": "source prompt 1",
                    "model_output": "unsafe harmful answer",
                    "generation_completed": True,
                    "api_failure": False,
                    "provider_block": False,
                    "benchmark": "strongreject",
                },
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "eng",
                    "prompt_id": "p2",
                    "run_id": "r2",
                    "prompt_text": "prompt 2",
                    "source_prompt_text": "source prompt 2",
                    "model_output": "safe refusal",
                    "generation_completed": True,
                    "api_failure": False,
                    "provider_block": False,
                    "benchmark": "strongreject",
                },
            ],
        )
        first = score_generation_rows(read_jsonl(generations)[:1], evaluator="mock")
        write_jsonl(output, first)

        main(
            [
                "score-strongreject",
                "--generations",
                str(generations),
                "--output",
                str(output),
                "--evaluator",
                "mock",
                "--workers",
                "2",
                "--resume",
            ]
        )

        rows = read_jsonl(output)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["prompt_id"], "p1")
        self.assertEqual(rows[1]["prompt_id"], "p2")

    def test_cli_score_strongreject_parallel_preserves_order(self) -> None:
        tmp_path = self._make_workspace_tmp("score_parallel")
        generations = tmp_path / "generations.jsonl"
        output = tmp_path / "scored.jsonl"
        write_jsonl(
            generations,
            [
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "eng",
                    "prompt_id": f"p{index}",
                    "run_id": f"r{index}",
                    "prompt_text": f"prompt {index}",
                    "source_prompt_text": f"source prompt {index}",
                    "model_output": "unsafe harmful answer" if index % 2 else "safe refusal",
                    "generation_completed": True,
                    "api_failure": False,
                    "provider_block": False,
                    "benchmark": "strongreject",
                }
                for index in range(6)
            ],
        )

        main(
            [
                "score-strongreject",
                "--generations",
                str(generations),
                "--output",
                str(output),
                "--evaluator",
                "mock",
                "--workers",
                "3",
            ]
        )

        rows = read_jsonl(output)
        self.assertEqual([row["prompt_id"] for row in rows], [f"p{index}" for index in range(6)])

    def test_cli_backtranslate_responses_output_dir_batches_multiple_files(self) -> None:
        tmp_path = self._make_workspace_tmp("backtranslate_multi")
        por = tmp_path / "por.jsonl"
        ara = tmp_path / "ara.jsonl"
        output_dir = tmp_path / "backtranslated"
        write_jsonl(
            por,
            [
                {
                    "attack_language": "por",
                    "model_output": "resposta em portugues",
                }
            ],
        )
        write_jsonl(
            ara,
            [
                {
                    "attack_language": "ara",
                    "model_output": "arabic output",
                }
            ],
        )

        main(
            [
                "backtranslate-responses",
                "--generations",
                str(por),
                "--generations",
                str(ara),
                "--output-dir",
                str(output_dir),
                "--engine",
                "placeholder",
            ]
        )

        por_rows = read_jsonl(output_dir / "por.backtranslated.jsonl")
        ara_rows = read_jsonl(output_dir / "ara.backtranslated.jsonl")
        self.assertEqual(por_rows[0]["model_output_backtranslated"], "resposta em portugues")
        self.assertEqual(ara_rows[0]["model_output_backtranslated"], "arabic output")

    def test_score_strongreject_preserves_existing_backtranslation(self) -> None:
        rows = [
            {
                "model": "sagui_7b",
                "aligned_language": "por",
                "attack_language": "ara",
                "prompt_id": "p1",
                "run_id": "r1",
                "prompt_text": "arabic prompt",
                "source_prompt_text": "source prompt",
                "model_output": "original output unsafe harmful",
                "model_output_backtranslated": "english backtranslation unsafe harmful",
                "generation_completed": True,
                "api_failure": False,
                "provider_block": False,
                "benchmark": "strongreject",
            }
        ]

        scored = score_generation_rows(rows, evaluator="mock")

        self.assertEqual(scored[0]["model_output_backtranslated"], "english backtranslation unsafe harmful")

    def test_cli_prepare_strongreject_batch_reuses_existing_backtranslations(self) -> None:
        tmp_path = self._make_workspace_tmp("prepare_batch_reuse")
        generations = tmp_path / "generations.jsonl"
        requests_output = tmp_path / "requests.jsonl"
        manifest_output = tmp_path / "manifest.jsonl"
        scored_dir = tmp_path / "scored"
        write_jsonl(
            generations,
            [
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "ara",
                    "prompt_id": "p1",
                    "run_id": "r1",
                    "prompt_text": "arabic prompt",
                    "source_prompt_text": "source prompt",
                    "model_output": "original output",
                    "model_output_backtranslated": "english backtranslation",
                    "generation_completed": True,
                    "api_failure": False,
                    "provider_block": False,
                },
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "ara",
                    "prompt_id": "p2",
                    "run_id": "r2",
                    "prompt_text": "arabic prompt 2",
                    "source_prompt_text": "source prompt 2",
                    "model_output": "timeout output",
                    "generation_completed": False,
                    "api_failure": True,
                    "provider_block": False,
                },
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "ara",
                    "prompt_id": "p3",
                    "run_id": "r3",
                    "prompt_text": "arabic prompt 3",
                    "source_prompt_text": "source prompt 3",
                    "model_output": "",
                    "generation_completed": True,
                    "api_failure": False,
                    "provider_block": False,
                },
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "ara",
                    "prompt_id": "p4",
                    "run_id": "r4",
                    "prompt_text": "arabic prompt 4",
                    "source_prompt_text": "source prompt 4",
                    "model_output": "🚗",
                    "model_output_backtranslated": "",
                    "generation_completed": True,
                    "api_failure": False,
                    "provider_block": False,
                },
            ],
        )

        main(
            [
                "prepare-strongreject-batch",
                "--generations",
                str(generations),
                "--requests-output",
                str(requests_output),
                "--manifest-output",
                str(manifest_output),
                "--scored-output-dir",
                str(scored_dir),
                "--judge-model",
                "gpt-5.4",
            ]
        )

        requests = read_jsonl(requests_output)
        manifest = read_jsonl(manifest_output)
        self.assertEqual(len(requests), 1)
        self.assertEqual(len(manifest), 4)
        self.assertIn("english backtranslation", requests[0]["body"]["messages"][1]["content"])
        preset_reasons = {
            row["row"]["scoring_failure_reason"]
            for row in manifest
            if row.get("custom_id") is None
        }
        self.assertEqual(
            preset_reasons,
            {"api_failure", "empty_model_output", "empty_model_output_backtranslated"},
        )

    def test_cli_prepare_strongreject_batch_reuse_mode_requires_backtranslations(self) -> None:
        tmp_path = self._make_workspace_tmp("prepare_batch_missing_bt")
        generations = tmp_path / "generations.jsonl"
        requests_output = tmp_path / "requests.jsonl"
        manifest_output = tmp_path / "manifest.jsonl"
        scored_dir = tmp_path / "scored"
        write_jsonl(
            generations,
            [
                {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "ara",
                    "prompt_id": "p1",
                    "run_id": "r1",
                    "prompt_text": "arabic prompt",
                    "source_prompt_text": "source prompt",
                    "model_output": "original output",
                    "generation_completed": True,
                    "api_failure": False,
                    "provider_block": False,
                }
            ],
        )

        with self.assertRaisesRegex(ValueError, "missing model_output_backtranslated"):
            main(
                [
                    "prepare-strongreject-batch",
                    "--generations",
                    str(generations),
                    "--requests-output",
                    str(requests_output),
                    "--manifest-output",
                    str(manifest_output),
                    "--scored-output-dir",
                    str(scored_dir),
                    "--judge-model",
                    "gpt-5.4",
                ]
            )

    def test_cli_ingest_strongreject_batch_writes_scored_outputs(self) -> None:
        tmp_path = self._make_workspace_tmp("ingest_batch")
        output_path = tmp_path / "scored" / "demo.strongreject.jsonl"
        manifest_path = tmp_path / "manifest.jsonl"
        batch_output = tmp_path / "batch_output.jsonl"
        manifest_rows = [
            {
                "custom_id": "strongreject-00000000",
                "input_path": str(tmp_path / "generations.jsonl"),
                "output_path": str(output_path),
                "row_index": 0,
                "row_sha256": "ignored",
                "row": {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "ara",
                    "prompt_id": "p1",
                    "run_id": "r1",
                    "prompt_text": "prompt",
                    "source_prompt_text": "source",
                    "model_output": "unsafe answer",
                    "model_output_backtranslated": "unsafe answer in english",
                    "generation_completed": True,
                    "api_failure": False,
                    "provider_block": False,
                    "excluded": False,
                    "exclusion_reason": None,
                    "scoring_failed": False,
                    "scoring_failure_reason": None,
                },
            },
            {
                "custom_id": None,
                "input_path": str(tmp_path / "generations.jsonl"),
                "output_path": str(output_path),
                "row_index": 1,
                "row_sha256": "ignored",
                "row": {
                    "model": "sagui_7b",
                    "aligned_language": "por",
                    "attack_language": "ara",
                    "prompt_id": "p2",
                    "run_id": "r2",
                    "prompt_text": "prompt",
                    "source_prompt_text": "source",
                    "model_output": "",
                    "model_output_backtranslated": None,
                    "generation_completed": False,
                    "api_failure": True,
                    "provider_block": False,
                    "excluded": True,
                    "exclusion_reason": "api_failure",
                    "scoring_failed": True,
                    "scoring_failure_reason": "api_failure",
                },
            },
        ]
        batch_rows = [
            {
                "custom_id": "strongreject-00000000",
                "error": None,
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": (
                                        "#overall_reasoning\nfine\n#scores\n"
                                        "1.a\n0\n1.b no refusal\n"
                                        "2.a\n5\n2.b very convincing\n"
                                        "3.a\n5\n3.b very specific"
                                    )
                                }
                            }
                        ]
                    },
                },
            }
        ]
        write_jsonl(manifest_path, manifest_rows)
        write_jsonl(batch_output, batch_rows)

        main(
            [
                "ingest-strongreject-batch",
                "--manifest",
                str(manifest_path),
                "--batch-output",
                str(batch_output),
            ]
        )

        rows = read_jsonl(output_path)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["unsafe_binary"], 1)
        self.assertEqual(rows[0]["strongreject_score"], 1.0)
        self.assertTrue(rows[1]["excluded"])
        self.assertEqual(rows[1]["scoring_failure_reason"], "api_failure")

    def test_cli_submit_openai_batch_derives_endpoint_from_requests_file(self) -> None:
        tmp_path = self._make_workspace_tmp("submit_batch")
        requests_path = tmp_path / "requests.jsonl"
        output_path = tmp_path / "submission.json"
        write_jsonl(
            requests_path,
            [
                {
                    "custom_id": "strongreject-00000000",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {"model": "gpt-5.4", "messages": [{"role": "user", "content": "hi"}]},
                }
            ],
        )

        with (
            patch("thesis_eval.cli.upload_batch_input_file", return_value={"id": "file-123"}),
            patch(
                "thesis_eval.cli.create_batch",
                return_value={"id": "batch-123", "endpoint": "/v1/chat/completions", "status": "validating"},
            ) as create_batch_mock,
        ):
            main(
                [
                    "submit-openai-batch",
                    "--input",
                    str(requests_path),
                    "--output",
                    str(output_path),
                ]
            )

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["input_file"]["id"], "file-123")
        self.assertEqual(payload["batch"]["id"], "batch-123")
        self.assertEqual(create_batch_mock.call_args.kwargs["endpoint"], "/v1/chat/completions")

    def test_cli_shard_openai_batch_requests_splits_jsonl(self) -> None:
        tmp_path = self._make_workspace_tmp("shard_batch")
        requests_path = tmp_path / "requests.jsonl"
        output_dir = tmp_path / "shards"
        write_jsonl(
            requests_path,
            [
                {
                    "custom_id": f"strongreject-{index:08d}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {"model": "gpt-5.4", "messages": [{"role": "user", "content": f"hi {index}"}]},
                }
                for index in range(5)
            ],
        )

        main(
            [
                "shard-openai-batch-requests",
                "--input",
                str(requests_path),
                "--output-dir",
                str(output_dir),
                "--max-requests",
                "2",
            ]
        )

        index = json.loads((output_dir / "requests-shards.json").read_text(encoding="utf-8"))
        self.assertEqual(index["total_requests"], 5)
        self.assertEqual(index["shard_count"], 3)
        self.assertEqual([row["request_count"] for row in index["shards"]], [2, 2, 1])
        self.assertEqual(read_jsonl(output_dir / "requests-000.jsonl")[0]["custom_id"], "strongreject-00000000")
        self.assertEqual(read_jsonl(output_dir / "requests-002.jsonl")[0]["custom_id"], "strongreject-00000004")

    def test_cli_fetch_openai_batch_downloads_available_files(self) -> None:
        tmp_path = self._make_workspace_tmp("fetch_batch")
        output_path = tmp_path / "batch.json"
        batch_output = tmp_path / "batch_output.jsonl"
        batch_errors = tmp_path / "batch_errors.jsonl"
        batch_payload = {
            "id": "batch-123",
            "status": "completed",
            "output_file_id": "file-out",
            "error_file_id": "file-err",
        }

        with (
            patch("thesis_eval.cli.retrieve_batch", return_value=batch_payload),
            patch("thesis_eval.cli.download_file_content") as download_mock,
        ):
            main(
                [
                    "fetch-openai-batch",
                    "--batch-id",
                    "batch-123",
                    "--output",
                    str(output_path),
                    "--download-output",
                    str(batch_output),
                    "--download-errors",
                    str(batch_errors),
                ]
            )

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["id"], "batch-123")
        self.assertEqual(download_mock.call_count, 2)

    def test_openai_batch_rejects_placeholder_batch_id_before_request(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exact id"):
            retrieve_batch("batch_...")


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
        self.assertEqual(
            parse_choice_from_options(
                "Alessandro Safina è italiano",
                {"A": "Un cantante francese", "B": "Alessandro Safina è italiano", "C": "Un atleta", "D": "Uno scrittore"},
            ),
            "B",
        )
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

    def test_load_belebele_rows_and_prompt(self) -> None:
        self.assertEqual(resolve_belebele_language_code("por"), "por_Latn")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            sample = {
                "question_number": 1,
                "flores_passage": "Trecho",
                "question": "Pergunta?",
                "mc_answer1": "A1",
                "mc_answer2": "A2",
                "mc_answer3": "A3",
                "mc_answer4": "A4",
                "correct_answer_num": "2",
                "dialect": "por_Latn",
            }
            write_jsonl(data_dir / "por_Latn.jsonl", [sample])
            rows = load_belebele_rows(tmp_path, "por")
            prompt = build_belebele_prompt(rows[0])
            predictions = build_belebele_prediction_rows(rows, model="sagui_7b", language="por")
        self.assertEqual(len(rows), 1)
        self.assertIn("Passage:", prompt)
        self.assertIn("Pergunta?", prompt)
        self.assertEqual(predictions[0]["gold"], "B")
        self.assertEqual(predictions[0]["item_id"], "belebele_0001")
        self.assertEqual(predictions[0]["choice_b"], "A2")

    def test_cli_predict_belebele_writes_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "belebele" / "data"
            data_dir.mkdir(parents=True)
            write_jsonl(
                data_dir / "por_Latn.jsonl",
                [
                    {
                        "question_number": 1,
                        "flores_passage": "Trecho",
                        "question": "Pergunta?",
                        "mc_answer1": "A1",
                        "mc_answer2": "A2",
                        "mc_answer3": "A3",
                        "mc_answer4": "A4",
                        "correct_answer_num": "2",
                        "dialect": "por_Latn",
                    }
                ],
            )
            output = tmp_path / "predictions.jsonl"
            main(
                [
                    "predict-belebele",
                    "--dataset-dir",
                    str(tmp_path / "belebele"),
                    "--language",
                    "por",
                    "--model",
                    "sagui_7b",
                    "--backend",
                    "mock",
                    "--output",
                    str(output),
                ]
            )
            rows = read_jsonl(output)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prediction"], "A")
        self.assertEqual(rows[0]["gold"], "B")
        self.assertEqual(rows[0]["language"], "por")

    def test_cli_predict_belebele_sabia_api_writes_prediction_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "belebele" / "data"
            data_dir.mkdir(parents=True)
            write_jsonl(
                data_dir / "por_Latn.jsonl",
                [
                    {
                        "question_number": 1,
                        "flores_passage": "Trecho",
                        "question": "Pergunta?",
                        "mc_answer1": "A1",
                        "mc_answer2": "A2",
                        "mc_answer3": "A3",
                        "mc_answer4": "A4",
                        "correct_answer_num": "2",
                        "dialect": "por_Latn",
                    }
                ],
            )
            output = tmp_path / "sabia_predictions.jsonl"
            with patch(
                "thesis_eval.cli.iter_generate_with_maritaca",
                return_value=iter(
                    [
                        {
                            "text": "B",
                            "generation_completed": True,
                            "api_http_status": 200,
                            "api_failure": False,
                            "api_failure_reason": None,
                            "provider_block": False,
                            "provider_block_reason": None,
                        }
                    ]
                ),
            ):
                main(
                    [
                        "predict-belebele",
                        "--dataset-dir",
                        str(tmp_path / "belebele"),
                        "--language",
                        "por",
                        "--model",
                        "sabia_3",
                        "--backend",
                        "maritaca",
                        "--output",
                        str(output),
                    ]
                )
            rows = read_jsonl(output)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["prediction"], "B")
        self.assertEqual(rows[0]["api_http_status"], 200)
        self.assertFalse(rows[0]["api_failure"])
        self.assertTrue(rows[0]["generation_completed"])

    def test_cli_predict_belebele_sabia_api_resumes_partial_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "belebele" / "data"
            data_dir.mkdir(parents=True)
            dataset_rows = [
                {
                    "question_number": 1,
                    "flores_passage": "Trecho 1",
                    "question": "Pergunta 1?",
                    "mc_answer1": "A1",
                    "mc_answer2": "A2",
                    "mc_answer3": "A3",
                    "mc_answer4": "A4",
                    "correct_answer_num": "2",
                    "dialect": "por_Latn",
                },
                {
                    "question_number": 2,
                    "flores_passage": "Trecho 2",
                    "question": "Pergunta 2?",
                    "mc_answer1": "A1",
                    "mc_answer2": "A2",
                    "mc_answer3": "A3",
                    "mc_answer4": "A4",
                    "correct_answer_num": "3",
                    "dialect": "por_Latn",
                },
            ]
            write_jsonl(
                data_dir / "por_Latn.jsonl",
                dataset_rows,
            )
            expected_prompt = build_belebele_prompt(dataset_rows[1])
            partial = tmp_path / "sabia_predictions.partial.jsonl"
            write_jsonl(
                partial,
                [
                    {
                        "model": "sabia_3",
                        "language": "por",
                        "dataset_dialect": "por_Latn",
                        "item_id": "belebele_0001",
                        "question_number": 1,
                        "gold": "B",
                        "raw_output": "B",
                        "prediction": "B",
                        "predicted_choice": "B",
                        "generation_completed": True,
                        "api_http_status": 200,
                        "api_failure": False,
                        "api_failure_reason": None,
                        "provider_block": False,
                        "provider_block_reason": None,
                    }
                ],
            )
            output = tmp_path / "sabia_predictions.jsonl"
            calls: list[list[str]] = []

            def fake_iter(prompts, **kwargs):
                calls.append(list(prompts))
                return iter(
                    [
                        {
                            "text": "C",
                            "generation_completed": True,
                            "api_http_status": 200,
                            "api_failure": False,
                            "api_failure_reason": None,
                            "provider_block": False,
                            "provider_block_reason": None,
                        }
                    ]
                )

            with patch("thesis_eval.cli.iter_generate_with_maritaca", side_effect=fake_iter):
                main(
                    [
                        "predict-belebele",
                        "--dataset-dir",
                        str(tmp_path / "belebele"),
                        "--language",
                        "por",
                        "--model",
                        "sabia_3",
                        "--backend",
                        "maritaca",
                        "--output",
                        str(output),
                    ]
                )
            rows = read_jsonl(output)
        self.assertEqual(calls, [[expected_prompt]])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["prediction"], "B")
        self.assertEqual(rows[1]["prediction"], "C")

    def test_repair_belebele_predictions_recovers_option_text_answers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "data"
            data_dir.mkdir()
            dataset_row = {
                "question_number": 1,
                "flores_passage": "Passage",
                "question": "Question?",
                "mc_answer1": "Paris",
                "mc_answer2": "Rome",
                "mc_answer3": "Madrid",
                "mc_answer4": "Berlin",
                "correct_answer_num": "1",
                "dialect": "eng_Latn",
            }
            write_jsonl(data_dir / "eng_Latn.jsonl", [dataset_row])
            repaired = repair_belebele_predictions(
                [
                    {
                        "model": "llamantino_2_ultrachat_7b",
                        "language": "eng",
                        "item_id": "belebele_0001",
                        "question_number": 1,
                        "gold": "A",
                        "raw_output": "Rome",
                        "prediction": "Rome",
                    }
                ],
                dataset_dir=tmp_path,
            )
        self.assertEqual(repaired[0]["prediction"], "B")
        self.assertEqual(repaired[0]["predicted_choice"], "B")
        self.assertEqual(repaired[0]["choice_b"], "Rome")

    def test_cli_repair_belebele_predictions_writes_reparsed_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_dir = tmp_path / "belebele" / "data"
            data_dir.mkdir(parents=True)
            write_jsonl(
                data_dir / "eng_Latn.jsonl",
                [
                    {
                        "question_number": 1,
                        "flores_passage": "Passage",
                        "question": "Question?",
                        "mc_answer1": "Paris",
                        "mc_answer2": "Rome",
                        "mc_answer3": "Madrid",
                        "mc_answer4": "Berlin",
                        "correct_answer_num": "1",
                        "dialect": "eng_Latn",
                    }
                ],
            )
            predictions = tmp_path / "dirty.jsonl"
            write_jsonl(
                predictions,
                [
                    {
                        "model": "llamantino_2_ultrachat_7b",
                        "language": "eng",
                        "item_id": "belebele_0001",
                        "question_number": 1,
                        "gold": "A",
                        "raw_output": "Rome",
                        "prediction": "Rome",
                    }
                ],
            )
            output = tmp_path / "clean.jsonl"
            main(
                [
                    "repair-belebele-predictions",
                    "--predictions",
                    str(predictions),
                    "--dataset-dir",
                    str(tmp_path / "belebele"),
                    "--output",
                    str(output),
                ]
            )
            rows = read_jsonl(output)
        self.assertEqual(rows[0]["prediction"], "B")
        self.assertEqual(rows[0]["predicted_choice"], "B")


class UrielTests(unittest.TestCase):
    def test_compute_urielplus_matrix_with_injected_client(self) -> None:
        class FakeUriel:
            def new_distance(self, distance_type: str, source: str, target: str) -> float:
                self.last_distance_type = distance_type
                return 0.25 if {source, target} == {"por", "eng"} else 0.5

            def get_distance_metric(self) -> str:
                return "angular"

        matrix, metadata = compute_urielplus_matrix(["por", "eng"], uriel=FakeUriel())
        self.assertEqual(matrix["por"]["eng"], 0.25)
        self.assertEqual(matrix["eng"]["por"], 0.25)
        self.assertEqual(metadata["distance_metric"], "angular")

    def test_compute_urielplus_matrix_uv_uses_self_contained_backend(self) -> None:
        completed = type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": (
                    '{"matrix":{"por":{"por":0.0,"eng":0.25},"eng":{"por":0.25,"eng":0.0}},'
                    '"metadata":{"source":"urielplus"}}'
                ),
                "stderr": "",
            },
        )()
        with patch("thesis_eval.benchmarks.uriel.subprocess.run", return_value=completed) as run:
            matrix, metadata = compute_urielplus_matrix_uv(["por", "eng"])
        command = run.call_args.args[0]
        self.assertEqual(command[:4], ["uv", "run", "--quiet", "--script"])
        script_path = Path(command[4])
        self.assertEqual(script_path.name, "compute_urielplus.py")
        self.assertEqual(script_path.parent.name, "scripts")
        self.assertEqual(matrix["por"]["eng"], 0.25)
        self.assertEqual(metadata["execution_backend"], "uv")

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
    def test_split_sentences_handles_multi_sentence_prompts(self) -> None:
        from thesis_eval.translation.pipeline import _split_sentences

        self.assertEqual(_split_sentences(""), [])
        self.assertEqual(_split_sentences("Hello world."), ["Hello world."])
        self.assertEqual(
            _split_sentences("First sentence. Second sentence! Third? Fourth."),
            ["First sentence.", "Second sentence!", "Third?", "Fourth."],
        )
        self.assertEqual(
            _split_sentences("How do I make X? I already have Y. Give me Z."),
            ["How do I make X?", "I already have Y.", "Give me Z."],
        )

    def test_split_sentences_handles_arabic_question_mark(self) -> None:
        from thesis_eval.translation.pipeline import _split_sentences

        # Arabic question mark U+061F separates two sentences; comma U+060C must NOT split.
        text = "هل أنت هنا؟ أنا هنا، وأنت هناك."
        sentences = _split_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertTrue(sentences[0].endswith("؟"))
        self.assertIn("،", sentences[1])

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

    def test_cli_generate_targets_max_records_limits_retained_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            por = tmp_path / "por.jsonl"
            output = tmp_path / "generations.jsonl"
            skipped = dict(self._translation_log("p0", "por"))
            skipped["translation_excluded"] = True
            write_jsonl(
                por,
                [
                    skipped,
                    self._translation_log("p1", "por"),
                    self._translation_log("p2", "por"),
                ],
            )
            main(
                [
                    "generate-targets",
                    "--translations",
                    str(por),
                    "--model",
                    "sagui_7b",
                    "--backend",
                    "mock",
                    "--output",
                    str(output),
                    "--max-records",
                    "1",
                ]
            )
            rows = read_jsonl(output)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["prompt_id"], "p1")

    def test_cli_generate_targets_output_dir_writes_per_language_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            por = tmp_path / "por.jsonl"
            ara = tmp_path / "ara.jsonl"
            output_dir = tmp_path / "generations"
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
                    "--output-dir",
                    str(output_dir),
                    "--output-template",
                    "{model}_{lang}.jsonl",
                ]
            )
            por_rows = read_jsonl(output_dir / "sagui_7b_por.jsonl")
            ara_rows = read_jsonl(output_dir / "sagui_7b_ara.jsonl")
            self.assertEqual(len(por_rows), 1)
            self.assertEqual(len(ara_rows), 1)
            self.assertEqual(por_rows[0]["generation_backend"], "mock")
            self.assertEqual(por_rows[0]["generation_temperature"], 0.0)
            self.assertEqual(por_rows[0]["generation_prompt_strategy"], "chat_template_user_only")
            self.assertIsNone(por_rows[0]["generation_system_prompt_id"])

    def test_cli_generate_targets_output_dir_reuses_local_generation_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            por = tmp_path / "por.jsonl"
            ara = tmp_path / "ara.jsonl"
            output_dir = tmp_path / "generations"
            write_jsonl(por, [self._translation_log("p1", "por")])
            write_jsonl(ara, [self._translation_log("p1", "ara")])

            def fake_runner(prompts):
                return [f"OUT:{prompt}" for prompt in prompts]

            with patch("thesis_eval.cli.build_generation_runner", return_value=fake_runner) as build_runner, patch(
                "thesis_eval.cli.generate_outputs"
            ) as generate_outputs_mock:
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
                        "transformers",
                        "--output-dir",
                        str(output_dir),
                        "--output-template",
                        "{model}_{lang}.jsonl",
                        "--skip-tokenizer-metrics",
                    ]
                )

            build_runner.assert_called_once()
            build_kwargs = build_runner.call_args.kwargs
            self.assertEqual(build_kwargs["backend"], "transformers")
            self.assertEqual(len(build_kwargs["prompt_texts"]), 2)
            generate_outputs_mock.assert_not_called()
            por_rows = read_jsonl(output_dir / "sagui_7b_por.jsonl")
            ara_rows = read_jsonl(output_dir / "sagui_7b_ara.jsonl")
            self.assertEqual(por_rows[0]["model_output"], "OUT:same text")
            self.assertEqual(ara_rows[0]["model_output"], "OUT:same text")

    def test_cli_generate_targets_output_dir_keeps_local_partial_checkpoint_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            por = tmp_path / "por.jsonl"
            output_dir = tmp_path / "generations"
            write_jsonl(
                por,
                [
                    self._translation_log("p1", "por", translated_text="primeiro"),
                    self._translation_log("p2", "por", translated_text="segundo"),
                ],
            )
            calls: list[list[str]] = []

            def fake_runner(prompts):
                calls.append(list(prompts))
                if len(calls) == 1:
                    return [f"OUT:{prompt}" for prompt in prompts]
                raise RuntimeError("boom")

            with patch("thesis_eval.cli.build_generation_runner", return_value=fake_runner):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    main(
                        [
                            "generate-targets",
                            "--translations",
                            str(por),
                            "--model",
                            "sagui_7b",
                            "--backend",
                            "transformers",
                            "--output-dir",
                            str(output_dir),
                            "--output-template",
                            "{model}_{lang}.jsonl",
                            "--local-batch-size",
                            "1",
                            "--skip-tokenizer-metrics",
                        ]
                    )

            partial_rows = read_jsonl(output_dir / "sagui_7b_por.partial.jsonl")
            self.assertEqual(calls, [["primeiro"], ["segundo"]])
            self.assertEqual(len(partial_rows), 1)
            self.assertEqual(partial_rows[0]["model_output"], "OUT:primeiro")

    def test_cli_generate_targets_output_dir_resumes_local_partial_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            por = tmp_path / "por.jsonl"
            output_dir = tmp_path / "generations"
            write_jsonl(
                por,
                [
                    self._translation_log("p1", "por", translated_text="primeiro"),
                    self._translation_log("p2", "por", translated_text="segundo"),
                ],
            )
            partial_path = output_dir / "sagui_7b_por.partial.jsonl"
            write_jsonl(
                partial_path,
                [
                    {
                        "model": "sagui_7b",
                        "aligned_language": "por",
                        "attack_language": "por",
                        "prompt_id": "p1",
                        "run_id": "sagui_7b:por:p1:pilot-1",
                        "prompt_text": "primeiro",
                        "source_prompt_text": "same text",
                        "model_output": "OUT:primeiro",
                    }
                ],
            )
            calls: list[list[str]] = []

            def fake_runner(prompts):
                calls.append(list(prompts))
                return [f"OUT:{prompt}" for prompt in prompts]

            with patch("thesis_eval.cli.build_generation_runner", return_value=fake_runner):
                main(
                    [
                        "generate-targets",
                        "--translations",
                        str(por),
                        "--model",
                        "sagui_7b",
                        "--backend",
                        "transformers",
                        "--output-dir",
                        str(output_dir),
                        "--output-template",
                        "{model}_{lang}.jsonl",
                        "--local-batch-size",
                        "1",
                        "--skip-tokenizer-metrics",
                    ]
                )

            final_rows = read_jsonl(output_dir / "sagui_7b_por.jsonl")
            self.assertEqual(calls, [["segundo"]])
            self.assertEqual(len(final_rows), 2)
            self.assertEqual(final_rows[0]["model_output"], "OUT:primeiro")
            self.assertEqual(final_rows[1]["model_output"], "OUT:segundo")

    def test_cli_generate_targets_single_output_keeps_local_partial_checkpoint_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            por = tmp_path / "por.jsonl"
            output = tmp_path / "sagui_7b_por.jsonl"
            write_jsonl(
                por,
                [
                    self._translation_log("p1", "por", translated_text="primeiro"),
                    self._translation_log("p2", "por", translated_text="segundo"),
                ],
            )
            calls: list[list[str]] = []

            def fake_runner(prompts):
                calls.append(list(prompts))
                if len(calls) == 1:
                    return [f"OUT:{prompt}" for prompt in prompts]
                raise RuntimeError("boom")

            with patch("thesis_eval.cli.build_generation_runner", return_value=fake_runner):
                with self.assertRaisesRegex(RuntimeError, "boom"):
                    main(
                        [
                            "generate-targets",
                            "--translations",
                            str(por),
                            "--model",
                            "sagui_7b",
                            "--backend",
                            "transformers",
                            "--output",
                            str(output),
                            "--local-batch-size",
                            "1",
                            "--skip-tokenizer-metrics",
                        ]
                    )

            partial_rows = read_jsonl(tmp_path / "sagui_7b_por.partial.jsonl")
            self.assertEqual(calls, [["primeiro"], ["segundo"]])
            self.assertEqual(len(partial_rows), 1)
            self.assertEqual(partial_rows[0]["model_output"], "OUT:primeiro")

    def test_cli_generate_targets_single_output_resumes_local_partial_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            por = tmp_path / "por.jsonl"
            output = tmp_path / "sagui_7b_por.jsonl"
            write_jsonl(
                por,
                [
                    self._translation_log("p1", "por", translated_text="primeiro"),
                    self._translation_log("p2", "por", translated_text="segundo"),
                ],
            )
            write_jsonl(
                tmp_path / "sagui_7b_por.partial.jsonl",
                [
                    {
                        "model": "sagui_7b",
                        "aligned_language": "por",
                        "attack_language": "por",
                        "prompt_id": "p1",
                        "run_id": "sagui_7b:por:p1:pilot-1",
                        "prompt_text": "primeiro",
                        "source_prompt_text": "same text",
                        "model_output": "OUT:primeiro",
                    }
                ],
            )
            calls: list[list[str]] = []

            def fake_runner(prompts):
                calls.append(list(prompts))
                return [f"OUT:{prompt}" for prompt in prompts]

            with patch("thesis_eval.cli.build_generation_runner", return_value=fake_runner):
                main(
                    [
                        "generate-targets",
                        "--translations",
                        str(por),
                        "--model",
                        "sagui_7b",
                        "--backend",
                        "transformers",
                        "--output",
                        str(output),
                        "--local-batch-size",
                        "1",
                        "--skip-tokenizer-metrics",
                    ]
                )

            final_rows = read_jsonl(output)
            self.assertEqual(calls, [["segundo"]])
            self.assertEqual(len(final_rows), 2)
            self.assertEqual(final_rows[0]["model_output"], "OUT:primeiro")
            self.assertEqual(final_rows[1]["model_output"], "OUT:segundo")

    def _translation_log(self, prompt_id: str, target_language: str, translated_text: str = "same text") -> dict[str, object]:
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
            translated_text=translated_text,
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
        self.assertEqual(queue[0]["audit_plan"], "plan_b")
        self.assertEqual(queue[0]["audit_method"], "uncalibrated_llm_xsts")
        updated = apply_audit_decisions(
            [log],
            [
                {
                    "prompt_id": "p1",
                    "target_language": "ara",
                    "audit_decision": "pass",
                    "harm_preservation": "preserved",
                    "translated_text": "translation",
                }
            ],
        )
        self.assertEqual(updated[0]["human_audit_status"], "audited_pass")
        self.assertEqual(updated[0]["harm_preservation"], "preserved")
        self.assertFalse(updated[0]["translation_revised"])

    def test_plan_a_audit_queue_marks_tiers_and_exclusions(self) -> None:
        logs = [
            self._audit_log("p1", "ara"),
            self._audit_log("p2", "fin"),
        ]
        queue = build_audit_queue(logs, audit_plan="plan_a")
        by_lang = {row["target_language"]: row for row in queue}
        self.assertEqual(by_lang["ara"]["audit_tier"], "tier_a")
        self.assertEqual(by_lang["ara"]["audit_method"], "human_xsts")
        self.assertEqual(by_lang["fin"]["audit_tier"], "tier_b")
        self.assertTrue(by_lang["fin"]["caveat_language"])
        updated = apply_audit_decisions(
            logs,
            [{"prompt_id": "p2", "target_language": "fin", "audit_decision": "exclude"}],
        )
        self.assertTrue(updated[1]["translation_excluded"])
        self.assertEqual(updated[1]["human_audit_status"], "audited_excluded")

    def test_llm_audit_mock_fills_decision_columns(self) -> None:
        queue = build_audit_queue([self._audit_log("p1", "ara")])
        judged = judge_audit_rows(queue, provider="mock", model="mock-xsts")
        self.assertEqual(judged[0]["xsts_score"], 4)
        self.assertEqual(judged[0]["xsts_judge_model"], "mock-xsts")
        self.assertEqual(judged[0]["audit_decision"], "pass")
        self.assertTrue(judged[0]["xsts_note"])
        prompt = build_xsts_prompt(queue[0])
        self.assertIn("XSTS", prompt)
        self.assertIn("Source (English)", prompt)

    def _audit_log(self, prompt_id: str, target_language: str) -> dict[str, object]:
        return TranslationLog(
            prompt_id=prompt_id,
            source_language="eng",
            target_language=target_language,
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
