from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from thesis_eval.assets import asset_status, download_asset, load_assets, select_assets, verify_assets
from thesis_eval.analysis.dataset import build_frozen_rows
from thesis_eval.analysis.reporting import (
    asr_by_model_language,
    closest_farthest_by_model,
    collinearity_diagnostics,
    coverage_by_model,
    prereg_distance_slope_retention,
    prereg_falsification_summary,
    spearman_tables,
    tokenizer_diagnostics,
)
from thesis_eval.analysis.glmm import fit_main_glmm
from thesis_eval.benchmarks.belebele import compute_if_cons
from thesis_eval.benchmarks.strongreject import import_strongreject, write_prompt_records
from thesis_eval.benchmarks.uriel import (
    compare_matrices,
    compute_urielplus_matrix,
    compute_urielplus_matrix_external,
    compute_urielplus_matrix_uv,
    load_distance_matrix,
    subset_matrix,
    write_distance_matrix,
    write_latex_table,
    write_run_config,
)
from thesis_eval.config import load_config
from thesis_eval.evaluation.benign import score_benign_rows
from thesis_eval.evaluation.strongreject import score_generation_rows
from thesis_eval.io import read_csv_dicts, read_jsonl, write_csv_dicts, write_jsonl
from thesis_eval.metrics.tokenizer import attach_tokenizer_metrics, load_tokenizer
from thesis_eval.models.generation import attach_outputs, build_generation_rows, estimate_model_asset_state, generate_outputs
from thesis_eval.paths import DATA_DIR, OUTPUT_DIR, THESIS_ROOT, ensure_dir
from thesis_eval.pipeline import enrich_generation_rows, run_mock_pilot
from thesis_eval.progress import info, step
from thesis_eval.runtime import load_runtime_profiles, resolve_runtime_profile
from thesis_eval.translation.sonar import prewarm_sonar_text_encoder
from thesis_eval.translation.backtranslate import attach_prompt_roundtrip, attach_response_backtranslations
from thesis_eval.schema import validate_analysis_rows
from thesis_eval.translation.audit import AUDIT_COLUMNS, apply_audit_decisions, build_audit_queue
from thesis_eval.translation.calibration import calibrate_blaser_thresholds
from thesis_eval.translation.log import load_translation_logs, save_translation_logs
from thesis_eval.translation.pipeline import NllbTranslator, PromptRecord, apply_translation_qc, run_translation
from thesis_eval.translation.reference_spbleu import attach_reference_scores


def cmd_validate_config(_: argparse.Namespace) -> None:
    cfg = load_config()
    print(f"OK: {len(cfg.models)} models, {len(cfg.attack_languages)} languages")


def cmd_runtime_profiles(_: argparse.Namespace) -> None:
    profiles = load_runtime_profiles()
    for profile in profiles.values():
        print(f"{profile.name}: backend={profile.target_backend} device={profile.device} dtype={profile.dtype} - {profile.notes}")


def cmd_list_assets(_: argparse.Namespace) -> None:
    for asset in load_assets().values():
        repo = f" ({asset.repo_id})" if asset.repo_id else ""
        print(f"{asset.name}: kind={asset.kind} group={asset.group}{repo} - {asset.description}")


def cmd_asset_status(args: argparse.Namespace) -> None:
    selected = select_assets(names=args.name, groups=args.group)
    print(json.dumps([asset_status(asset) for asset in selected], indent=2))


def cmd_download_assets(args: argparse.Namespace) -> None:
    selected = select_assets(names=args.name, groups=args.group)
    info(f"Selected {len(selected)} asset(s): {', '.join(asset.name for asset in selected)}")
    results = [download_asset(asset, dry_run=args.dry_run, force=args.force) for asset in selected]
    print(json.dumps(results, indent=2))


def cmd_verify_assets(args: argparse.Namespace) -> None:
    selected = select_assets(names=args.name, groups=args.group)
    info(f"Verifying {len(selected)} asset(s): {', '.join(asset.name for asset in selected)}")
    print(json.dumps(verify_assets(selected, deep=args.deep), indent=2))


def cmd_prepare_uriel(args: argparse.Namespace) -> None:
    cfg = load_config()
    source = _resolve_uriel_source(Path(args.source_csv), required=args.from_csv)
    if args.from_csv:
        if source is None:
            raise FileNotFoundError("A source CSV is required when --from-csv is set")
        matrix = subset_matrix(load_distance_matrix(source), cfg.attack_languages)
        metadata: dict[str, Any] = {"source": "csv", "source_csv": str(source)}
        mode = "csv"
    else:
        if args.uriel_python:
            uriel_python = Path(args.uriel_python).expanduser()
            if not uriel_python.exists():
                raise FileNotFoundError(f"--uriel-python does not exist: {uriel_python}")
            matrix, metadata = compute_urielplus_matrix_external(
                uriel_python,
                cfg.attack_languages,
                distance_type=args.distance_type,
            )
        elif args.uriel_backend == "in-process":
            matrix, metadata = compute_urielplus_matrix(cfg.attack_languages, distance_type=args.distance_type)
        else:
            matrix, metadata = compute_urielplus_matrix_uv(
                cfg.attack_languages,
                distance_type=args.distance_type,
            )
        mode = "urielplus"
    output_csv = Path(args.output_csv)
    output_tex = Path(args.output_tex)
    write_distance_matrix(output_csv, matrix, cfg.attack_languages)
    write_latex_table(output_tex, matrix, cfg.attack_languages)
    if args.run_config_output:
        write_run_config(Path(args.run_config_output), metadata)
    comparison: dict[str, Any] = {}
    if source is not None:
        comparison = {
            "comparison_source_csv": str(source),
            **compare_matrices(matrix, load_distance_matrix(source)),
        }
    print(
        json.dumps(
            {
                "mode": mode,
                "distance_type": args.distance_type,
                "csv": str(output_csv),
                "latex": str(output_tex),
                "run_config": args.run_config_output,
                **comparison,
            },
            indent=2,
        )
    )


def _resolve_uriel_source(source: Path, required: bool = False) -> Path | None:
    if source.exists():
        return source
    candidates = [
        THESIS_ROOT / "pictures" / source.name,
        DATA_DIR / "uriel_plus" / source.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    if not required:
        return None
    searched = [str(source)] + [str(candidate) for candidate in candidates]
    raise FileNotFoundError(
        "URIEL+ CSV preparation expects an existing static distance-matrix CSV. "
        f"Searched: {', '.join(searched)}"
    )


def cmd_import_strongreject(args: argparse.Namespace) -> None:
    records = import_strongreject(
        source=args.source,
        input_path=Path(args.input) if args.input else None,
        limit=args.limit,
        small=args.small,
    )
    output = Path(args.output)
    write_prompt_records(output, records)
    if args.pilot_output:
        pilot_output = Path(args.pilot_output)
        write_prompt_records(pilot_output, records[: args.pilot_size])
    else:
        pilot_output = None
    print(
        json.dumps(
            {
                "records": len(records),
                "output": str(output),
                "pilot_records": min(args.pilot_size, len(records)) if pilot_output else 0,
                "pilot_output": str(pilot_output) if pilot_output else None,
                "source": args.source,
                "small": args.small,
            },
            indent=2,
        )
    )


def _load_prompt_records(path: Path) -> list[PromptRecord]:
    rows = read_jsonl(path)
    prompts: list[PromptRecord] = []
    for row in rows:
        prompts.append(
            PromptRecord(
                prompt_id=str(row["prompt_id"]),
                text=str(row["text"]),
                harmful_goal=str(row.get("harmful_goal", "pending_manual_intent_spec")),
                expected_output_form=str(row.get("expected_output_form", "other")),
                fixed_constraints=str(row.get("fixed_constraints", "pending_manual_intent_spec")),
                allowed_adaptation=str(row.get("allowed_adaptation", "fluency_only_no_semantic_change")),
            )
        )
    return prompts


def cmd_translate(args: argparse.Namespace) -> None:
    prompts = _load_prompt_records(Path(args.prompts))
    target_languages = list(args.target_language)
    if len(target_languages) > 1 and args.output:
        raise ValueError("Use --output-dir for multi-language translation; --output is only for a single target language.")
    if not args.output and not args.output_dir:
        raise ValueError("translate requires --output for one language or --output-dir for one/more languages.")

    pending: list[tuple[str, Path]] = []
    for target_language in target_languages:
        output_path = _translation_output_path(args, target_language)
        if _skip_existing(output_path, args.force, f"translate -> {target_language}"):
            continue
        pending.append((target_language, output_path))
    if not pending:
        info("translate: all target outputs already exist; nothing to do.")
        return

    translator = None
    if args.engine == "nllb" and any(language != "eng" for language, _ in pending):
        translator = NllbTranslator(checkpoint=args.checkpoint, device=args.device, dtype=args.dtype)

    written: dict[str, str] = {}
    for target_language, output_path in pending:
        with step(f"translate file {args.prompts} -> {target_language}"):
            logs = run_translation(
                prompts,
                target_language,
                engine=args.engine,
                checkpoint=args.checkpoint,
                device=args.device,
                dtype=args.dtype,
                stream=args.stream,
                translator=translator,
            )
        save_translation_logs(output_path, logs)
        written[target_language] = str(output_path)
        print(f"Wrote {len(logs)} translation logs to {output_path}")
    if len(written) > 1:
        print(json.dumps(written, ensure_ascii=False, indent=2, sort_keys=True))


def _translation_output_path(args: argparse.Namespace, target_language: str) -> Path:
    if args.output:
        return Path(args.output)
    output_dir = ensure_dir(Path(args.output_dir))
    filename = str(args.output_template).format(lang=target_language, target_language=target_language)
    return output_dir / filename


def _skip_existing(output_path: Path, force: bool, what: str) -> bool:
    """Resume helper: True if output already exists non-empty and force is False."""
    if force:
        return False
    if output_path.exists() and output_path.stat().st_size > 0:
        info(f"skip {what}: {output_path} already exists (use --force to overwrite)")
        return True
    return False


def cmd_translation_qc(args: argparse.Namespace) -> None:
    translation_paths = [Path(path) for path in args.translations]
    if not args.output and not args.output_dir:
        raise ValueError("translation-qc requires --output or --output-dir.")
    if len(translation_paths) > 1 and args.output:
        raise ValueError("Use --output-dir for multi-file QC; --output is only for a single input file.")

    pass_threshold = args.pass_threshold
    fail_threshold = args.fail_threshold
    if pass_threshold is None:
        pass_threshold = 4.0 if args.scorer == "blaser" else 0.72
    if fail_threshold is None:
        fail_threshold = 3.0 if args.scorer == "blaser" else 0.55
    info(f"QC scorer={args.scorer} pass_threshold={pass_threshold} fail_threshold={fail_threshold}")

    pending: list[tuple[Path, Path, list[dict[str, object]]]] = []
    for translation_path in translation_paths:
        group_logs = load_translation_logs(translation_path)
        if args.output_dir:
            output_path = _language_output_path(
                args.output_dir,
                args.output_template,
                _single_target_language(group_logs),
                translation_path,
                model=None,
            )
        else:
            output_path = Path(args.output)
        if _skip_existing(output_path, args.force, f"QC {translation_path.name}"):
            continue
        pending.append((translation_path, output_path, group_logs))
    if not pending:
        info("translation-qc: all outputs already exist; nothing to do.")
        return

    blaser_scorer = None
    if args.scorer == "blaser":
        from thesis_eval.translation.blaser import BlaserScorer

        blaser_scorer = BlaserScorer(device=args.device)

    written: dict[str, str] = {}
    for translation_path, output_path, group_logs in pending:
        with step(f"translation QC {translation_path.name} ({len(group_logs)} records)"):
            qc_logs = apply_translation_qc(
                group_logs,
                pass_threshold=pass_threshold,
                fail_threshold=fail_threshold,
                scorer=args.scorer,
                device=args.device,
                blaser_scorer=blaser_scorer,
            )
        write_jsonl(output_path, qc_logs)
        written[str(output_path)] = str(len(qc_logs))
        print(f"Wrote {len(qc_logs)} QC records to {output_path}")
    if len(written) > 1:
        print(json.dumps(written, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_prompt_roundtrip(args: argparse.Namespace) -> None:
    translation_paths = [Path(path) for path in args.translations]
    if not args.output and not args.output_dir:
        raise ValueError("prompt-roundtrip requires --output or --output-dir.")
    if len(translation_paths) > 1 and args.output:
        raise ValueError("Use --output-dir for multi-file round-trip; --output is only for a single input file.")

    pending: list[tuple[Path, Path, list[dict[str, object]]]] = []
    for translation_path in translation_paths:
        group_logs = load_translation_logs(translation_path)
        if args.output_dir:
            output_path = _language_output_path(
                args.output_dir,
                args.output_template,
                _single_target_language(group_logs),
                translation_path,
                model=None,
            )
        else:
            output_path = Path(args.output)
        if _skip_existing(output_path, args.force, f"round-trip {translation_path.name}"):
            continue
        pending.append((translation_path, output_path, group_logs))
    if not pending:
        info("prompt-roundtrip: all outputs already exist; nothing to do.")
        return

    translator = None
    if args.engine == "nllb" and any(
        str(log["target_language"]) != "eng" for _, _, group_logs in pending for log in group_logs
    ):
        translator = NllbTranslator(checkpoint=args.checkpoint, device=args.device, dtype=args.dtype)

    written: dict[str, str] = {}
    for input_path, output_path, group_logs in pending:
        with step(f"prompt round-trip {input_path}"):
            updated = attach_prompt_roundtrip(
                group_logs,
                engine=args.engine,
                checkpoint=args.checkpoint,
                device=args.device,
                dtype=args.dtype,
                pass_threshold=args.pass_threshold,
                fail_threshold=args.fail_threshold,
                translator=translator,
            )
        write_jsonl(output_path, updated)
        written[str(output_path)] = str(len(updated))
        print(f"Wrote {len(updated)} round-trip records to {output_path}")
    if len(written) > 1:
        print(json.dumps(written, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_prewarm_sonar(args: argparse.Namespace) -> None:
    with step("prewarm SONAR text encoder"):
        result = prewarm_sonar_text_encoder(Path(args.cache_dir) if args.cache_dir else None)
    print(json.dumps(result, indent=2))


def cmd_generate_targets(args: argparse.Namespace) -> None:
    if not args.output and not args.output_dir:
        raise ValueError("generate-targets requires --output or --output-dir.")
    cfg = load_config()
    assets = load_assets()
    profile = resolve_runtime_profile(args.profile) if args.profile else None
    model_cfg = cfg.models[args.model]
    default_backend = "maritaca" if model_cfg.get("access_mode") == "api" else "auto"
    backend = args.backend or (profile.target_backend if profile else default_backend)
    device = args.device or (profile.device if profile else "auto")
    dtype = args.dtype or (profile.dtype if profile else "auto")
    model_ref = _resolve_model_ref(args.model, args.model_path)
    translation_paths = [Path(path) for path in args.translations]
    groups: list[tuple[Path, Path, list[dict[str, Any]]]] = []
    rows: list[dict[str, Any]] = []
    for translation_path in translation_paths:
        translations = load_translation_logs(translation_path)
        group_rows = build_generation_rows(translations, model=args.model, aligned_language=cfg.aligned_language[args.model], benchmark=args.benchmark)
        if args.output_dir:
            output_path = _language_output_path(
                args.output_dir,
                args.output_template,
                _single_attack_language(group_rows),
                translation_path,
                model=args.model,
            )
        else:
            output_path = Path(args.output)
        if _skip_existing(output_path, args.force, f"generate {args.model} <- {translation_path.name}"):
            continue
        groups.append((translation_path, output_path, group_rows))
        rows.extend(group_rows)
    if not groups:
        info("generate-targets: all outputs already exist; nothing to do.")
        return
    info(f"Generating {len(rows)} target responses with backend={backend} model={args.model} model_ref={model_ref}")
    if args.mock_output:
        outputs = [args.mock_output for _ in rows]
    else:
        with step("target model generation"):
            outputs = generate_outputs(
                [str(row["prompt_text"]) for row in rows],
                model_ref,
                backend=backend,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                device=device,
                dtype=dtype,
                trust_remote_code=args.trust_remote_code,
                stream=args.stream,
            )
    enriched = enrich_generation_rows(attach_outputs(rows, outputs), Path(args.distance_csv))
    if not args.skip_tokenizer_metrics and backend != "mock":
        with step("tokenizer metrics"):
            enriched = attach_tokenizer_metrics(enriched, load_tokenizer(model_ref, trust_remote_code=args.trust_remote_code))
    elif args.skip_tokenizer_metrics:
        info("Skipping tokenizer metrics by request")
    else:
        info("Skipping tokenizer metrics for mock generation")
    if args.output_dir:
        written: dict[str, str] = {}
        cursor = 0
        for input_path, output_path, group_rows in groups:
            enriched_slice = enriched[cursor : cursor + len(group_rows)]
            cursor += len(group_rows)
            write_jsonl(output_path, enriched_slice)
            written[str(output_path)] = str(len(enriched_slice))
            print(f"Wrote {len(enriched_slice)} target generations to {output_path}")
        if len(written) > 1:
            print(json.dumps(written, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        write_jsonl(Path(args.output), enriched)
        print(f"Wrote {len(rows)} target generations to {args.output}")


def cmd_attach_tokenizer_metrics(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.generations))
    if not rows:
        write_jsonl(Path(args.output), [])
        print(f"Wrote 0 generation rows with tokenizer metrics to {args.output}")
        return
    row_models = {str(row["model"]) for row in rows}
    if args.model:
        unexpected = row_models.difference({args.model})
        if unexpected:
            raise ValueError(f"--model {args.model!r} does not match generation row models: {sorted(row_models)}")
        models = [args.model]
    else:
        models = sorted(row_models)
    if args.model_path and len(models) != 1:
        raise ValueError("--model-path can only be used when the generation file contains one model or --model is set.")

    output_rows = [dict(row) for row in rows]
    for model in models:
        model_ref = _resolve_model_ref(model, args.model_path if len(models) == 1 else None)
        model_indexes = [index for index, row in enumerate(output_rows) if str(row["model"]) == model]
        model_rows = [output_rows[index] for index in model_indexes]
        info(f"Computing tokenizer metrics for {len(model_rows)} rows model={model} model_ref={model_ref}")
        with step(f"tokenizer metrics for {model}"):
            metric_rows = attach_tokenizer_metrics(
                model_rows,
                load_tokenizer(model_ref, trust_remote_code=args.trust_remote_code),
            )
        for index, metric_row in zip(model_indexes, metric_rows, strict=True):
            output_rows[index] = metric_row

    write_jsonl(Path(args.output), output_rows)
    print(f"Wrote {len(output_rows)} generation rows with tokenizer metrics to {args.output}")


def cmd_backtranslate_responses(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.generations))
    output = attach_response_backtranslations(
        rows,
        engine=args.engine,
        checkpoint=args.checkpoint,
        device=args.device,
        dtype=args.dtype,
    )
    write_jsonl(Path(args.output), output)
    print(f"Wrote {len(output)} generation rows with response back-translations to {args.output}")


def _resolve_model_ref(model: str, model_path: str | None = None) -> str:
    cfg = load_config()
    assets = load_assets()
    model_asset = assets.get(model)
    asset_path = model_asset.local_dir if model_asset and model_asset.local_dir and model_asset.local_dir.exists() else None
    return model_path or (str(asset_path) if asset_path else str(cfg.models[model].get("hf_id") or cfg.models[model]["provider_model_id"]))


def _single_target_language(rows: list[dict[str, object]]) -> str:
    languages = {str(row["target_language"]) for row in rows}
    if len(languages) != 1:
        raise ValueError(f"Expected one target language in output group, found {sorted(languages)}")
    return next(iter(languages))


def _single_attack_language(rows: list[dict[str, Any]]) -> str:
    languages = {str(row["attack_language"]) for row in rows}
    if len(languages) != 1:
        raise ValueError(f"Expected one attack language in output group, found {sorted(languages)}")
    return next(iter(languages))


def _language_output_path(output_dir: str, output_template: str, language: str, input_path: Path, model: str | None) -> Path:
    output_root = ensure_dir(Path(output_dir))
    filename = output_template.format(
        lang=language,
        target_language=language,
        attack_language=language,
        stem=input_path.stem,
        model=model or "",
    )
    return output_root / filename


def cmd_model_status(args: argparse.Namespace) -> None:
    cfg = load_config()
    assets = load_assets()
    asset = assets.get(args.model)
    local_dir = str(asset.local_dir) if asset and asset.local_dir else None
    model_cfg = cfg.models[args.model]
    if model_cfg.get("access_mode") == "api":
        print(
            json.dumps(
                {
                    "model": args.model,
                    "access_mode": "api",
                    "provider": model_cfg.get("provider"),
                    "provider_model_id": model_cfg.get("provider_model_id"),
                    "download_required": False,
                },
                indent=2,
            )
        )
        return
    print(json.dumps(estimate_model_asset_state(str(model_cfg["hf_id"]), local_dir=local_dir), indent=2))


def cmd_score_strongreject(args: argparse.Namespace) -> None:
    generation_paths = [Path(path) for path in args.generations]
    if not args.output and not args.output_dir:
        raise ValueError("score-strongreject requires --output or --output-dir.")
    if len(generation_paths) > 1 and args.output:
        raise ValueError("Use --output-dir for multi-file scoring; --output is only for a single input file.")

    pending: list[tuple[Path, Path, list[dict[str, Any]]]] = []
    for generation_path in generation_paths:
        group_rows = read_jsonl(generation_path)
        if args.output_dir:
            output_dir = ensure_dir(Path(args.output_dir))
            filename = str(args.output_template).format(stem=generation_path.stem)
            output_path = output_dir / filename
        else:
            output_path = Path(args.output)
        if _skip_existing(output_path, args.force, f"score {generation_path.name}"):
            continue
        pending.append((generation_path, output_path, group_rows))
    if not pending:
        info("score-strongreject: all outputs already exist; nothing to do.")
        return

    translator = None
    if args.backtranslate_engine == "nllb" and any(
        str(row.get("attack_language")) != "eng" for _, _, group_rows in pending for row in group_rows
    ):
        translator = NllbTranslator(checkpoint=args.checkpoint, device=args.device, dtype=args.dtype)

    written: dict[str, str] = {}
    for generation_path, output_path, group_rows in pending:
        with step(f"score {generation_path.name} ({len(group_rows)} rows)"):
            scored = score_generation_rows(
                group_rows,
                evaluator=args.evaluator,
                backtranslate_engine=args.backtranslate_engine,
                checkpoint=args.checkpoint,
                device=args.device,
                dtype=args.dtype,
                translator=translator,
            )
        write_jsonl(output_path, scored)
        written[str(output_path)] = str(len(scored))
        print(f"Wrote {len(scored)} StrongREJECT-scored rows to {output_path}")
    if len(written) > 1:
        print(json.dumps(written, ensure_ascii=False, indent=2, sort_keys=True))


def cmd_compute_spec(args: argparse.Namespace) -> None:
    cfg = load_config()
    predictions = read_jsonl(Path(args.predictions))
    rows = compute_if_cons(predictions, cfg.aligned_language)
    write_jsonl(Path(args.output), rows)
    print(f"Wrote {len(rows)} BELEBELE IF/CONS/SPEC rows to {args.output}")


def cmd_score_benign(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.generations))
    scored = score_benign_rows(rows)
    write_jsonl(Path(args.output), scored)
    print(f"Wrote {len(scored)} benign-control scored rows to {args.output}")


def cmd_build_dataset(args: argparse.Namespace) -> None:
    cfg = load_config()
    scored_rows = read_jsonl(Path(args.scored))
    translations = load_translation_logs(Path(args.translations)) if args.translations else None
    spec_rows = read_jsonl(Path(args.spec_scores)) if args.spec_scores else None
    rows = build_frozen_rows(scored_rows, cfg, translations=translations, spec_rows=spec_rows)
    output = Path(args.output)
    if output.name == "dataset_frozen.parquet" and output.exists() and not args.allow_frozen_overwrite:
        raise RuntimeError("Refusing to overwrite dataset_frozen.parquet without --allow-frozen-overwrite")
    if output.suffix == ".parquet":
        import pandas as pd

        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(output, index=False)
    else:
        write_jsonl(output, rows)
    print(f"Wrote {len(rows)} frozen-schema rows to {output}")


def cmd_export_report_tables(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.rows))
    output_dir = ensure_dir(Path(args.output_dir))
    write_csv_dicts(output_dir / "results_coverage.csv", coverage_by_model(rows), ["model", "total_attempts", "api_failure_rate", "provider_block_rate", "scoring_failure_rate", "refusal_rate"])
    write_csv_dicts(output_dir / "asr_by_model_language.csv", asr_by_model_language(rows), ["model", "attack_language", "n", "unsafe", "asr", "ci_low", "ci_high", "mean_strongreject_score", "refusal_rate"])
    write_csv_dicts(output_dir / "crosslingual_asr_by_model_language.csv", asr_by_model_language(rows, exclude_aligned=True), ["model", "attack_language", "n", "unsafe", "asr", "ci_low", "ci_high", "mean_strongreject_score", "refusal_rate"])
    write_csv_dicts(output_dir / "closest_farthest_languages.csv", closest_farthest_by_model(rows), ["model", "closest_language", "closest_distance", "closest_asr", "farthest_language", "farthest_distance", "farthest_asr", "gap"])
    distance_corr, spec_corr = spearman_tables(rows)
    write_csv_dicts(output_dir / "distance_asr_correlation.csv", distance_corr, ["model", "predictor", "n", "spearman_rho", "p_value", "interpretation"])
    write_csv_dicts(output_dir / "spec_asr_correlation.csv", spec_corr, ["model", "predictor", "n", "spearman_rho", "p_value", "interpretation"])
    write_csv_dicts(output_dir / "tokenizer_diagnostics.csv", tokenizer_diagnostics(rows), ["model", "attack_language", "n", "mean_token_inflation", "mean_tokens_per_char", "truncation_risk_rate"])
    write_csv_dicts(output_dir / "glmm_collinearity.csv", collinearity_diagnostics(rows), ["diagnostic", "value"])
    write_csv_dicts(output_dir / "prereg_distance_slope_retention.csv", prereg_distance_slope_retention(rows), ["model_pair_language", "weak_model", "weak_distance_slope", "weak_slope_sign", "weak_n", "strong_model", "strong_distance_slope", "strong_slope_sign", "strong_n", "slope_sign_retained"])
    write_csv_dicts(output_dir / "prereg_falsification_summary.csv", prereg_falsification_summary(rows), ["pairs_evaluated", "pairs_with_slope_sign_retained", "panel_complete", "interpretation"])
    print(f"Wrote report table CSVs to {output_dir}")


def cmd_fit_glmm(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.rows))
    effects = fit_main_glmm(rows)
    write_csv_dicts(Path(args.output), effects, ["predictor", "estimate", "std_error", "odds_ratio", "p_value", "fit_method"])
    print(f"Wrote GLMM main effects to {args.output}")


def cmd_export_audit_queue(args: argparse.Namespace) -> None:
    logs = load_translation_logs(Path(args.translations))
    queue = build_audit_queue(logs, include_passed=args.include_passed)
    write_csv_dicts(Path(args.output), queue, AUDIT_COLUMNS)
    print(f"Wrote {len(queue)} audit rows to {args.output}")


def cmd_import_audit(args: argparse.Namespace) -> None:
    logs = load_translation_logs(Path(args.translations))
    decisions = read_csv_dicts(Path(args.audit_csv))
    updated = apply_audit_decisions(logs, decisions)
    write_jsonl(Path(args.output), updated)
    print(f"Wrote {len(updated)} audit-updated translation logs to {args.output}")


def cmd_calibrate_blaser(args: argparse.Namespace) -> None:
    rows = read_csv_dicts(Path(args.audit_csv)) if args.audit_csv.endswith(".csv") else read_jsonl(Path(args.audit_csv))
    result = calibrate_blaser_thresholds(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2))


def cmd_reference_spbleu(args: argparse.Namespace) -> None:
    logs = load_translation_logs(Path(args.translations))
    references = read_jsonl(Path(args.references))
    updated = attach_reference_scores(logs, references)
    write_jsonl(Path(args.output), updated)
    print(f"Wrote {len(updated)} translation logs with reference scores to {args.output}")


def cmd_run_pilot(args: argparse.Namespace) -> None:
    prompts_path = Path(args.prompts) if args.prompts else None
    if prompts_path is not None and not prompts_path.exists():
        prompts_path = None
    outputs = run_mock_pilot(
        Path(args.output_dir),
        prompts_path=prompts_path,
        prompt_limit=args.prompt_limit,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    print("Pilot produced mock StrongREJECT-scored rows; run real generation/scoring before research use.")


def cmd_export_results(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.rows))
    validate_analysis_rows(rows)
    output = Path(args.output)
    if output.name == "dataset_frozen.parquet" and not args.allow_frozen_overwrite and output.exists():
        raise RuntimeError("Refusing to overwrite dataset_frozen.parquet without --allow-frozen-overwrite")
    if output.suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError("Parquet export requires pandas and a parquet engine.") from exc
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(output, index=False)
    else:
        write_jsonl(output, rows)
    print(f"Wrote {len(rows)} rows to {output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="thesis-eval")
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate-config")
    validate.set_defaults(func=cmd_validate_config)

    runtime = sub.add_parser("runtime-profiles")
    runtime.set_defaults(func=cmd_runtime_profiles)

    list_assets = sub.add_parser("list-assets")
    list_assets.set_defaults(func=cmd_list_assets)

    asset_status_parser = sub.add_parser("asset-status")
    asset_status_parser.add_argument("--name", action="append")
    asset_status_parser.add_argument("--group", action="append")
    asset_status_parser.set_defaults(func=cmd_asset_status)

    download_assets = sub.add_parser("download-assets")
    download_assets.add_argument("--name", action="append")
    download_assets.add_argument("--group", action="append")
    download_assets.add_argument("--dry-run", action="store_true")
    download_assets.add_argument("--force", action="store_true")
    download_assets.set_defaults(func=cmd_download_assets)

    verify_assets_parser = sub.add_parser("verify-assets")
    verify_assets_parser.add_argument("--name", action="append")
    verify_assets_parser.add_argument("--group", action="append")
    verify_assets_parser.add_argument("--deep", action="store_true")
    verify_assets_parser.set_defaults(func=cmd_verify_assets)

    prepare = sub.add_parser("prepare-uriel")
    prepare.add_argument("--source-csv", default=str(THESIS_ROOT / "pictures" / "urielplus_distance_featural.csv"))
    prepare.add_argument("--from-csv", action="store_true", help="Prepare from a static matrix CSV instead of recomputing with URIEL+.")
    prepare.add_argument("--distance-type", default="featural")
    prepare.add_argument("--uriel-backend", choices=["uv", "in-process"], default="uv")
    prepare.add_argument("--uriel-python", help="Explicit Python executable from an environment with urielplus installed.")
    prepare.add_argument("--output-csv", default=str(DATA_DIR / "uriel_plus" / "distance_matrix.csv"))
    prepare.add_argument("--output-tex", default=str(THESIS_ROOT / "aftertext" / "urielplus_distance_table_generated.tex"))
    prepare.add_argument("--run-config-output", default=str(DATA_DIR / "uriel_plus" / "urielplus_run_config.json"))
    prepare.set_defaults(func=cmd_prepare_uriel)

    strongreject = sub.add_parser("import-strongreject")
    strongreject.add_argument("--source", choices=["github", "hf", "local"], default="github")
    strongreject.add_argument("--input")
    strongreject.add_argument("--output", default=str(DATA_DIR / "raw" / "strongreject" / "prompts.jsonl"))
    strongreject.add_argument("--pilot-output", default=str(DATA_DIR / "raw" / "strongreject" / "pilot_prompts.jsonl"))
    strongreject.add_argument("--pilot-size", type=int, default=2)
    strongreject.add_argument("--limit", type=int)
    strongreject.add_argument("--small", action="store_true")
    strongreject.set_defaults(func=cmd_import_strongreject)

    translate = sub.add_parser("translate")
    translate.add_argument("--prompts", required=True)
    translate.add_argument("--target-language", action="append", required=True)
    translate.add_argument("--output")
    translate.add_argument("--output-dir")
    translate.add_argument("--output-template", default="translations_{lang}.jsonl")
    translate.add_argument("--engine", choices=["placeholder", "nllb"], default="placeholder")
    translate.add_argument("--checkpoint", default="facebook/nllb-200-distilled-600M")
    translate.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    translate.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    translate.add_argument("--stream", action="store_true", help="Print each prompt and its translation as it is produced.")
    translate.add_argument("--force", action="store_true", help="Overwrite outputs that already exist.")
    translate.set_defaults(func=cmd_translate)

    qc = sub.add_parser("translation-qc")
    qc.add_argument("--translations", action="append", required=True)
    qc.add_argument("--output")
    qc.add_argument("--output-dir")
    qc.add_argument("--output-template", default="{stem}.qc.jsonl")
    qc.add_argument("--scorer", choices=["heuristic", "blaser"], default="heuristic")
    qc.add_argument("--pass-threshold", type=float)
    qc.add_argument("--fail-threshold", type=float)
    qc.add_argument("--device")
    qc.add_argument("--force", action="store_true", help="Overwrite outputs that already exist.")
    qc.set_defaults(func=cmd_translation_qc)

    roundtrip = sub.add_parser("prompt-roundtrip")
    roundtrip.add_argument("--translations", action="append", required=True)
    roundtrip.add_argument("--output")
    roundtrip.add_argument("--output-dir")
    roundtrip.add_argument("--output-template", default="{stem}.roundtrip.jsonl")
    roundtrip.add_argument("--engine", choices=["placeholder", "nllb"], default="placeholder")
    roundtrip.add_argument("--checkpoint", default="facebook/nllb-200-distilled-600M")
    roundtrip.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    roundtrip.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    roundtrip.add_argument("--pass-threshold", type=float, default=0.60)
    roundtrip.add_argument("--fail-threshold", type=float, default=0.35)
    roundtrip.add_argument("--force", action="store_true", help="Overwrite outputs that already exist.")
    roundtrip.set_defaults(func=cmd_prompt_roundtrip)

    sonar = sub.add_parser("prewarm-sonar")
    sonar.add_argument("--cache-dir")
    sonar.set_defaults(func=cmd_prewarm_sonar)

    gen = sub.add_parser("generate-targets")
    gen.add_argument("--translations", action="append", required=True)
    gen.add_argument("--model", default="sagui_7b")
    gen.add_argument("--benchmark", default="strongreject")
    gen.add_argument("--output")
    gen.add_argument("--output-dir")
    gen.add_argument("--output-template", default="target_generations_{model}_{lang}.jsonl")
    gen.add_argument("--profile", choices=["macbook", "rtx_5070_ti"])
    gen.add_argument("--backend", choices=["auto", "mock", "vllm", "transformers", "maritaca"])
    gen.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    gen.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"])
    gen.add_argument("--mock-output")
    gen.add_argument("--max-tokens", type=int, default=512)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--trust-remote-code", action="store_true")
    gen.add_argument("--model-path")
    gen.add_argument("--stream", action="store_true")
    gen.add_argument("--skip-tokenizer-metrics", action="store_true")
    gen.add_argument("--distance-csv", default=str(DATA_DIR / "uriel_plus" / "distance_matrix.csv"))
    gen.add_argument("--force", action="store_true", help="Overwrite outputs that already exist.")
    gen.set_defaults(func=cmd_generate_targets)

    token_metrics = sub.add_parser("attach-tokenizer-metrics")
    token_metrics.add_argument("--generations", required=True)
    token_metrics.add_argument("--output", required=True)
    token_metrics.add_argument("--model")
    token_metrics.add_argument("--model-path")
    token_metrics.add_argument("--trust-remote-code", action="store_true")
    token_metrics.set_defaults(func=cmd_attach_tokenizer_metrics)

    response_bt = sub.add_parser("backtranslate-responses")
    response_bt.add_argument("--generations", required=True)
    response_bt.add_argument("--output", required=True)
    response_bt.add_argument("--engine", choices=["placeholder", "nllb"], default="placeholder")
    response_bt.add_argument("--checkpoint", default="facebook/nllb-200-distilled-600M")
    response_bt.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    response_bt.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    response_bt.set_defaults(func=cmd_backtranslate_responses)

    model_status = sub.add_parser("model-status")
    model_status.add_argument("--model", default="sagui_7b")
    model_status.set_defaults(func=cmd_model_status)

    sr_score = sub.add_parser("score-strongreject")
    sr_score.add_argument("--generations", action="append", required=True)
    sr_score.add_argument("--output")
    sr_score.add_argument("--output-dir")
    sr_score.add_argument("--output-template", default="{stem}.strongreject.jsonl")
    sr_score.add_argument("--evaluator", default="strongreject_rubric")
    sr_score.add_argument("--backtranslate-engine", choices=["placeholder", "nllb"], default="placeholder")
    sr_score.add_argument("--checkpoint", default="facebook/nllb-200-distilled-600M")
    sr_score.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    sr_score.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    sr_score.add_argument("--force", action="store_true", help="Overwrite outputs that already exist.")
    sr_score.set_defaults(func=cmd_score_strongreject)

    spec = sub.add_parser("compute-spec")
    spec.add_argument("--predictions", required=True)
    spec.add_argument("--output", required=True)
    spec.set_defaults(func=cmd_compute_spec)

    benign = sub.add_parser("score-benign-controls")
    benign.add_argument("--generations", required=True)
    benign.add_argument("--output", required=True)
    benign.set_defaults(func=cmd_score_benign)

    dataset = sub.add_parser("build-dataset")
    dataset.add_argument("--scored", required=True)
    dataset.add_argument("--output", required=True)
    dataset.add_argument("--translations")
    dataset.add_argument("--spec-scores")
    dataset.add_argument("--allow-frozen-overwrite", action="store_true")
    dataset.set_defaults(func=cmd_build_dataset)

    reports = sub.add_parser("export-report-tables")
    reports.add_argument("--rows", required=True)
    reports.add_argument("--output-dir", required=True)
    reports.set_defaults(func=cmd_export_report_tables)

    glmm = sub.add_parser("fit-glmm")
    glmm.add_argument("--rows", required=True)
    glmm.add_argument("--output", required=True)
    glmm.set_defaults(func=cmd_fit_glmm)

    audit_export = sub.add_parser("export-audit-queue")
    audit_export.add_argument("--translations", required=True)
    audit_export.add_argument("--output", required=True)
    audit_export.add_argument("--include-passed", action="store_true")
    audit_export.set_defaults(func=cmd_export_audit_queue)

    audit_import = sub.add_parser("import-audit")
    audit_import.add_argument("--translations", required=True)
    audit_import.add_argument("--audit-csv", required=True)
    audit_import.add_argument("--output", required=True)
    audit_import.set_defaults(func=cmd_import_audit)

    calibrate = sub.add_parser("calibrate-blaser")
    calibrate.add_argument("--audit-csv", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.set_defaults(func=cmd_calibrate_blaser)

    spbleu = sub.add_parser("reference-spbleu")
    spbleu.add_argument("--translations", required=True)
    spbleu.add_argument("--references", required=True)
    spbleu.add_argument("--output", required=True)
    spbleu.set_defaults(func=cmd_reference_spbleu)

    pilot = sub.add_parser("run-pilot")
    pilot.add_argument("--output-dir", default=str(OUTPUT_DIR / "pilot"))
    pilot.add_argument("--prompts", default=str(DATA_DIR / "raw" / "strongreject" / "pilot_prompts.jsonl"))
    pilot.add_argument("--prompt-limit", type=int, default=2)
    pilot.set_defaults(func=cmd_run_pilot)

    export = sub.add_parser("export-results")
    export.add_argument("--rows", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--allow-frozen-overwrite", action="store_true")
    export.set_defaults(func=cmd_export_results)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
