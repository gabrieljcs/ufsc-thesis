from __future__ import annotations

import argparse
import json
from pathlib import Path

from thesis_eval.assets import asset_status, download_asset, load_assets, select_assets, verify_assets
from thesis_eval.benchmarks.strongreject import import_strongreject, write_prompt_records
from thesis_eval.benchmarks.uriel import compare_matrices, load_distance_matrix, subset_matrix, write_distance_matrix, write_latex_table
from thesis_eval.config import load_config
from thesis_eval.evaluation.api_batches import (
    anthropic_batch_body,
    hosted_judge_rows,
    normalize_hosted_results,
    openai_batch_rows,
    parse_anthropic_batch_results,
    parse_openai_batch_results,
)
from thesis_eval.evaluation.judge_batch import build_judge_requests, combine_judge_results
from thesis_eval.io import read_csv_dicts, read_jsonl, write_csv_dicts, write_jsonl
from thesis_eval.models.generation import attach_outputs, build_generation_rows, estimate_model_asset_state, generate_outputs
from thesis_eval.paths import DATA_DIR, OUTPUT_DIR, THESIS_ROOT, ensure_dir
from thesis_eval.pipeline import enrich_generation_rows, run_mock_pilot
from thesis_eval.progress import info, step
from thesis_eval.runtime import load_runtime_profiles, resolve_runtime_profile
from thesis_eval.translation.sonar import prewarm_sonar_text_encoder
from thesis_eval.schema import validate_analysis_rows
from thesis_eval.translation.audit import AUDIT_COLUMNS, apply_audit_decisions, build_audit_queue
from thesis_eval.translation.log import load_translation_logs, save_translation_logs
from thesis_eval.translation.pipeline import NllbTranslator, PromptRecord, apply_translation_qc, run_translation
from thesis_eval.translation.reference_spbleu import attach_reference_scores


def cmd_validate_config(_: argparse.Namespace) -> None:
    cfg = load_config()
    print(f"OK: {len(cfg.models)} models, {len(cfg.attack_languages)} languages, {len(cfg.judges)} judges")


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
    source = Path(args.source_csv)
    matrix = subset_matrix(load_distance_matrix(source), cfg.attack_languages)
    output_csv = Path(args.output_csv)
    output_tex = Path(args.output_tex)
    write_distance_matrix(output_csv, matrix, cfg.attack_languages)
    write_latex_table(output_tex, matrix, cfg.attack_languages)
    comparison = compare_matrices(matrix, load_distance_matrix(source))
    print(json.dumps({"csv": str(output_csv), "latex": str(output_tex), **comparison}, indent=2))


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

    translator = None
    if args.engine == "nllb" and any(language != "eng" for language in target_languages):
        translator = NllbTranslator(checkpoint=args.checkpoint, device=args.device, dtype=args.dtype)

    written: dict[str, str] = {}
    for target_language in target_languages:
        output_path = _translation_output_path(args, target_language)
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


def cmd_translation_qc(args: argparse.Namespace) -> None:
    logs = load_translation_logs(Path(args.translations))
    pass_threshold = args.pass_threshold
    fail_threshold = args.fail_threshold
    if pass_threshold is None:
        pass_threshold = 4.0 if args.scorer == "blaser" else 0.72
    if fail_threshold is None:
        fail_threshold = 3.0 if args.scorer == "blaser" else 0.55
    info(f"QC scorer={args.scorer} pass_threshold={pass_threshold} fail_threshold={fail_threshold}")
    with step(f"translation QC for {len(logs)} records"):
        qc_logs = apply_translation_qc(
            logs,
            pass_threshold=pass_threshold,
            fail_threshold=fail_threshold,
            scorer=args.scorer,
            device=args.device,
        )
    write_jsonl(Path(args.output), qc_logs)
    print(f"Wrote {len(qc_logs)} QC records to {args.output}")


def cmd_prewarm_sonar(args: argparse.Namespace) -> None:
    with step("prewarm SONAR text encoder"):
        result = prewarm_sonar_text_encoder(Path(args.cache_dir) if args.cache_dir else None)
    print(json.dumps(result, indent=2))


def cmd_generate_targets(args: argparse.Namespace) -> None:
    cfg = load_config()
    assets = load_assets()
    profile = resolve_runtime_profile(args.profile) if args.profile else None
    backend = args.backend or (profile.target_backend if profile else "auto")
    device = args.device or (profile.device if profile else "auto")
    dtype = args.dtype or (profile.dtype if profile else "auto")
    model_asset = assets.get(args.model)
    asset_path = model_asset.local_dir if model_asset and model_asset.local_dir and model_asset.local_dir.exists() else None
    model_ref = args.model_path or (str(asset_path) if asset_path else cfg.models[args.model]["hf_id"])
    translations = load_translation_logs(Path(args.translations))
    rows = build_generation_rows(translations, model=args.model, aligned_language=cfg.aligned_language[args.model], benchmark=args.benchmark)
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
    write_jsonl(Path(args.output), enriched)
    print(f"Wrote {len(rows)} target generations to {args.output}")


def cmd_model_status(args: argparse.Namespace) -> None:
    cfg = load_config()
    assets = load_assets()
    asset = assets.get(args.model)
    local_dir = str(asset.local_dir) if asset and asset.local_dir else None
    print(json.dumps(estimate_model_asset_state(cfg.models[args.model]["hf_id"], local_dir=local_dir), indent=2))


def cmd_export_judge_batch(args: argparse.Namespace) -> None:
    cfg = load_config()
    generations = read_jsonl(Path(args.generations))
    rows = build_judge_requests(generations, list(cfg.judges))
    info(f"Exporting {len(rows)} judge requests from {len(generations)} generations")
    write_jsonl(Path(args.output), rows)
    print(f"Wrote {len(rows)} judge requests to {args.output}")


def _judge_model_id(cfg: dict[str, dict[str, str]], judge: str) -> str:
    judge_cfg = cfg[judge]
    return judge_cfg.get("model_id") or judge_cfg.get("hf_id") or judge


def cmd_export_api_batches(args: argparse.Namespace) -> None:
    cfg = load_config()
    requests = read_jsonl(Path(args.requests))
    by_judge = {judge: [row for row in requests if row["judge"] == judge] for judge in cfg.judges}
    output_dir = ensure_dir(Path(args.output_dir))
    info(f"Exporting provider batches to {output_dir}")

    openai_rows = openai_batch_rows(by_judge["gpt_4o"], _judge_model_id(cfg.judges, "gpt_4o"), max_tokens=args.max_tokens)
    openai_path = output_dir / "openai_gpt_4o_batch.jsonl"
    write_jsonl(openai_path, openai_rows)

    anthropic_body = anthropic_batch_body(
        by_judge["claude_opus_4"],
        _judge_model_id(cfg.judges, "claude_opus_4"),
        max_tokens=args.max_tokens,
    )
    anthropic_path = output_dir / "anthropic_claude_opus_4_batch.json"
    anthropic_path.write_text(json.dumps(anthropic_body, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    llama_rows = hosted_judge_rows(by_judge["llama_guard_4"], _judge_model_id(cfg.judges, "llama_guard_4"))
    llama_path = output_dir / "hosted_llama_guard_4_batch.jsonl"
    write_jsonl(llama_path, llama_rows)

    manifest = {
        "openai_gpt_4o": str(openai_path),
        "anthropic_claude_opus_4": str(anthropic_path),
        "hosted_llama_guard_4": str(llama_path),
        "note": "Use provider APIs to submit these files; normalize downloaded results before majority vote.",
    }
    manifest_path = output_dir / "api_batch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def cmd_normalize_judge_results(args: argparse.Namespace) -> None:
    rows = read_jsonl(Path(args.input))
    if args.provider == "openai":
        normalized = parse_openai_batch_results(rows)
    elif args.provider == "anthropic":
        normalized = parse_anthropic_batch_results(rows)
    else:
        normalized = normalize_hosted_results(rows)
    write_jsonl(Path(args.output), normalized)
    print(f"Wrote {len(normalized)} normalized judge rows to {args.output}")


def cmd_import_judge_batch(args: argparse.Namespace) -> None:
    generations = read_jsonl(Path(args.generations))
    results = read_jsonl(Path(args.judge_results))
    combined = combine_judge_results(generations, results)
    write_jsonl(Path(args.output), combined)
    print(f"Wrote {len(combined)} majority-voted rows to {args.output}")


def cmd_majority_vote(args: argparse.Namespace) -> None:
    generations = read_jsonl(Path(args.generations))
    results = read_jsonl(Path(args.judge_results))
    combined = combine_judge_results(generations, results)
    validate_analysis_rows(combined)
    write_jsonl(Path(args.output), combined)
    print(f"Wrote validated analysis rows to {args.output}")


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
    print("Pilot exported judge requests only; batch API judge results must be imported before majority vote.")


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
    prepare.add_argument("--output-csv", default=str(DATA_DIR / "uriel_plus" / "distance_matrix.csv"))
    prepare.add_argument("--output-tex", default=str(THESIS_ROOT / "aftertext" / "urielplus_distance_table_generated.tex"))
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
    translate.set_defaults(func=cmd_translate)

    qc = sub.add_parser("translation-qc")
    qc.add_argument("--translations", required=True)
    qc.add_argument("--output", required=True)
    qc.add_argument("--scorer", choices=["heuristic", "blaser"], default="heuristic")
    qc.add_argument("--pass-threshold", type=float)
    qc.add_argument("--fail-threshold", type=float)
    qc.add_argument("--device")
    qc.set_defaults(func=cmd_translation_qc)

    sonar = sub.add_parser("prewarm-sonar")
    sonar.add_argument("--cache-dir")
    sonar.set_defaults(func=cmd_prewarm_sonar)

    gen = sub.add_parser("generate-targets")
    gen.add_argument("--translations", required=True)
    gen.add_argument("--model", default="sabia_7b")
    gen.add_argument("--benchmark", default="strongreject")
    gen.add_argument("--output", required=True)
    gen.add_argument("--profile", choices=["macbook", "rtx_5070_ti"])
    gen.add_argument("--backend", choices=["auto", "mock", "vllm", "transformers"])
    gen.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"])
    gen.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"])
    gen.add_argument("--mock-output")
    gen.add_argument("--max-tokens", type=int, default=512)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--trust-remote-code", action="store_true")
    gen.add_argument("--model-path")
    gen.add_argument("--stream", action="store_true")
    gen.add_argument("--distance-csv", default=str(DATA_DIR / "uriel_plus" / "distance_matrix.csv"))
    gen.set_defaults(func=cmd_generate_targets)

    model_status = sub.add_parser("model-status")
    model_status.add_argument("--model", default="sabia_7b")
    model_status.set_defaults(func=cmd_model_status)

    export_batch = sub.add_parser("export-judge-batch")
    export_batch.add_argument("--generations", required=True)
    export_batch.add_argument("--output", required=True)
    export_batch.set_defaults(func=cmd_export_judge_batch)

    export_api = sub.add_parser("export-api-batches")
    export_api.add_argument("--requests", required=True)
    export_api.add_argument("--output-dir", required=True)
    export_api.add_argument("--max-tokens", type=int, default=8)
    export_api.set_defaults(func=cmd_export_api_batches)

    normalize = sub.add_parser("normalize-judge-results")
    normalize.add_argument("--provider", choices=["openai", "anthropic", "hosted"], required=True)
    normalize.add_argument("--input", required=True)
    normalize.add_argument("--output", required=True)
    normalize.set_defaults(func=cmd_normalize_judge_results)

    import_batch = sub.add_parser("import-judge-batch")
    import_batch.add_argument("--generations", required=True)
    import_batch.add_argument("--judge-results", required=True)
    import_batch.add_argument("--output", required=True)
    import_batch.set_defaults(func=cmd_import_judge_batch)

    vote = sub.add_parser("majority-vote")
    vote.add_argument("--generations", required=True)
    vote.add_argument("--judge-results", required=True)
    vote.add_argument("--output", required=True)
    vote.set_defaults(func=cmd_majority_vote)

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
