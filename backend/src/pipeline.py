"""
Full pipeline orchestrator — runs all stages end-to-end.
Called by the `run-full` CLI command.
"""
from __future__ import annotations

import shutil
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import load_config, get_system_info, ProjectConfig
from src.ingest.loader import load_dataset
from src.ingest.snapshot import create_snapshot
from src.preprocessing.pipeline import save_preprocessed
from src.preprocessing.spec import DEFAULT_PREPROCESS_SPEC, apply_preprocess
from src.schema import resolve_roles
from src.storage.model_signature import build_input_signature
from src.psychometrics.runner import run_psychometrics
from src.splits.splitter import stratified_split, validate_split_no_leakage
from src.text_tasks.trainer import train_all_baselines
from src.training.config import TrainingConfig
from src.training.runner import run_training, TASK_LABEL_COLS
from src.fusion.runner import run_fusion
from src.contradiction.runner import run_contradiction_monitoring
from src.evaluation.runner import run_evaluation
from src.reporting.evaluation_report import generate_evaluation_report
from src.reporting.model_card import generate_all_model_cards
from src.reporting.data_dictionary import generate_data_dictionary
from src.utils.run_manager import RunManager
from src.utils.reproducibility import set_all_seeds, collect_library_versions, hash_file
from src.utils.logging import get_logger
from src.storage.database import Database
from src.storage.dataset_manager import DatasetManager
from src.storage.model_registry import ModelRegistry
import orjson

log = get_logger(__name__)


def _copy_pipeline_model_artifact(
    source_artifact_path: Path,
    destination_dir: Path,
) -> Path:
    """Mirror a registered training artifact into the pipeline run layout."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    if source_artifact_path.is_dir():
        destination_path = destination_dir / "model"
        if destination_path.exists():
            shutil.rmtree(destination_path)
        shutil.copytree(source_artifact_path, destination_path)
        return destination_path

    destination_path = destination_dir / "model.joblib"
    shutil.copy2(source_artifact_path, destination_path)
    return destination_path


def _train_pipeline_tasks_via_runner(
    *,
    run_id: str,
    run_dir: Path,
    data_path: Path | None,
    dataset_id: str | None,
    dataset_version: int | None,
    branch_id: str | None,
    seed: int,
    dataset_manager: DatasetManager | None,
    model_registry: ModelRegistry,
    pipeline_training: dict[str, Any],
) -> list[str]:
    """Train all text tasks with the shared training runner and mirror outputs."""
    model_type = str(pipeline_training["model_type"])
    config = TrainingConfig(**dict(pipeline_training.get("config") or {}))
    training_artifacts_dir = run_dir / "_training_runs"
    resolved_data_path: str | None = None

    if data_path is not None:
        resolved_data_path = str(data_path)
    elif dataset_id is None or dataset_manager is None:
        training_data_path = run_dir / "raw" / "pipeline_training_input.csv"
        raw_snapshot_path = run_dir / "raw" / "dataset_snapshot.parquet"
        if not raw_snapshot_path.exists():
            raise ValueError(
                "Pipeline training requires either data_path, dataset_id, or a saved raw snapshot."
            )
        pd.read_parquet(raw_snapshot_path).to_csv(training_data_path, index=False)
        resolved_data_path = str(training_data_path)

    registered_model_ids: list[str] = []
    for task_name in TASK_LABEL_COLS:
        result = run_training(
            dataset_id=dataset_id,
            data_path=resolved_data_path,
            task=task_name,
            model_type=model_type,
            dataset_manager=dataset_manager,
            model_registry=model_registry,
            artifacts_dir=training_artifacts_dir,
            config=config,
            dataset_version=dataset_version,
            branch_id=branch_id,
            seed=seed,
            name=f"pipeline_{task_name}_{model_type}_{run_id[:12]}",
            producer_run_id=run_id,
        )
        registered_model_ids.append(result["model_id"])

        artifact_path = model_registry.load_model_artifact(result["model_id"])
        task_dir = run_dir / "text_tasks" / task_name / model_type
        _copy_pipeline_model_artifact(artifact_path, task_dir)
        (task_dir / "metrics.json").write_bytes(
            orjson.dumps(result["metrics"], option=orjson.OPT_INDENT_2)
        )

    return registered_model_ids


def run_full_pipeline(
    data_path: Path | None = None,
    config_path: Path = None,
    factor_structure_path: Path = None,
    runs_dir: Path = None,
    seed: int = 42,
    existing_run_id: str | None = None,
    df_raw: pd.DataFrame | None = None,
    dataset_id: str | None = None,
    dataset_version: int | None = None,
    branch_id: str | None = None,
    dataset_manager: DatasetManager | None = None,
    pipeline_training: dict[str, Any] | None = None,
    model_registry: ModelRegistry | None = None,
    db: Database | None = None,
) -> str:
    """
    Execute the full analysis pipeline:
    1. Load and snapshot dataset
    2. Preprocess text
    3. Run psychometrics (CFA)
    4. Split data
    5. Train text classification baselines
    6. Run fusion experiments
    7. Run contradiction monitoring
    8. Evaluate all models
    9. Generate reports

    Either ``data_path`` (file on disk) or ``df_raw`` (pre-loaded DataFrame) must
    be provided.  When ``df_raw`` is given the snapshot_id is derived from its
    content hash instead of the file hash.

    Returns the run_id of the created run.
    """
    import hashlib

    if df_raw is None and data_path is None:
        raise ValueError("Either data_path or df_raw must be provided")

    set_all_seeds(seed)
    system_info = get_system_info()
    system_info.update(collect_library_versions())

    # --- Setup run ---
    from src.utils.reproducibility import hash_file
    config_hash = hash_file(config_path)[:16] if config_path and config_path.exists() else "unknown"
    run_mgr = RunManager(runs_dir)

    if df_raw is None:
        log.info("pipeline_start", data_path=str(data_path))
        df_raw = load_dataset(data_path)
        snapshot_id = hash_file(data_path)[:16]
    else:
        log.info("pipeline_start", dataset_id=dataset_id, dataset_version=dataset_version)
        snapshot_id = hashlib.sha256(df_raw.to_csv(index=False).encode()).hexdigest()[:16]

    if existing_run_id is not None:
        # Reuse the run already created by the API — do not create a duplicate
        run_id = existing_run_id
        run_dir = run_mgr.get_run_dir(run_id)
        log.info("pipeline_reusing_run", run_id=run_id)
    else:
        run_id = run_mgr.create_run(
            config_hash=config_hash,
            data_snapshot_id=snapshot_id,
            random_seed=seed,
            system_info=system_info,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            branch_id=branch_id,
            db=db,
        )
        run_dir = run_mgr.get_run_dir(run_id)
        log.info("run_created", run_id=run_id)

    # Save config to run dir (best-effort)
    import shutil as _shutil
    if config_path.exists():
        _shutil.copy2(config_path, run_dir / "experiment.yaml")
    if factor_structure_path.exists():
        _shutil.copy2(factor_structure_path, run_dir / "factor_structure.yaml")

    registered_model_ids: list[str] = []
    try:
        # --- Stage 1: Ingest & Preprocess ---
        stage = "ingest_preprocess"
        started = datetime.now(timezone.utc).isoformat()
        run_mgr.start_stage(run_id, stage)
        try:
            if data_path is not None:
                snapshot_id = create_snapshot(data_path, run_dir)
            else:
                # Always persist a deterministic raw snapshot for df_raw inputs.
                raw_dir = run_dir / "raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                raw_path = raw_dir / "dataset_snapshot.parquet"
                df_raw.to_parquet(raw_path, index=False)
                snapshot_id = hashlib.sha256(
                    df_raw.to_csv(index=False).encode()
                ).hexdigest()[:16]
                (raw_dir / "snapshot_metadata.json").write_bytes(
                    orjson.dumps(
                        {
                            "source": "dataframe_direct",
                            "dataset_id": dataset_id,
                            "dataset_version": dataset_version,
                            "sha256_prefix": snapshot_id,
                        },
                        option=orjson.OPT_INDENT_2,
                    )
                )
            df = apply_preprocess(df_raw, text_col="text_feedback")
            preprocessed_path = save_preprocessed(df, run_dir)
            run_mgr.register_artifact(run_id, "preprocessed_data", preprocessed_path, "data", stage)
            run_mgr.complete_stage(run_id, stage, started)
        except Exception:
            run_mgr.fail_stage(run_id, stage, traceback.format_exc())
            raise

        # --- Stage 2: Psychometrics ---
        stage = "psychometrics"
        started = datetime.now(timezone.utc).isoformat()
        run_mgr.start_stage(run_id, stage)
        try:
            cfa_result = run_psychometrics(df, factor_structure_path, run_dir)
            run_mgr.register_artifact(run_id, "factor_scores", run_dir / "psychometrics" / "factor_scores.csv", "data", stage)
            run_mgr.complete_stage(run_id, stage, started)
        except Exception:
            run_mgr.fail_stage(run_id, stage, traceback.format_exc())
            raise

        # --- Stage 3: Split ---
        stage = "splits"
        started = datetime.now(timezone.utc).isoformat()
        run_mgr.start_stage(run_id, stage)
        try:
            df_train, df_val, df_test = stratified_split(
                df, stratify_col="sentiment_class", seed=seed
            )
            validate_split_no_leakage(df_train, df_val, df_test)

            splits_dir = run_dir / "splits"
            splits_dir.mkdir(exist_ok=True)
            df_train.to_parquet(splits_dir / "train.parquet", index=False)
            df_val.to_parquet(splits_dir / "val.parquet", index=False)
            df_test.to_parquet(splits_dir / "test.parquet", index=False)

            # Save split info
            split_info = {
                "n_train": len(df_train), "n_val": len(df_val), "n_test": len(df_test),
                "strategy": "stratified_by_sentiment_class",
                "seed": seed,
            }
            (splits_dir / "split_info.json").write_bytes(orjson.dumps(split_info, option=orjson.OPT_INDENT_2))

            run_mgr.complete_stage(run_id, stage, started)
        except Exception:
            run_mgr.fail_stage(run_id, stage, traceback.format_exc())
            raise

        # Align factor scores to splits
        factor_scores_train = cfa_result.factor_scores.iloc[df_train.index]
        factor_scores_val = cfa_result.factor_scores.iloc[df_val.index]
        factor_scores_test = cfa_result.factor_scores.iloc[df_test.index]

        # --- Stage 4: Text Classification ---
        stage = "text_tasks"
        started = datetime.now(timezone.utc).isoformat()
        run_mgr.start_stage(run_id, stage)
        try:
            text_col = "text_processed" if "text_processed" in df_train.columns else "text_feedback"
            source_text_col = "text_feedback" if "text_feedback" in df_train.columns else text_col
            resolved_columns = resolve_roles(df=df_train, column_roles={}, overrides={})
            if pipeline_training is not None:
                if model_registry is None:
                    raise ValueError(
                        "pipeline_training requires model_registry so models can be registered."
                    )
                registered_model_ids.extend(
                    _train_pipeline_tasks_via_runner(
                        run_id=run_id,
                        run_dir=run_dir,
                        data_path=data_path,
                        dataset_id=dataset_id,
                        dataset_version=dataset_version,
                        branch_id=branch_id,
                        seed=seed,
                        dataset_manager=dataset_manager,
                        model_registry=model_registry,
                        pipeline_training=pipeline_training,
                    )
                )
            else:
                task_results = train_all_baselines(
                    df_train, df_val, run_dir, seed=seed, text_col=text_col
                )
                if model_registry is not None:
                    for task_name, models_for_task in task_results.items():
                        for model_type, cls_result in models_for_task.items():
                            metrics_path = run_dir / "text_tasks" / task_name / model_type / "metrics.json"
                            try:
                                model_meta = model_registry.register_model(
                                    name=f"pipeline_{task_name}_{model_type}_{run_id[:12]}",
                                    task=task_name,
                                    model_type=model_type,
                                    source_model_path=cls_result.model_path,
                                    source_metrics_path=metrics_path if metrics_path.exists() else None,
                                    dataset_id=dataset_id,
                                    dataset_version=dataset_version,
                                    config=cls_result.hyperparameters,
                                    run_id=run_id,
                                    input_signature=build_input_signature(
                                        task=task_name,
                                        resolved_columns=resolved_columns,
                                        source_text_col=source_text_col,
                                        model_input_col=text_col,
                                        label_col=resolved_columns.label_col_by_task.get(task_name, task_name),
                                        preprocess_spec=DEFAULT_PREPROCESS_SPEC,
                                        classes=[str(value) for value in cls_result.classes],
                                    ),
                                    preprocess_spec=DEFAULT_PREPROCESS_SPEC.as_dict(),
                                )
                                registered_model_ids.append(model_meta.id)
                            except Exception:
                                log.error(
                                    "pipeline_model_register_failed",
                                    run_id=run_id,
                                    task=task_name,
                                    model_type=model_type,
                                    error=traceback.format_exc(),
                                )
            run_mgr.complete_stage(run_id, stage, started)
        except Exception:
            run_mgr.fail_stage(run_id, stage, traceback.format_exc())
            raise

        # --- Stage 5: Fusion ---
        stage = "fusion"
        started = datetime.now(timezone.utc).isoformat()
        run_mgr.start_stage(run_id, stage)
        try:
            fusion_results = run_fusion(
                train_df=df_train,
                test_df=df_test,
                factor_scores_train=factor_scores_train,
                factor_scores_test=factor_scores_test,
                run_dir=run_dir,
                seed=seed,
                text_col=text_col,
            )
            run_mgr.complete_stage(run_id, stage, started)
        except Exception:
            run_mgr.fail_stage(run_id, stage, traceback.format_exc())
            raise

        # --- Stage 6: Contradiction Monitoring ---
        stage = "contradiction"
        started = datetime.now(timezone.utc).isoformat()
        run_mgr.start_stage(run_id, stage)
        try:
            run_contradiction_monitoring(
                df=df_test,
                factor_scores=factor_scores_test,
                run_dir=run_dir,
            )
            run_mgr.complete_stage(run_id, stage, started)
        except Exception:
            run_mgr.fail_stage(run_id, stage, traceback.format_exc())
            raise

        # --- Stage 7: Evaluation ---
        stage = "evaluation"
        started = datetime.now(timezone.utc).isoformat()
        run_mgr.start_stage(run_id, stage)
        try:
            eval_results = run_evaluation(df_test, run_dir, text_col=text_col)
            # Link pipeline evaluation into analysis history.
            if db is not None and dataset_id is not None:
                analysis_id = f"analysis_{uuid.uuid4().hex[:16]}"
                db.execute(
                    """INSERT INTO analysis_runs
                    (id, name, description, tags, comments, dataset_id, dataset_version, model_ids,
                     created_at, status, pipeline_run_id, result_summary, branch_id)
                    VALUES (?, ?, ?, ?, '', ?, ?, ?, ?, 'completed', ?, ?, ?)""",
                    (
                        analysis_id,
                        f"Pipeline Evaluation {run_id[:12]}",
                        "Auto-generated evaluation record from full pipeline run.",
                        orjson.dumps(["pipeline", "evaluation", "auto"]).decode(),
                        dataset_id,
                        dataset_version,
                        orjson.dumps(registered_model_ids).decode(),
                        datetime.now(timezone.utc).isoformat(),
                        run_id,
                        orjson.dumps(
                            {
                                "source": "pipeline_evaluation",
                                "run_id": run_id,
                                "evaluation": eval_results,
                                "registered_model_ids": registered_model_ids,
                            }
                        ).decode(),
                        branch_id,
                    ),
                )
                for model_id in registered_model_ids:
                    db.execute(
                        "INSERT OR IGNORE INTO analysis_model_refs (analysis_id, model_id) VALUES (?, ?)",
                        (analysis_id, model_id),
                    )
                db.commit()
            run_mgr.complete_stage(run_id, stage, started)
        except Exception:
            run_mgr.fail_stage(run_id, stage, traceback.format_exc())
            raise

        # --- Stage 8: Reports ---
        stage = "reporting"
        started = datetime.now(timezone.utc).isoformat()
        run_mgr.start_stage(run_id, stage)
        try:
            report_path = generate_evaluation_report(run_dir)
            model_card_paths = generate_all_model_cards(run_dir)
            dict_path = generate_data_dictionary(run_dir)
            run_mgr.register_artifact(run_id, "evaluation_report", report_path, "report", stage)
            for p in model_card_paths:
                run_mgr.register_artifact(run_id, f"model_card_{p.stem}", p, "model_card", stage)
            run_mgr.complete_stage(run_id, stage, started)
        except Exception:
            run_mgr.fail_stage(run_id, stage, traceback.format_exc())
            raise

        run_mgr.complete_run(run_id, db=db)
        log.info("pipeline_complete", run_id=run_id, run_dir=str(run_dir))
        return run_id
    except Exception:
        run_mgr.fail_run(run_id, db=db)
        raise
