"""
CLI entry point for the Student Feedback Analysis Platform.
Usage: uv run python -m src.cli <command> [options]
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

# Ensure backend/src is on sys.path when running as __main__
_BACKEND = Path(__file__).parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from src.utils.logging import configure_logging, get_logger

_DEFAULT_DATA = "../mnt/data/dataset.csv"
_DEFAULT_CONFIG = "configs/experiment.yaml"
_DEFAULT_FACTOR_CONFIG = "configs/factor_structure.yaml"
_DEFAULT_RUNS_DIR = "runs"


@click.group()
@click.option("--log-level", default="INFO", help="Logging level")
def main(log_level: str) -> None:
    """Student Feedback Analysis Platform CLI."""
    configure_logging(log_level)


# ---------------------------------------------------------------------------
# run-full
# ---------------------------------------------------------------------------

@main.command("run-full")
@click.option("--data", default=_DEFAULT_DATA, type=click.Path(), help="Path to dataset CSV")
@click.option("--config", default=_DEFAULT_CONFIG, type=click.Path(), help="Path to experiment.yaml")
@click.option("--factor-config", default=_DEFAULT_FACTOR_CONFIG, type=click.Path())
@click.option("--runs-dir", default=_DEFAULT_RUNS_DIR, type=click.Path())
@click.option("--seed", default=42, type=int)
def cmd_run_full(data: str, config: str, factor_config: str, runs_dir: str, seed: int) -> None:
    """Run the full analysis pipeline end-to-end."""
    from src.pipeline import run_full_pipeline
    run_id = run_full_pipeline(
        data_path=Path(data),
        config_path=Path(config),
        factor_structure_path=Path(factor_config),
        runs_dir=Path(runs_dir),
        seed=seed,
    )
    click.echo(f"Pipeline complete. Run ID: {run_id}")


# ---------------------------------------------------------------------------
# run-psychometrics
# ---------------------------------------------------------------------------

@main.command("run-psychometrics")
@click.option("--data", default=_DEFAULT_DATA, type=click.Path())
@click.option("--factor-config", default=_DEFAULT_FACTOR_CONFIG, type=click.Path())
@click.option("--run-id", default=None, help="Existing run ID to update, or None to create new")
@click.option("--runs-dir", default=_DEFAULT_RUNS_DIR, type=click.Path())
@click.option("--seed", default=42, type=int)
def cmd_run_psychometrics(data: str, factor_config: str, run_id: str | None, runs_dir: str, seed: int) -> None:
    """Run psychometrics (CFA) stage only."""
    from src.ingest.loader import load_dataset
    from src.preprocessing.pipeline import run_preprocessing
    from src.psychometrics.runner import run_psychometrics
    from src.utils.run_manager import RunManager
    from src.utils.reproducibility import set_all_seeds

    set_all_seeds(seed)
    df_raw = load_dataset(Path(data))
    df = run_preprocessing(df_raw)

    mgr = RunManager(Path(runs_dir))
    if run_id and mgr.run_exists(run_id):
        rdir = mgr.get_run_dir(run_id)
    else:
        from src.utils.reproducibility import hash_file
        from src.config import get_system_info
        run_id = mgr.create_run(
            config_hash=hash_file(Path(factor_config))[:16],
            data_snapshot_id=hash_file(Path(data))[:16],
            random_seed=seed,
            system_info=get_system_info(),
        )
        rdir = mgr.get_run_dir(run_id)

    mgr.start_stage(run_id, "psychometrics")
    from datetime import datetime, timezone
    started = datetime.now(timezone.utc).isoformat()
    result = run_psychometrics(df, Path(factor_config), rdir)
    mgr.complete_stage(run_id, "psychometrics", started)
    click.echo(f"Psychometrics complete. Method: {result.method}. Run: {run_id}")


# ---------------------------------------------------------------------------
# train-language, train-sentiment, train-length
# ---------------------------------------------------------------------------

def _train_task_cmd(task_name: str, label_col: str):
    @main.command(f"train-{task_name}")
    @click.option("--data", default=_DEFAULT_DATA, type=click.Path())
    @click.option("--runs-dir", default=_DEFAULT_RUNS_DIR, type=click.Path())
    @click.option("--model-type", default="tfidf", type=click.Choice(["tfidf", "char_ngram"]))
    @click.option("--seed", default=42, type=int)
    def _cmd(data: str, runs_dir: str, model_type: str, seed: int) -> None:
        f"""Train {task_name} classifier."""
        from src.ingest.loader import load_dataset
        from src.preprocessing.pipeline import run_preprocessing
        from src.splits.splitter import stratified_split
        from src.text_tasks.trainer import train_single_task
        from src.utils.run_manager import RunManager
        from src.utils.reproducibility import set_all_seeds, hash_file
        from src.config import get_system_info

        set_all_seeds(seed)
        df = run_preprocessing(load_dataset(Path(data)))
        df_train, df_val, _ = stratified_split(df, seed=seed)

        mgr = RunManager(Path(runs_dir))
        run_id = mgr.create_run(
            config_hash="train_only",
            data_snapshot_id=hash_file(Path(data))[:16],
            random_seed=seed,
            system_info=get_system_info(),
        )
        rdir = mgr.get_run_dir(run_id)
        text_col = "text_processed" if "text_processed" in df_train.columns else "text_feedback"

        result = train_single_task(
            task_name=task_name,
            train_df=df_train,
            val_df=df_val,
            text_col=text_col,
            label_col=label_col,
            model_type=model_type,
            run_dir=rdir,
            seed=seed,
        )
        click.echo(f"Trained {task_name} ({model_type}). Val macro F1: {result.val_metrics['macro_f1']:.4f}. Run: {run_id}")

    return _cmd


cmd_train_language = _train_task_cmd("language", "language")
cmd_train_sentiment = _train_task_cmd("sentiment", "sentiment_class")
cmd_train_length = _train_task_cmd("length", "detail_level")


# ---------------------------------------------------------------------------
# run-fusion
# ---------------------------------------------------------------------------

@main.command("run-fusion")
@click.option("--run-id", required=True, help="Existing run ID with psychometrics + text tasks complete")
@click.option("--runs-dir", default=_DEFAULT_RUNS_DIR, type=click.Path())
def cmd_run_fusion(run_id: str, runs_dir: str) -> None:
    """Run fusion experiments for an existing run."""
    import pandas as pd
    from src.fusion.runner import run_fusion
    from src.utils.run_manager import RunManager

    mgr = RunManager(Path(runs_dir))
    rdir = mgr.get_run_dir(run_id)
    splits_dir = rdir / "splits"

    df_train = pd.read_parquet(splits_dir / "train.parquet")
    df_test = pd.read_parquet(splits_dir / "test.parquet")

    import pandas as pd
    fs_train = pd.read_csv(rdir / "psychometrics" / "factor_scores.csv", index_col=0)
    fs_test = fs_train.iloc[df_test.index]

    results = run_fusion(df_train, df_test, fs_train.iloc[df_train.index], fs_test, rdir, seed=42)
    click.echo(f"Fusion complete. Survey-only MAE: {results['survey_only']['mae']:.4f}. Run: {run_id}")


# ---------------------------------------------------------------------------
# run-contradiction
# ---------------------------------------------------------------------------

@main.command("run-contradiction")
@click.option("--run-id", required=True)
@click.option("--runs-dir", default=_DEFAULT_RUNS_DIR, type=click.Path())
def cmd_run_contradiction(run_id: str, runs_dir: str) -> None:
    """Run contradiction monitoring for an existing run."""
    import pandas as pd
    from src.contradiction.runner import run_contradiction_monitoring
    from src.utils.run_manager import RunManager

    mgr = RunManager(Path(runs_dir))
    rdir = mgr.get_run_dir(run_id)

    df_test = pd.read_parquet(rdir / "splits" / "test.parquet")
    fs_test = pd.read_csv(rdir / "psychometrics" / "factor_scores.csv", index_col=0)
    fs_test = fs_test.iloc[df_test.index]

    result = run_contradiction_monitoring(df_test, fs_test, rdir)
    click.echo(f"Contradiction monitoring done. Rate: {result.overall_rate:.4f}. Run: {run_id}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@main.command("evaluate")
@click.option("--run-id", required=True)
@click.option("--runs-dir", default=_DEFAULT_RUNS_DIR, type=click.Path())
def cmd_evaluate(run_id: str, runs_dir: str) -> None:
    """Evaluate all trained models for an existing run."""
    import pandas as pd
    from src.evaluation.runner import run_evaluation
    from src.utils.run_manager import RunManager

    mgr = RunManager(Path(runs_dir))
    rdir = mgr.get_run_dir(run_id)
    df_test = pd.read_parquet(rdir / "splits" / "test.parquet")
    text_col = "text_processed" if "text_processed" in df_test.columns else "text_feedback"
    results = run_evaluation(df_test, rdir, text_col=text_col)
    click.echo(f"Evaluation complete. Run: {run_id}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@main.command("report")
@click.option("--run-id", required=True)
@click.option("--runs-dir", default=_DEFAULT_RUNS_DIR, type=click.Path())
def cmd_report(run_id: str, runs_dir: str) -> None:
    """Generate evaluation report and model cards for an existing run."""
    from src.reporting.evaluation_report import generate_evaluation_report
    from src.reporting.model_card import generate_all_model_cards
    from src.reporting.data_dictionary import generate_data_dictionary
    from src.utils.run_manager import RunManager

    mgr = RunManager(Path(runs_dir))
    rdir = mgr.get_run_dir(run_id)
    report = generate_evaluation_report(rdir)
    cards = generate_all_model_cards(rdir)
    dd = generate_data_dictionary(rdir)
    click.echo(f"Reports generated. Evaluation report: {report}. Model cards: {len(cards)}. Run: {run_id}")


if __name__ == "__main__":
    main()
