"""Model card generator for each released text classification model."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson  # type: ignore

from src.utils.logging import get_logger

log = get_logger(__name__)


def generate_model_card(
    task: str,
    model_type: str,
    metrics: dict[str, Any],
    hyperparameters: dict[str, Any],
    run_id: str,
    run_dir: Path,
) -> Path:
    """
    Generate a markdown model card for a trained text classifier.
    Saves to run_dir/reports/model_cards/{task}_{model_type}_model_card.md
    """
    cards_dir = run_dir / "reports" / "model_cards"
    cards_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat()

    lines = [
        f"# Model Card: {task.replace('_', ' ').title()} — {model_type}",
        f"",
        f"**Run ID:** `{run_id}`",
        f"**Generated:** {now}",
        f"",
        f"---",
        f"",
        f"## Model Details",
        f"",
        f"- **Task:** {task}",
        f"- **Model Type:** {model_type}",
        f"- **Framework:** scikit-learn",
        f"",
        f"### Hyperparameters",
        f"",
        f"```json",
        orjson.dumps(hyperparameters, option=orjson.OPT_INDENT_2).decode(),
        f"```",
        f"",
        f"---",
        f"",
        f"## Performance (Validation Set)",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
    ]

    val = metrics.get("val", metrics)
    for k, v in val.items():
        if isinstance(v, float):
            lines.append(f"| {k} | {v:.4f} |")
        elif isinstance(v, dict):
            for sub_k, sub_v in v.items():
                sv_str = f"{sub_v:.4f}" if isinstance(sub_v, float) else str(sub_v)
                lines.append(f"| {k}/{sub_k} | {sv_str} |")

    lines += [
        f"",
        f"---",
        f"",
        f"## Intended Use",
        f"",
        f"- This model classifies student feedback text by {task}.",
        f"- Intended for aggregate reporting and quality monitoring only.",
        f"- **Not for individual-level decisions about students or staff.**",
        f"",
        f"## Limitations",
        f"",
        f"- Trained on Russian/Kazakh educational feedback. May not generalize to other domains.",
        f"- Performance varies by language (see stratified metrics in evaluation report).",
        f"- Class imbalance: the training set has {task}-specific class imbalance (see report).",
        f"- Balanced class weights used during training to mitigate imbalance effects.",
        f"",
        f"## No Causal Claims",
        f"",
        f"This model identifies patterns in text. It does not establish causation.",
        f"",
        f"## Training Data",
        f"",
        f"- Dataset: Multilingual student feedback (Russian/Kazakh/mixed)",
        f"- Labels: Pre-existing annotations (language, sentiment, detail level)",
        f"- Split: 80% train, 10% validation, 10% test (stratified by sentiment class)",
    ]

    card_text = "\n".join(lines)
    out_path = cards_dir / f"{task}_{model_type}_model_card.md"
    out_path.write_text(card_text, encoding="utf-8")
    log.info("model_card_generated", path=str(out_path))
    return out_path


def generate_all_model_cards(run_dir: Path) -> list[Path]:
    """Generate model cards for all trained models in a run."""
    paths: list[Path] = []
    metadata = orjson.loads((run_dir / "metadata.json").read_bytes())
    run_id = metadata.get("run_id", "unknown")

    text_tasks_dir = run_dir / "text_tasks"
    if not text_tasks_dir.exists():
        return paths

    for task_dir in text_tasks_dir.iterdir():
        if not task_dir.is_dir():
            continue
        for model_dir in task_dir.iterdir():
            if not model_dir.is_dir():
                continue
            metrics_path = model_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            metrics_data = orjson.loads(metrics_path.read_bytes())
            path = generate_model_card(
                task=metrics_data.get("task", task_dir.name),
                model_type=metrics_data.get("model_type", model_dir.name),
                metrics=metrics_data,
                hyperparameters=metrics_data.get("hyperparameters", {}),
                run_id=run_id,
                run_dir=run_dir,
            )
            paths.append(path)

    return paths
