"""Generate markdown evaluation report per run."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson  # type: ignore

from src.utils.logging import get_logger

log = get_logger(__name__)


def _safe_load(path: Path) -> Any:
    if path.exists():
        return orjson.loads(path.read_bytes())
    return {}


def generate_evaluation_report(run_dir: Path) -> Path:
    """
    Generate a markdown evaluation report from all artifacts in run_dir.
    Saves to run_dir/reports/evaluation_report.md.
    """
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    metadata = _safe_load(run_dir / "metadata.json")
    cfa_summary = _safe_load(run_dir / "psychometrics" / "cfa_summary.json")
    reliability = _safe_load(run_dir / "psychometrics" / "reliability.json")
    fit_stats = _safe_load(run_dir / "psychometrics" / "fit_statistics.json")
    fusion = _safe_load(run_dir / "fusion" / "results.json")
    eval_class = _safe_load(run_dir / "evaluation" / "classification_results.json")
    contradiction = _safe_load(run_dir / "contradiction" / "results.json")

    now = datetime.now(timezone.utc).isoformat()
    run_id = metadata.get("run_id", str(run_dir.name))

    lines = [
        f"# Evaluation Report",
        f"",
        f"**Run ID:** `{run_id}`",
        f"**Generated:** {now}",
        f"**Config Hash:** `{metadata.get('config_hash', 'N/A')}`",
        f"**Data Snapshot:** `{metadata.get('data_snapshot_id', 'N/A')}`",
        f"**Random Seed:** {metadata.get('random_seed', 'N/A')}",
        f"",
        f"---",
        f"",
        f"> **Disclaimers:**",
        f"> - This system is for analytical reporting only.",
        f"> - Not for individual-level decision making.",
        f"> - No causal claims are made.",
        f"> - Latent quality factors are derived from survey items only, not from text.",
        f"",
        f"---",
        f"",
        f"## 1. Psychometrics",
        f"",
        f"**Method:** {cfa_summary.get('method', 'N/A')}",
        f"**Observations:** {cfa_summary.get('n_obs', 'N/A')}",
        f"",
        f"### Fit Statistics",
        f"",
    ]

    if fit_stats:
        lines.append("| Statistic | Value |")
        lines.append("|-----------|-------|")
        for k, v in fit_stats.items():
            if isinstance(v, float):
                lines.append(f"| {k} | {v:.4f} |")
            elif not isinstance(v, (list, dict)):
                lines.append(f"| {k} | {v} |")
    lines.append("")

    lines += [
        "### Reliability",
        "",
        "| Factor | Cronbach α | McDonald ω |",
        "|--------|-----------|-----------|",
    ]
    alpha = reliability.get("cronbach_alpha", {})
    omega = reliability.get("mcdonald_omega", {})
    for factor in cfa_summary.get("factor_names", []):
        a = alpha.get(factor, "N/A")
        w = omega.get(factor, "N/A")
        a_str = f"{a:.4f}" if isinstance(a, float) else str(a)
        w_str = f"{w:.4f}" if isinstance(w, float) else str(w)
        lines.append(f"| {factor} | {a_str} | {w_str} |")
    lines += ["", "---", ""]

    lines += [
        "## 2. Text Classification",
        "",
        "All tasks trained independently. No shared gradients or loss functions.",
        "",
    ]

    for task, task_results in eval_class.items():
        lines.append(f"### {task.replace('_', ' ').title()}")
        lines.append("")
        for model_type, metrics in task_results.items():
            overall = metrics.get("overall", {})
            mf1 = overall.get("macro_f1", "N/A")
            acc = overall.get("accuracy", "N/A")
            mf1_str = f"{mf1:.4f}" if isinstance(mf1, float) else str(mf1)
            acc_str = f"{acc:.4f}" if isinstance(acc, float) else str(acc)
            lines.append(f"**{model_type}:** Macro F1 = {mf1_str}, Accuracy = {acc_str}")
        lines.append("")

    lines += ["---", ""]

    lines += [
        "## 3. Fusion Experiments",
        "",
        "Regression targets: psychometric factor scores (standardized).",
        "Model: HuberRegressor. Lower MAE is better. Higher R² is better.",
        "",
        "| Model | MAE | R² |",
        "|-------|-----|-----|",
    ]
    if fusion:
        for model_key in ["survey_only", "text_only", "late_fusion"]:
            m = fusion.get(model_key, {})
            mae = m.get("mae", "N/A")
            r2 = m.get("r_squared", "N/A")
            mae_str = f"{mae:.4f}" if isinstance(mae, float) else str(mae)
            r2_str = f"{r2:.4f}" if isinstance(r2, float) else str(r2)
            lines.append(f"| {model_key.replace('_', ' ')} | {mae_str} | {r2_str} |")

    lines += [
        "",
        "### Deltas (fusion - survey_only)",
        "",
        "Negative Δ MAE = improvement. Null results reported without spin.",
        "",
        "| Factor | Δ MAE | Δ R² |",
        "|--------|-------|------|",
    ]
    for f in fusion.get("factor_names", []):
        dmae = fusion.get("delta_mae", {}).get(f, "N/A")
        dr2 = fusion.get("delta_r2", {}).get(f, "N/A")
        dmae_str = f"{dmae:+.4f}" if isinstance(dmae, float) else str(dmae)
        dr2_str = f"{dr2:+.4f}" if isinstance(dr2, float) else str(dr2)
        lines.append(f"| {f} | {dmae_str} | {dr2_str} |")
    lines += ["", "---", ""]

    lines += [
        "## 4. Contradiction Monitoring",
        "",
        "> Monitoring only. Predictions are not altered.",
        "",
        f"**Overall contradiction rate:** {contradiction.get('overall_rate', 'N/A')}",
        f"**Total flagged:** {contradiction.get('n_contradictions', 'N/A')} / {contradiction.get('n_total', 'N/A')}",
        "",
        "| Type | Rate |",
        "|------|------|",
    ]
    for ctype, rate in contradiction.get("by_type", {}).items():
        rate_str = f"{rate:.4f}" if isinstance(rate, float) else str(rate)
        lines.append(f"| {ctype} | {rate_str} |")
    lines += ["", "---", ""]

    report_text = "\n".join(lines)
    out_path = reports_dir / "evaluation_report.md"
    out_path.write_text(report_text, encoding="utf-8")
    log.info("evaluation_report_generated", path=str(out_path))
    return out_path
