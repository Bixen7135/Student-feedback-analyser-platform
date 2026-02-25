"use client";

import { ReactNode, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Disclaimer } from "@/app/components/Disclaimer";
import { fetchSummary, SummaryData } from "@/app/lib/api";

const PIPELINE_STAGES = [
  { num: "01", label: "Ingest", detail: "Snapshot and normalize" },
  { num: "02", label: "Psychometrics", detail: "CFA and reliability checks" },
  { num: "03", label: "Splits", detail: "Leak-safe train and test split" },
  { num: "04", label: "Text Tasks", detail: "Language, sentiment, detail" },
  { num: "05", label: "Fusion", detail: "Text + survey signal blend" },
  { num: "06", label: "Contradiction", detail: "Flag rating and text mismatch" },
  { num: "07", label: "Evaluate", detail: "Macro F1 and fit diagnostics" },
  { num: "08", label: "Report", detail: "Model cards and run artifacts" },
];

const QUICK_ACTIONS = [
  {
    href: "/runs/new",
    title: "Launch Pipeline",
    detail: "Start a full reproducible run from selected dataset version.",
  },
  {
    href: "/datasets",
    title: "Manage Datasets",
    detail: "Inspect versions, edit rows, and control branch history.",
  },
  {
    href: "/analyses/new",
    title: "Start Analysis",
    detail: "Apply trained models and compare prediction distributions.",
  },
];

const RUNBOOK_COMMANDS = `# run full pipeline
cd backend
uv run python -m src.cli run-full --data ../mnt/data/dataset.csv

# start backend api
uv run uvicorn src.api.main:app --port 8000`;

const SectionLabel = ({ children }: { children: ReactNode }) => (
  <div className="db-section-label">{children}</div>
);

export default function DashboardPage() {
  const [summary, setSummary] = useState<SummaryData | null>(null);
  const [summaryStatus, setSummaryStatus] = useState<"loading" | "ready" | "error">(
    "loading"
  );

  useEffect(() => {
    fetchSummary()
      .then((data) => {
        setSummary(data);
        setSummaryStatus("ready");
      })
      .catch(() => {
        setSummary(null);
        setSummaryStatus("error");
      });
  }, []);

  const stats = useMemo(
    () => [
      {
        value: summary ? summary.total_datasets.toLocaleString() : "-",
        label: "Datasets",
      },
      {
        value: summary ? summary.total_models.toLocaleString() : "-",
        label: "Registered Models",
      },
      {
        value: summary ? summary.total_analyses.toLocaleString() : "-",
        label: "Analyses",
      },
      {
        value: summary ? summary.total_responses.toLocaleString() : "-",
        label: "Responses",
      },
      {
        value: summary ? summary.n_latent_factors.toLocaleString() : "-",
        label: "Latent Factors",
      },
      {
        value: summary ? summary.n_survey_items.toLocaleString() : "-",
        label: "Survey Items",
      },
    ],
    [summary]
  );

  const summaryStatusLabel =
    summaryStatus === "loading"
      ? "syncing summary..."
      : summaryStatus === "ready"
      ? "summary synced"
      : "summary unavailable";

  const headlineCount = summary
    ? (
        summary.total_datasets +
        summary.total_models +
        summary.total_analyses
      ).toLocaleString()
    : "-";

  return (
    <div className="db-shell animate-fade-up">
      <section className="db-hero">
        <div className="dot-grid db-hero-grid" aria-hidden="true" />
        <div className="db-hero-content">
          <div>
            <div className="db-kicker">
              <span className="db-kicker-dot" />
              Multilingual Batch Analysis
            </div>
            <h1 className="db-title">
              Student Feedback
              <br />
              <span>Command Dashboard</span>
            </h1>
            <p className="db-subtitle">
              One workspace for psychometrics, supervised text models, and reproducible
              experiment tracking across RU and KZ student feedback.
            </p>
            <div className="db-chip-row">
              <span className="db-chip">8-stage pipeline</span>
              <span className="db-chip">batch-only execution</span>
              <span className="db-chip">traceable artifacts</span>
            </div>
            <div className="db-cta-row">
              <Link href="/runs/new" className="db-cta-primary">
                Launch New Run
              </Link>
              <Link href="/runs" className="db-cta-secondary">
                View Run History
              </Link>
            </div>
          </div>

          <div className="db-stat-grid">
            <div className="db-stat-card db-stat-card-highlight">
              <div className="db-stat-label">Active Inventory</div>
              <div className="db-stat-value">{headlineCount}</div>
              <div className="db-stat-note">datasets + models + analyses</div>
            </div>
            {stats.map(({ value, label }) => (
              <div key={label} className="db-stat-card">
                <div className="db-stat-value">{value}</div>
                <div className="db-stat-label">{label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <div className="db-two-col">
        <section className="db-panel">
          <SectionLabel>Quick Actions</SectionLabel>
          <div className="db-action-grid">
            {QUICK_ACTIONS.map((action) => (
              <Link key={action.href} href={action.href} className="db-action-card">
                <div className="db-action-title">{action.title}</div>
                <p className="db-action-detail">{action.detail}</p>
                <div className="db-action-link">Open -&gt;</div>
              </Link>
            ))}
          </div>
        </section>

        <section className="db-panel">
          <SectionLabel>Runbook</SectionLabel>
          <p className="db-panel-note">
            Use these commands when running the pipeline manually from your local shell.
          </p>
          <pre className="db-terminal">{RUNBOOK_COMMANDS}</pre>
        </section>
      </div>

      <section className="db-panel">
        <div className="db-panel-header">
          <SectionLabel>Pipeline Stages</SectionLabel>
          <span className="db-inline-status">{summaryStatusLabel}</span>
        </div>
        <div className="db-stage-scroll">
          {PIPELINE_STAGES.map((stage, index) => (
            <div key={stage.num} className="db-stage-item">
              <div className="db-stage-card">
                <span className="db-stage-num">{stage.num}</span>
                <div className="db-stage-title">{stage.label}</div>
                <div className="db-stage-detail">{stage.detail}</div>
              </div>
              {index < PIPELINE_STAGES.length - 1 && (
                <span className="db-stage-arrow">-&gt;</span>
              )}
            </div>
          ))}
        </div>
      </section>

      <Disclaimer />
    </div>
  );
}
