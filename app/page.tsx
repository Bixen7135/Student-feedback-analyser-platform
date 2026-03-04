"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { fetchSummary, SummaryData } from "@/app/lib/api";

export default function DashboardPage() {
  const [summary, setSummary] = useState<SummaryData | null>(null);

  useEffect(() => {
    fetchSummary()
      .then((data) => {
        setSummary(data);
      })
      .catch(() => {
        setSummary(null);
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

  const headlineCount = summary
    ? (
        summary.total_datasets +
        summary.total_models +
        summary.total_analyses
      ).toLocaleString()
    : "-";

  return (
    <div className="db-shell page-standard animate-fade-up">
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
              <span>Analyser Dashboard</span>
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
                Run Pipeline
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
    </div>
  );
}
