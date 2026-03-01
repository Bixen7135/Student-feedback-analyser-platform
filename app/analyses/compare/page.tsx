"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  fetchAnalyses,
  compareAnalyses,
  AnalysisRecord,
  AnalysisComparison,
} from "@/app/lib/api";

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function delta(v: number): string {
  if (v > 0) return `+${pct(v)}`;
  if (v < 0) return `-${pct(Math.abs(v))}`;
  return "0.0%";
}

const TASK_LABELS: Record<string, string> = {
  language: "Language",
  sentiment: "Sentiment",
  detail_level: "Detail Level",
};

export default function CompareAnalysesPage() {
  const [analyses, setAnalyses] = useState<AnalysisRecord[]>([]);
  const [analysesLoading, setAnalysesLoading] = useState(true);
  const [analysesError, setAnalysesError] = useState<string | null>(null);

  const [id1, setId1] = useState("");
  const [id2, setId2] = useState("");

  const [comparison, setComparison] = useState<AnalysisComparison | null>(null);
  const [comparing, setComparing] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);

  useEffect(() => {
    fetchAnalyses({ status: "completed", per_page: 100, sort: "created_at", order: "desc" })
      .then((r) => {
        setAnalyses(r.analyses);
        setAnalysesLoading(false);
      })
      .catch((e) => {
        setAnalysesError(e.message);
        setAnalysesLoading(false);
      });
  }, []);

  async function handleCompare() {
    if (!id1 || !id2 || id1 === id2) return;
    setComparing(true);
    setCompareError(null);
    setComparison(null);
    try {
      const result = await compareAnalyses([id1, id2]);
      setComparison(result);
    } catch (e: unknown) {
      setCompareError(e instanceof Error ? e.message : String(e));
    } finally {
      setComparing(false);
    }
  }

  const completedAnalyses = analyses.filter((a) => a.status === "completed");

  return (
    <div className="page-shell page-standard page-shell--lg animate-fade-up">
      {/* Back link */}
      <div style={{ marginBottom: "20px" }}>
        <Link
          href="/analyses"
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "0.35rem",
            color: "var(--text-tertiary)",
            textDecoration: "none",
            fontFamily: "var(--font-jetbrains)",
            fontSize: "11px",
          }}
        >
          <span aria-hidden="true">&larr;</span>
          <span>Analyses</span>
        </Link>
      </div>

      <h1
        style={{
          fontFamily: "var(--font-syne)",
          fontWeight: 700,
          fontSize: "clamp(20px, 2vw, 24px)",
          color: "var(--text-primary)",
          marginBottom: "6px",
        }}
      >
        Compare Analyses
      </h1>
      <p style={{ color: "var(--text-tertiary)", fontSize: "13px", marginBottom: "28px" }}>
        Select two completed analyses to compare prediction distributions and disagreement rates.
      </p>

      {/* Selection */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 18rem), 1fr))",
          gap: "16px",
          marginBottom: "20px",
        }}
      >
        {[
          { label: "Analysis A", value: id1, set: setId1 },
          { label: "Analysis B", value: id2, set: setId2 },
        ].map(({ label, value, set }) => (
          <div key={label}>
            <label style={{ fontSize: "12px", color: "var(--text-secondary)", display: "block", marginBottom: "6px", fontWeight: 500 }}>
              {label}
            </label>
            {analysesLoading ? (
              <div style={{ color: "var(--text-tertiary)", fontSize: "12px" }}>Loading…</div>
            ) : analysesError ? (
              <div style={{ color: "var(--error, #ef4444)", fontSize: "12px" }}>{analysesError}</div>
            ) : (
              <select
                value={value}
                onChange={(e) => set(e.target.value)}
                style={{
                  width: "100%",
                  padding: "8px 12px",
                  border: "1px solid var(--border-dim)",
                  borderRadius: "6px",
                  background: "var(--bg-surface)",
                  color: "var(--text-primary)",
                  fontSize: "13px",
                }}
              >
                <option value="">Select analysis…</option>
                {completedAnalyses.map((a) => (
                  <option key={a.id} value={a.id}>
                    {a.name || a.id.slice(0, 20)} — {a.dataset_id?.slice(0, 10)} · {new Date(a.created_at).toLocaleDateString()}
                  </option>
                ))}
              </select>
            )}
          </div>
        ))}
      </div>

      <div className="flex gap-3 items-center" style={{ marginBottom: "28px", flexWrap: "wrap" }}>
        <button
          onClick={handleCompare}
          disabled={!id1 || !id2 || id1 === id2 || comparing}
          style={{
            marginLeft: "auto",
            padding: "9px 20px",
            background: id1 && id2 && id1 !== id2 && !comparing ? "var(--gold)" : "var(--border-dim)",
            color: id1 && id2 && id1 !== id2 && !comparing ? "#000" : "var(--text-tertiary)",
            border: "none",
            borderRadius: "6px",
            fontSize: "13px",
            fontWeight: 600,
            cursor: id1 && id2 && id1 !== id2 ? "pointer" : "not-allowed",
          }}
        >
          {comparing ? "Comparing…" : "Compare"}
        </button>
        {id1 === id2 && id1 !== "" && (
          <span style={{ fontSize: "12px", color: "var(--error, #ef4444)" }}>
            Select two different analyses.
          </span>
        )}
      </div>

      {compareError && (
        <div
          style={{
            padding: "12px 16px",
            borderRadius: "6px",
            background: "rgba(239,68,68,0.07)",
            border: "1px solid rgba(239,68,68,0.3)",
            color: "var(--error, #ef4444)",
            fontSize: "13px",
            marginBottom: "20px",
          }}
        >
          {compareError}
        </div>
      )}

      {/* Comparison results */}
      {comparison && (
        <div>
          {/* Run overview */}
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 18rem), 1fr))",
              gap: "16px",
              marginBottom: "24px",
            }}
          >
            {[comparison.run_1, comparison.run_2].map((run, i) => (
              <div
                key={run.id}
                style={{
                  padding: "16px 20px",
                  border: `1px solid ${i === 0 ? "var(--gold-muted)" : "var(--border-dim)"}`,
                  borderRadius: "8px",
                  background: "var(--bg-surface)",
                }}
              >
                <div style={{ fontSize: "11px", color: "var(--text-tertiary)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "8px" }}>
                  Analysis {i === 0 ? "A" : "B"}
                </div>
                <Link
                  href={`/analyses/${run.id}`}
                  style={{ fontWeight: 600, fontSize: "14px", color: "var(--text-primary)", textDecoration: "none" }}
                >
                  {run.name || run.id.slice(0, 20)}
                </Link>
                <div style={{ fontSize: "12px", color: "var(--text-tertiary)", marginTop: "6px" }}>
                  {run.n_rows != null && `${run.n_rows.toLocaleString()} rows · `}
                  {run.dataset_id?.slice(0, 12)}
                  {run.dataset_version != null && ` v${run.dataset_version}`}
                </div>
                <div style={{ fontSize: "11px", color: "var(--text-tertiary)", marginTop: "3px" }}>
                  {new Date(run.created_at).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}
                </div>
              </div>
            ))}
          </div>

          {/* Disagreement */}
          {comparison.disagreement && comparison.disagreement.same_dataset && !comparison.disagreement.error && (
            <div
              style={{
                padding: "14px 20px",
                border: "1px solid var(--border-dim)",
                borderRadius: "8px",
                background: "var(--bg-surface)",
                marginBottom: "24px",
              }}
            >
              <div style={{ fontSize: "11px", color: "var(--text-tertiary)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "10px" }}>
                Prediction Disagreement (same dataset)
              </div>
              {comparison.disagreement.overall != null && (
                <div style={{ fontSize: "24px", fontWeight: 700, color: "var(--text-primary)", fontFamily: "var(--font-jetbrains)", marginBottom: "8px" }}>
                  {pct(comparison.disagreement.overall)}
                  <span style={{ fontSize: "13px", color: "var(--text-tertiary)", fontWeight: 400, marginLeft: "8px" }}>
                    overall disagreement
                  </span>
                </div>
              )}
              {comparison.disagreement.by_task && (
                <div className="flex gap-4" style={{ flexWrap: "wrap" }}>
                  {Object.entries(comparison.disagreement.by_task).map(([task, rate]) => (
                    <div key={task}>
                      <div style={{ fontSize: "11px", color: "var(--text-tertiary)" }}>{TASK_LABELS[task] ?? task}</div>
                      <div style={{ fontSize: "16px", fontWeight: 600, color: "var(--text-primary)", fontFamily: "var(--font-jetbrains)" }}>
                        {pct(rate)}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Task comparisons */}
          {comparison.task_comparisons.length > 0 ? (
            <div>
              <h2 style={{ fontSize: "15px", fontWeight: 600, color: "var(--text-primary)", marginBottom: "16px" }}>
                Distribution Comparison by Task
              </h2>
              <div className="flex flex-col gap-5">
                {comparison.task_comparisons.map((tc) => {
                  const allClasses = Object.keys(tc.distribution_deltas);
                  return (
                    <div
                      key={tc.task}
                      style={{
                        padding: "20px",
                        border: "1px solid var(--border-dim)",
                        borderRadius: "8px",
                        background: "var(--bg-surface)",
                      }}
                    >
                      <h3 style={{ fontWeight: 600, fontSize: "14px", color: "var(--text-primary)", marginBottom: "16px" }}>
                        {TASK_LABELS[tc.task] ?? tc.task}
                      </h3>
                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 14rem), 1fr))",
                          gap: "20px",
                        }}
                      >
                        {/* Run A */}
                        <div>
                          <div style={{ fontSize: "11px", color: "var(--text-tertiary)", marginBottom: "8px" }}>
                            Analysis A — {tc.run_1_model.model_name} ({tc.run_1_model.n_predicted.toLocaleString()} rows)
                          </div>
                          {allClasses.map((cls) => (
                            <div key={cls} style={{ marginBottom: "6px" }}>
                              <div className="flex justify-between">
                                <span style={{ fontSize: "12px", color: "var(--text-secondary)" }}>{cls}</span>
                                <span style={{ fontSize: "12px", color: "var(--text-tertiary)", fontFamily: "var(--font-jetbrains)" }}>
                                  {pct(tc.run_1_model.class_distribution[cls] ?? 0)}
                                </span>
                              </div>
                              <div style={{ height: "5px", borderRadius: "2px", background: "var(--border-dim)", overflow: "hidden", marginTop: "3px" }}>
                                <div style={{ height: "100%", width: pct(tc.run_1_model.class_distribution[cls] ?? 0), background: "var(--gold)", borderRadius: "2px" }} />
                              </div>
                            </div>
                          ))}
                        </div>

                        {/* Run B */}
                        <div>
                          <div style={{ fontSize: "11px", color: "var(--text-tertiary)", marginBottom: "8px" }}>
                            Analysis B — {tc.run_2_model.model_name} ({tc.run_2_model.n_predicted.toLocaleString()} rows)
                          </div>
                          {allClasses.map((cls) => (
                            <div key={cls} style={{ marginBottom: "6px" }}>
                              <div className="flex justify-between">
                                <span style={{ fontSize: "12px", color: "var(--text-secondary)" }}>{cls}</span>
                                <span style={{ fontSize: "12px", color: "var(--text-tertiary)", fontFamily: "var(--font-jetbrains)" }}>
                                  {pct(tc.run_2_model.class_distribution[cls] ?? 0)}
                                </span>
                              </div>
                              <div style={{ height: "5px", borderRadius: "2px", background: "var(--border-dim)", overflow: "hidden", marginTop: "3px" }}>
                                <div style={{ height: "100%", width: pct(tc.run_2_model.class_distribution[cls] ?? 0), background: "var(--gold)", borderRadius: "2px" }} />
                              </div>
                            </div>
                          ))}
                        </div>

                        {/* Deltas */}
                        <div>
                          <div style={{ fontSize: "11px", color: "var(--text-tertiary)", marginBottom: "8px" }}>
                            Delta (B − A)
                          </div>
                          {allClasses.map((cls) => {
                            const d = tc.distribution_deltas[cls] ?? 0;
                            const pos = d > 0.002;
                            const neg = d < -0.002;
                            return (
                              <div key={cls} style={{ marginBottom: "6px", height: "28px", display: "flex", alignItems: "center" }}>
                                <span
                                  style={{
                                    fontSize: "13px",
                                    fontFamily: "var(--font-jetbrains)",
                                    fontWeight: 600,
                                    color: pos
                                      ? "var(--success)"
                                      : neg
                                      ? "var(--error, #ef4444)"
                                      : "var(--text-tertiary)",
                                  }}
                                >
                                  {delta(d)}
                                </span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div style={{ color: "var(--text-tertiary)", fontSize: "13px", padding: "20px 0" }}>
              No shared tasks found between these analyses.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

