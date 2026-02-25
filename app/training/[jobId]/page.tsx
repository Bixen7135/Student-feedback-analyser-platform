"use client";

import { useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { fetchTrainingStatus, TrainingJob } from "@/app/lib/api";

const TASK_LABELS: Record<string, string> = {
  language: "Language Detection",
  sentiment: "Sentiment Classification",
  detail_level: "Detail Level",
};

const MODEL_LABELS: Record<string, string> = {
  tfidf: "TF-IDF + LR",
  char_ngram: "Char N-gram + LR",
};

function StatusBadge({ status }: { status: string }) {
  const cfg: Record<string, { bg: string; color: string; label: string }> = {
    pending:   { bg: "rgba(245,158,11,0.12)", color: "var(--warning,#f59e0b)", label: "pending" },
    running:   { bg: "rgba(59,130,246,0.12)",  color: "#3b82f6",               label: "running" },
    completed: { bg: "rgba(16,185,129,0.12)",  color: "var(--success)",        label: "completed" },
    failed:    { bg: "rgba(239,68,68,0.12)",   color: "var(--error,#ef4444)",  label: "failed" },
  };
  const s = cfg[status] ?? cfg.pending;
  return (
    <span
      style={{
        background: s.bg,
        color: s.color,
        border: `1px solid ${s.color}`,
        borderRadius: "4px",
        padding: "2px 8px",
        fontSize: "11px",
        fontFamily: "var(--font-jetbrains)",
        fontWeight: 600,
      }}
    >
      {s.label}
    </span>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div
      className="flex"
      style={{
        padding: "7px 0",
        borderBottom: "1px solid var(--border-dim)",
        fontSize: "13px",
      }}
    >
      <span
        style={{
          width: "180px",
          color: "var(--text-tertiary)",
          flexShrink: 0,
          fontFamily: "var(--font-jetbrains)",
          fontSize: "12px",
        }}
      >
        {label}
      </span>
      <span style={{ color: "var(--text-primary)", fontFamily: "var(--font-jetbrains)" }}>
        {value}
      </span>
    </div>
  );
}

function ConfusionMatrix({
  matrix,
  classes,
}: {
  matrix: number[][];
  classes: string[];
}) {
  return (
    <div style={{ overflowX: "auto", marginTop: "8px" }}>
      <table
        style={{
          borderCollapse: "collapse",
          fontSize: "12px",
          fontFamily: "var(--font-jetbrains)",
        }}
      >
        <thead>
          <tr>
            <th
              style={{
                padding: "6px 10px",
                color: "var(--text-tertiary)",
                textAlign: "right",
                borderBottom: "1px solid var(--border)",
              }}
            >
              actual ↓ / pred →
            </th>
            {classes.map((c) => (
              <th
                key={c}
                style={{
                  padding: "6px 10px",
                  color: "var(--gold)",
                  borderBottom: "1px solid var(--border)",
                  whiteSpace: "nowrap",
                }}
              >
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => {
            const rowSum = row.reduce((a, b) => a + b, 0);
            return (
              <tr key={i}>
                <td
                  style={{
                    padding: "6px 10px",
                    color: "var(--gold)",
                    borderBottom: "1px solid var(--border-dim)",
                    textAlign: "right",
                    whiteSpace: "nowrap",
                  }}
                >
                  {classes[i]}
                </td>
                {row.map((cell, j) => {
                  const pct = rowSum > 0 ? cell / rowSum : 0;
                  const isDiag = i === j;
                  return (
                    <td
                      key={j}
                      style={{
                        padding: "6px 14px",
                        textAlign: "center",
                        borderBottom: "1px solid var(--border-dim)",
                        background: isDiag
                          ? `rgba(250,200,0,${0.1 + pct * 0.5})`
                          : pct > 0.05
                          ? `rgba(239,68,68,${pct * 0.4})`
                          : "transparent",
                        color: isDiag ? "var(--text-primary)" : "var(--text-secondary)",
                        fontWeight: isDiag ? 600 : 400,
                      }}
                    >
                      {cell}
                    </td>
                  );
                })}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default function TrainingJobPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const router = useRouter();
  const [job, setJob] = useState<TrainingJob | null>(null);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  function stopPolling() {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }

  async function loadJob() {
    try {
      const j = await fetchTrainingStatus(jobId);
      setJob(j);
      if (j.status === "completed" || j.status === "failed") {
        stopPolling();
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      stopPolling();
    }
  }

  useEffect(() => {
    loadJob();
    intervalRef.current = setInterval(loadJob, 2000);
    return stopPolling;
  }, [jobId]);

  const card = {
    background: "var(--bg-surface)",
    border: "1px solid var(--border)",
    borderRadius: "10px",
    padding: "24px",
    marginBottom: "16px",
  } as const;

  const sectionTitle = {
    fontFamily: "var(--font-syne)",
    fontSize: "11px",
    fontWeight: 700,
    letterSpacing: "0.12em",
    textTransform: "uppercase" as const,
    color: "var(--text-tertiary)",
    marginBottom: "12px",
  } as const;

  if (error) {
    return (
      <div style={{ padding: "32px", maxWidth: "720px" }}>
        <div
          style={{
            background: "rgba(239,68,68,0.08)",
            border: "1px solid rgba(239,68,68,0.3)",
            borderRadius: "8px",
            padding: "16px",
            fontSize: "13px",
            color: "var(--error,#ef4444)",
          }}
        >
          {error}
        </div>
        <button
          style={{
            marginTop: "16px",
            background: "transparent",
            border: "1px solid var(--border)",
            borderRadius: "6px",
            padding: "8px 16px",
            cursor: "pointer",
            color: "var(--text-secondary)",
            fontSize: "13px",
          }}
          onClick={() => router.push("/training")}
        >
          ← Back to Training
        </button>
      </div>
    );
  }

  if (!job) {
    return (
      <div style={{ padding: "32px" }}>
        <span style={{ fontSize: "13px", color: "var(--text-tertiary)" }}>
          Loading…
        </span>
      </div>
    );
  }

  const val = job.metrics?.val as Record<string, unknown> | undefined;
  const perClassF1 = val?.per_class_f1 as Record<string, number> | undefined;
  const confMatrix = val?.confusion_matrix as number[][] | undefined;
  const classes = job.metrics?.classes as string[] | undefined;

  return (
    <div style={{ padding: "32px", maxWidth: "800px" }} className="animate-fade-up">
      {/* Header */}
      <div
        className="flex items-start justify-between"
        style={{ marginBottom: "24px" }}
      >
        <div>
          <div className="flex items-center gap-3" style={{ marginBottom: "6px" }}>
            <h1
              style={{
                fontFamily: "var(--font-syne)",
                fontSize: "18px",
                fontWeight: 700,
                color: "var(--text-primary)",
              }}
            >
              Training Job
            </h1>
            <StatusBadge status={job.status} />
          </div>
          <div
            style={{
              fontFamily: "var(--font-jetbrains)",
              fontSize: "11px",
              color: "var(--text-tertiary)",
            }}
          >
            {job.job_id}
          </div>
        </div>
        <Link
          href="/training"
          style={{
            fontSize: "12px",
            color: "var(--text-tertiary)",
            textDecoration: "none",
            border: "1px solid var(--border)",
            borderRadius: "6px",
            padding: "6px 12px",
          }}
        >
          ← All Jobs
        </Link>
      </div>

      {/* Live spinner for running/pending */}
      {(job.status === "running" || job.status === "pending") && (
        <div
          style={{
            ...card,
            display: "flex",
            alignItems: "center",
            gap: "12px",
          }}
        >
          <div
            style={{
              width: "18px",
              height: "18px",
              borderRadius: "50%",
              border: "2px solid var(--border)",
              borderTopColor: "var(--gold)",
              animation: "spin 0.8s linear infinite",
            }}
          />
          <span style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
            {job.status === "running" ? "Training in progress…" : "Waiting to start…"}
          </span>
          <style>{`@keyframes spin{to{transform:rotate(360deg)}}`}</style>
        </div>
      )}

      {/* Error */}
      {job.status === "failed" && (
        <div
          style={{
            background: "rgba(239,68,68,0.08)",
            border: "1px solid rgba(239,68,68,0.3)",
            borderRadius: "8px",
            padding: "16px",
            fontSize: "13px",
            color: "var(--error,#ef4444)",
            marginBottom: "16px",
            fontFamily: "var(--font-jetbrains)",
          }}
        >
          <strong>Training failed:</strong>
          <br />
          {job.error}
        </div>
      )}

      {/* Job details */}
      <div style={card}>
        <div style={sectionTitle}>Job Details</div>
        <MetricRow label="Task" value={TASK_LABELS[job.task] ?? job.task} />
        <MetricRow label="Model type" value={MODEL_LABELS[job.model_type] ?? job.model_type} />
        <MetricRow label="Dataset" value={job.dataset_id} />
        <MetricRow label="Seed" value={String(job.seed)} />
        {job.started_at && (
          <MetricRow
            label="Started"
            value={new Date(job.started_at).toLocaleString()}
          />
        )}
        {job.completed_at && (
          <MetricRow
            label="Completed"
            value={new Date(job.completed_at).toLocaleString()}
          />
        )}
      </div>

      {/* Results — show when completed */}
      {job.status === "completed" && job.metrics && (
        <>
          {/* Model link */}
          {job.model_id && (
            <div
              style={{
                ...card,
                background: "var(--gold-faint)",
                border: "1px solid var(--gold-muted)",
              }}
            >
              <div className="flex items-center justify-between">
                <div>
                  <div
                    style={{
                      fontSize: "13px",
                      fontWeight: 600,
                      color: "var(--text-primary)",
                      marginBottom: "4px",
                    }}
                  >
                    Model registered: {job.model_name}
                  </div>
                  <div
                    style={{
                      fontSize: "11px",
                      color: "var(--text-tertiary)",
                      fontFamily: "var(--font-jetbrains)",
                    }}
                  >
                    v{job.model_version} · {job.model_id}
                  </div>
                </div>
                <Link
                  href={`/models/${job.model_id}`}
                  style={{
                    background: "var(--gold)",
                    color: "#000",
                    borderRadius: "6px",
                    padding: "8px 16px",
                    fontSize: "12px",
                    fontWeight: 600,
                    textDecoration: "none",
                    fontFamily: "var(--font-jetbrains)",
                  }}
                >
                  View Registered Model →
                </Link>
              </div>
            </div>
          )}

          {/* Validation metrics */}
          <div style={card}>
            <div style={sectionTitle}>Validation Metrics</div>
            <MetricRow
              label="Val macro F1"
              value={
                val?.macro_f1 != null
                  ? (val.macro_f1 as number).toFixed(4)
                  : "—"
              }
            />
            <MetricRow
              label="Val accuracy"
              value={
                val?.accuracy != null
                  ? (val.accuracy as number).toFixed(4)
                  : "—"
              }
            />
            <MetricRow
              label="Train macro F1"
              value={
                (job.metrics.train as Record<string, number>)?.macro_f1 != null
                  ? ((job.metrics.train as Record<string, number>).macro_f1).toFixed(4)
                  : "—"
              }
            />
            <MetricRow
              label="n_train / n_val / n_test"
              value={`${job.metrics.n_train ?? "?"} / ${job.metrics.n_val ?? "?"} / ${job.metrics.n_test ?? "?"}`}
            />
            {Boolean(job.metrics.text_col) && (
              <MetricRow label="Text column" value={String(job.metrics.text_col)} />
            )}
            {Boolean(job.metrics.label_col) && (
              <MetricRow label="Label column" value={String(job.metrics.label_col)} />
            )}
          </div>

          {/* Per-class F1 */}
          {perClassF1 && Object.keys(perClassF1).length > 0 && (
            <div style={card}>
              <div style={sectionTitle}>Per-Class F1</div>
              {Object.entries(perClassF1).map(([cls, f1]) => (
                <div
                  key={cls}
                  className="flex items-center gap-3"
                  style={{ marginBottom: "8px" }}
                >
                  <span
                    style={{
                      width: "120px",
                      fontSize: "12px",
                      fontFamily: "var(--font-jetbrains)",
                      color: "var(--text-secondary)",
                      flexShrink: 0,
                    }}
                  >
                    {cls}
                  </span>
                  <div
                    style={{
                      flex: 1,
                      background: "var(--bg-base)",
                      borderRadius: "3px",
                      height: "8px",
                      overflow: "hidden",
                    }}
                  >
                    <div
                      style={{
                        width: `${Math.round(f1 * 100)}%`,
                        height: "100%",
                        background: "var(--gold)",
                        borderRadius: "3px",
                        transition: "width 0.4s ease",
                      }}
                    />
                  </div>
                  <span
                    style={{
                      width: "48px",
                      fontSize: "12px",
                      fontFamily: "var(--font-jetbrains)",
                      color: "var(--text-primary)",
                      textAlign: "right",
                    }}
                  >
                    {f1.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          )}

          {/* Confusion matrix */}
          {confMatrix && classes && (
            <div style={card}>
              <div style={sectionTitle}>Confusion Matrix (validation)</div>
              <ConfusionMatrix matrix={confMatrix} classes={classes} />
            </div>
          )}
        </>
      )}
    </div>
  );
}

