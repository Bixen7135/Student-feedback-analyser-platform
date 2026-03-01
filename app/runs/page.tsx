"use client";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { fetchRuns, deleteRun, RunSummary } from "@/app/lib/api";
import { StageProgress } from "@/app/components/StageProgress";

function pluralize(count: number, singular: string, plural: string): string {
  return count === 1 ? singular : plural;
}

function overallStatus(run: RunSummary): "completed" | "running" | "failed" | "partial" {
  const stages = Object.values(run.stages);
  if (stages.some((s) => s.status === "failed")) return "failed";
  if (stages.some((s) => s.status === "running")) return "running";
  if (stages.length > 0 && stages.every((s) => s.status === "completed")) return "completed";
  return "partial";
}

const STATUS_STYLE: Record<string, { color: string; bg: string; border: string; dot: string }> = {
  completed: { color: "var(--success)", bg: "var(--success-dim)", border: "var(--success)", dot: "var(--success)" },
  running:   { color: "var(--running)", bg: "var(--running-dim)", border: "var(--running)", dot: "var(--running)" },
  failed:    { color: "var(--error)",   bg: "var(--error-dim)",   border: "var(--error)",   dot: "var(--error)" },
  partial:   { color: "var(--warning)", bg: "var(--warning-dim)", border: "var(--warning)", dot: "var(--warning)" },
};

const PER_PAGE = 20;

export default function RunHistoryPage() {
  const router = useRouter();
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [expandedStageRuns, setExpandedStageRuns] = useState<Record<string, boolean>>({});

  useEffect(() => {
    let active = true;
    let timeout: ReturnType<typeof setTimeout> | null = null;

    async function poll() {
      try {
        const data = await fetchRuns();
        if (!active) return;
        setRuns(data);
        setLoading(false);
        // Keep polling while any run is actively running
        const anyRunning = data.some((r) =>
          Object.values(r.stages).some((s) => s.status === "running")
        );
        if (anyRunning) {
          timeout = setTimeout(poll, 3000);
        }
      } catch (e: Error | unknown) {
        if (!active) return;
        setError((e as Error).message);
        setLoading(false);
      }
    }

    poll();
    return () => {
      active = false;
      if (timeout) clearTimeout(timeout);
    };
  }, []);

  useEffect(() => {
    const totalPages = Math.max(1, Math.ceil(runs.length / PER_PAGE));
    if (page > totalPages) {
      setPage(totalPages);
    }
  }, [page, runs.length]);

  async function handleDelete(runId: string) {
    setDeletingId(runId);
    setDeleteError(null);
    try {
      await deleteRun(runId);
      const nextRuns = runs.filter((r) => r.run_id !== runId);
      const nextTotalPages = Math.max(1, Math.ceil(nextRuns.length / PER_PAGE));
      setRuns(nextRuns);
      if (page > nextTotalPages) {
        setPage(nextTotalPages);
      }
      setConfirmingId(null);
    } catch (e: unknown) {
      setDeleteError(e instanceof Error ? e.message : "Delete failed");
    } finally {
      setDeletingId(null);
    }
  }

  const total = runs.length;
  const totalPages = Math.max(1, Math.ceil(total / PER_PAGE));
  const rangeStart = total === 0 ? 0 : (page - 1) * PER_PAGE + 1;
  const rangeEnd = total === 0 ? 0 : Math.min(total, page * PER_PAGE);
  const pagedRuns = total === 0 ? [] : runs.slice(rangeStart - 1, rangeEnd);

  return (
    <div className="page-shell page-standard page-shell--md animate-fade-up">

      {/* ── Header ──────────────────────────────────────── */}
      <div className="flex items-center justify-between" style={{ marginBottom: "28px" }}>
        <div>
          <div
            style={{
              fontFamily: "var(--font-syne)",
              fontSize: "9.5px",
              fontWeight: 700,
              letterSpacing: "0.18em",
              textTransform: "uppercase",
              color: "var(--text-tertiary)",
              marginBottom: "6px",
            }}
          >
            Analysis
          </div>
          <h1
            style={{
              fontFamily: "var(--font-syne)",
              fontWeight: 700,
              fontSize: "22px",
              color: "var(--text-primary)",
              letterSpacing: "-0.01em",
            }}
          >
            Run History
            {!loading && !error && (
              <span
                style={{
                  fontFamily: "var(--font-jetbrains)",
                  fontSize: "12px",
                  fontWeight: 400,
                  color: "var(--text-tertiary)",
                  marginLeft: "10px",
                }}
              >
                {total} total
              </span>
            )}
          </h1>
        </div>
        <div className="runs-page__header-actions">
          <Link
            href="/models"
            className="runs-page__header-action runs-page__header-action--secondary"
            style={{
              background: "transparent",
              color: "var(--text-secondary)",
              border: "1px solid var(--border-dim)",
            }}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
              <path d="M6 1L10.5 3.5V8.5L6 11L1.5 8.5V3.5L6 1Z" stroke="currentColor" strokeWidth="1.2" strokeLinejoin="round" />
              <path d="M6 6L10.5 3.5M6 6L1.5 3.5M6 6V11" stroke="currentColor" strokeWidth="1.2" />
            </svg>
            Models
          </Link>
          <Link
            href="/runs/new"
            className="runs-page__header-action runs-page__header-action--primary"
            style={{
              background: "var(--gold)",
              color: "#08080B",
            }}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
              <circle cx="6" cy="6" r="4.5" stroke="currentColor" strokeWidth="1.3" />
              <path d="M6 3.5V8.5M3.5 6H8.5" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
            </svg>
            Run Pipeline
          </Link>
        </div>
      </div>

      {/* ── States ──────────────────────────────────────── */}
      {loading && (
        <div
          style={{
            color: "var(--text-tertiary)",
            fontFamily: "var(--font-jetbrains)",
            fontSize: "12px",
            padding: "48px 0",
          }}
        >
          Loading runs…
        </div>
      )}

      {error && (
        <div
          className="rounded-lg"
          style={{
            background: "var(--error-dim)",
            border: "1px solid var(--error)",
            padding: "12px 16px",
            color: "var(--error)",
            fontSize: "13px",
            fontFamily: "var(--font-jetbrains)",
          }}
        >
          Failed to load runs: {error}. Ensure the API server is running on port 8000.
        </div>
      )}

      {deleteError && (
        <div
          className="rounded-lg"
          style={{
            background: "var(--error-dim)",
            border: "1px solid var(--error)",
            padding: "10px 16px",
            color: "var(--error)",
            fontSize: "12px",
            fontFamily: "var(--font-jetbrains)",
            marginBottom: "12px",
          }}
        >
          {deleteError}
        </div>
      )}

      {!loading && !error && total === 0 && (
        <div
          className="rounded-xl text-center"
          style={{
            background: "var(--bg-surface)",
            border: "1px solid var(--border-dim)",
            padding: "64px 24px",
          }}
        >
          <div
            style={{
              fontFamily: "var(--font-syne)",
              fontWeight: 700,
              fontSize: "16px",
              color: "var(--text-secondary)",
              marginBottom: "8px",
            }}
          >
            No runs yet
          </div>
          <div style={{ fontSize: "13px", color: "var(--text-tertiary)" }}>
            Launch your first run to get started.
          </div>
        </div>
      )}

      {/* ── Run cards ───────────────────────────────────── */}
      <div className="flex flex-col gap-3">
        {pagedRuns.map((run) => {
          const status = overallStatus(run);
          const ss = STATUS_STYLE[status] ?? STATUS_STYLE.partial;
          const isConfirming = confirmingId === run.run_id;
          const isDeleting = deletingId === run.run_id;
          const isRunning = status === "running";
          const isStagesExpanded = expandedStageRuns[run.run_id] ?? false;
          const stagePanelId = `run-stage-progress-${run.run_id}`;

          return (
            <div
              key={run.run_id}
              className="rounded-xl transition-all duration-150"
              style={{
                background: "var(--bg-surface)",
                border: "1px solid var(--border-dim)",
                borderLeft: `3px solid ${ss.border}`,
                padding: "18px 20px",
                cursor: "pointer",
              }}
              onClick={() => router.push(`/runs/${run.run_id}`)}
            >
              {/* Top row — id+badge on left, timestamp+delete on right */}
              <div className="flex items-center justify-between" style={{ marginBottom: "10px" }}>
                {/* Left */}
                <div className="flex items-center gap-3 flex-wrap">
                  <span
                    style={{
                      fontFamily: "var(--font-jetbrains)",
                      fontSize: "12px",
                      color: "var(--text-primary)",
                    }}
                  >
                    {run.run_id}
                  </span>
                  <span
                    className="inline-flex items-center gap-1.5 rounded"
                    style={{
                      background: ss.bg,
                      border: `1px solid ${ss.border}`,
                      color: ss.color,
                      padding: "2px 8px",
                      fontFamily: "var(--font-jetbrains)",
                      fontSize: "10px",
                    }}
                  >
                    <span
                      className={`rounded-full shrink-0 ${status === "running" ? "animate-pulse-dot" : ""}`}
                      style={{ width: "5px", height: "5px", background: ss.dot }}
                    />
                    {status}
                  </span>
                </div>

                {/* Right — timestamp + delete control side by side */}
                <div
                  className="flex items-center"
                  style={{ gap: "10px", flexShrink: 0 }}
                  onClick={(e) => e.stopPropagation()}
                >
                  <span
                    style={{
                      fontFamily: "var(--font-jetbrains)",
                      fontSize: "10px",
                      color: "var(--text-tertiary)",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {new Date(run.created_at).toLocaleString()}
                  </span>

                  {isConfirming ? (
                    <div className="flex items-center" style={{ gap: "6px" }}>
                      <span
                        style={{
                          fontFamily: "var(--font-jetbrains)",
                          fontSize: "10px",
                          color: "var(--error)",
                          whiteSpace: "nowrap",
                        }}
                      >
                        Delete?
                      </span>
                      <button
                        onClick={() => setConfirmingId(null)}
                        style={{
                          background: "transparent",
                          border: "1px solid var(--border-dim)",
                          borderRadius: "4px",
                          color: "var(--text-tertiary)",
                          fontFamily: "var(--font-jetbrains)",
                          fontSize: "10px",
                          padding: "2px 8px",
                          cursor: "pointer",
                          whiteSpace: "nowrap",
                        }}
                      >
                        Cancel
                      </button>
                      <button
                        onClick={() => handleDelete(run.run_id)}
                        disabled={isDeleting}
                        style={{
                          background: "var(--error-dim)",
                          border: "1px solid var(--error)",
                          borderRadius: "4px",
                          color: "var(--error)",
                          fontFamily: "var(--font-jetbrains)",
                          fontSize: "10px",
                          padding: "2px 8px",
                          cursor: isDeleting ? "not-allowed" : "pointer",
                          opacity: isDeleting ? 0.6 : 1,
                          whiteSpace: "nowrap",
                        }}
                      >
                        {isDeleting ? "…" : "Confirm"}
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => { if (!isRunning) setConfirmingId(run.run_id); }}
                      title={isRunning ? "Cannot delete a running run" : "Delete run"}
                      disabled={isRunning}
                      style={{
                        background: "transparent",
                        border: "none",
                        padding: "4px",
                        cursor: isRunning ? "not-allowed" : "pointer",
                        opacity: isRunning ? 0.3 : 0.4,
                        color: "var(--text-tertiary)",
                        display: "flex",
                        alignItems: "center",
                        borderRadius: "4px",
                        transition: "opacity 0.15s",
                        flexShrink: 0,
                      }}
                      onMouseEnter={(e) => { if (!isRunning) e.currentTarget.style.opacity = "1"; }}
                      onMouseLeave={(e) => { if (!isRunning) e.currentTarget.style.opacity = "0.4"; }}
                    >
                      <svg width="14" height="14" viewBox="0 0 14 14" fill="none" aria-label="Delete run">
                        <path
                          d="M2.5 3.5H11.5M5.5 3.5V2.5H8.5V3.5M6 6V10M8 6V10M3.5 3.5L4 11.5H10L10.5 3.5"
                          stroke="currentColor"
                          strokeWidth="1.2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        />
                      </svg>
                    </button>
                  )}
                </div>
              </div>

              {/* Meta */}
              <div
                className="runs-page__run-meta"
                style={{
                  fontFamily: "var(--font-jetbrains)",
                  fontSize: "10px",
                  color: "var(--text-tertiary)",
                  marginBottom: "12px",
                }}
              >
                <span>seed:{run.random_seed}</span>
                <span>cfg:{run.config_hash.slice(0, 8)}</span>
                {run.produced_models_count > 0 && (
                  <Link
                    href={`/models?run_id=${encodeURIComponent(run.run_id)}&include_archived=true`}
                    onClick={(e) => e.stopPropagation()}
                    style={{
                      color: "var(--gold)",
                      textDecoration: "none",
                      borderBottom: "1px solid var(--gold-muted)",
                    }}
                  >
                    {run.produced_models_count}{" "}
                    {pluralize(run.produced_models_count, "model", "models")}
                  </Link>
                )}
                <button
                  type="button"
                  className={`runs-page__stage-toggle${isStagesExpanded ? " is-open" : ""}`}
                  aria-expanded={isStagesExpanded}
                  aria-controls={stagePanelId}
                  aria-label={isStagesExpanded ? "Hide stage timings" : "Show stage timings"}
                  onClick={(e) => {
                    e.stopPropagation();
                    setExpandedStageRuns((prev) => ({
                      ...prev,
                      [run.run_id]: !prev[run.run_id],
                    }));
                  }}
                >
                  <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true">
                    <path
                      d="M2 3.5L5 6.5L8 3.5"
                      stroke="currentColor"
                      strokeWidth="1.2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </button>
              </div>

              {/* Stages */}
              <div
                id={stagePanelId}
                className={`runs-page__stage-panel${isStagesExpanded ? " is-open" : ""}`}
              >
                <StageProgress stages={run.stages} />
              </div>
            </div>
          );
        })}
      </div>

      {!loading && !error && total > 0 && (
        <div
          className="runs-page__pagination-bar"
          style={{ marginTop: "14px", flexWrap: "wrap" }}
        >
          <div
            className="runs-page__pagination-summary"
            style={{
              fontFamily: "var(--font-jetbrains)",
              fontSize: "11px",
              color: "var(--text-tertiary)",
            }}
          >
            Showing {rangeStart}-{rangeEnd} of {total}
          </div>
          <div className="runs-page__pagination-controls">
            <button
              onClick={() => setPage((prev) => Math.max(1, prev - 1))}
              disabled={page <= 1}
              style={{
                background: "transparent",
                border: "1px solid var(--border-dim)",
                borderRadius: "6px",
                color: page <= 1 ? "var(--text-tertiary)" : "var(--text-secondary)",
                fontFamily: "var(--font-jetbrains)",
                fontSize: "10px",
                padding: "4px 10px",
                cursor: page <= 1 ? "not-allowed" : "pointer",
                opacity: page <= 1 ? 0.5 : 1,
              }}
            >
              Prev
            </button>
            <span
              style={{
                fontFamily: "var(--font-jetbrains)",
                fontSize: "10px",
                color: "var(--text-tertiary)",
                minWidth: "64px",
                textAlign: "center",
              }}
            >
              Page {page} / {totalPages}
            </span>
            <button
              onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
              disabled={page >= totalPages}
              style={{
                background: "transparent",
                border: "1px solid var(--border-dim)",
                borderRadius: "6px",
                color: page >= totalPages ? "var(--text-tertiary)" : "var(--text-secondary)",
                fontFamily: "var(--font-jetbrains)",
                fontSize: "10px",
                padding: "4px 10px",
                cursor: page >= totalPages ? "not-allowed" : "pointer",
                opacity: page >= totalPages ? 0.5 : 1,
              }}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

