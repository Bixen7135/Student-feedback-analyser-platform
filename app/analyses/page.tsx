"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { fetchAnalyses, deleteAnalysis, AnalysisRecord } from "@/app/lib/api";

const STATUS_COLORS: Record<string, string> = {
  completed: "var(--success)",
  running:   "var(--warning, #f59e0b)",
  pending:   "var(--text-tertiary)",
  failed:    "var(--error, #ef4444)",
};

function fmtDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    year: "numeric", month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

export default function AnalysesPage() {
  const [analyses, setAnalyses] = useState<AnalysisRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const perPage = 20;

  // Filters
  const [statusFilter, setStatusFilter] = useState("");
  const [datasetFilter, setDatasetFilter] = useState("");

  // Delete state
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [confirmId, setConfirmId] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    let timeout: ReturnType<typeof setTimeout>;

    async function load() {
      try {
        const resp = await fetchAnalyses({
          status: statusFilter || undefined,
          dataset_id: datasetFilter || undefined,
          sort: "created_at",
          order: "desc",
          page,
          per_page: perPage,
        });
        if (!active) return;
        setAnalyses(resp.analyses);
        setTotal(resp.total);
        setError(null);

        // Keep polling while any analysis is running/pending
        const anyActive = resp.analyses.some(
          (a) => a.status === "running" || a.status === "pending"
        );
        if (anyActive) {
          timeout = setTimeout(load, 3000);
        }
      } catch (e: unknown) {
        if (!active) return;
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (active) setLoading(false);
      }
    }

    setLoading(true);
    load();
    return () => {
      active = false;
      clearTimeout(timeout);
    };
  }, [page, statusFilter, datasetFilter]);

  async function handleDelete(id: string) {
    if (confirmId !== id) {
      setConfirmId(id);
      return;
    }
    setDeletingId(id);
    setConfirmId(null);
    try {
      await deleteAnalysis(id);
      const nextTotal = Math.max(0, total - 1);
      const nextPage = Math.min(page, Math.max(1, Math.ceil(nextTotal / perPage)));
      setTotal(nextTotal);
      if (nextPage !== page) {
        setPage(nextPage);
        return;
      }
      setAnalyses((prev) => prev.filter((a) => a.id !== id));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setDeletingId(null);
    }
  }

  const totalPages = Math.max(1, Math.ceil(total / perPage));
  const rangeStart = total === 0 ? 0 : (page - 1) * perPage + 1;
  const rangeEnd = total === 0 ? 0 : Math.min(total, rangeStart + Math.max(analyses.length, 1) - 1);

  return (
    <div className="page-shell page-standard page-shell--lg animate-fade-up">
      {/* Header */}
      <div className="flex items-center justify-between" style={{ marginBottom: "28px" }}>
        <div>
          <h1
            style={{
              fontFamily: "var(--font-syne)",
              fontWeight: 700,
              fontSize: "22px",
              color: "var(--text-primary)",
              margin: 0,
            }}
          >
            Analysis History
          </h1>
          <p style={{ color: "var(--text-tertiary)", fontSize: "13px", marginTop: "4px" }}>
            Batch analyses — {total} total
          </p>
        </div>
        <div className="analyses-page__header-actions">
          <Link
            href="/analyses/compare"
            className="analyses-page__header-action analyses-page__header-action--secondary"
            style={{
              border: "1px solid var(--border-dim)",
              color: "var(--text-secondary)",
            }}
          >
            Compare
          </Link>
          <Link
            href="/analyses/new"
            className="analyses-page__header-action analyses-page__header-action--primary"
            style={{
              background: "var(--gold)",
              color: "#000",
            }}
          >
            + New Analysis
          </Link>
        </div>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3" style={{ marginBottom: "20px" }}>
        <select
          value={statusFilter}
          onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }}
          style={{
            padding: "7px 12px",
            border: "1px solid var(--border-dim)",
            borderRadius: "6px",
            background: "var(--bg-surface)",
            color: "var(--text-primary)",
            fontSize: "13px",
          }}
        >
          <option value="">All statuses</option>
          <option value="completed">Completed</option>
          <option value="running">Running</option>
          <option value="pending">Pending</option>
          <option value="failed">Failed</option>
        </select>
        <input
          type="text"
          placeholder="Search by dataset ID…"
          value={datasetFilter}
          onChange={(e) => { setDatasetFilter(e.target.value); setPage(1); }}
          style={{
            padding: "7px 12px",
            border: "1px solid var(--border-dim)",
            borderRadius: "6px",
            background: "var(--bg-surface)",
            color: "var(--text-primary)",
            fontSize: "13px",
            width: "220px",
          }}
        />
      </div>

      {/* Disclaimer */}
      <div
        style={{
          padding: "10px 14px",
          borderRadius: "6px",
          background: "var(--gold-faint)",
          border: "1px solid var(--gold-muted)",
          fontSize: "12px",
          color: "var(--text-secondary)",
          marginBottom: "20px",
        }}
      >
        Batch analysis only. Results are aggregate summaries — not for individual-level decisions.
        No causal claims are implied.
      </div>

      {/* Error */}
      {error && (
        <div
          style={{
            padding: "12px 16px",
            borderRadius: "6px",
            background: "rgba(239,68,68,0.08)",
            border: "1px solid rgba(239,68,68,0.3)",
            color: "var(--error, #ef4444)",
            fontSize: "13px",
            marginBottom: "16px",
          }}
        >
          {error}
        </div>
      )}

      {/* Table */}
      {loading ? (
        <div style={{ color: "var(--text-tertiary)", fontSize: "13px", padding: "40px 0", textAlign: "center" }}>
          Loading analyses…
        </div>
      ) : analyses.length === 0 ? (
        <div
          style={{
            padding: "60px 24px",
            textAlign: "center",
            border: "1px dashed var(--border-dim)",
            borderRadius: "8px",
          }}
        >
          <div style={{ fontSize: "13px", color: "var(--text-tertiary)", marginBottom: "16px" }}>
            No analyses yet.
          </div>
          <Link
            href="/analyses/new"
            style={{
              padding: "8px 20px",
              background: "var(--gold)",
              color: "#000",
              borderRadius: "6px",
              fontSize: "13px",
              fontWeight: 600,
              textDecoration: "none",
            }}
          >
            Run your first analysis
          </Link>
        </div>
      ) : (
        <div
          style={{
            border: "1px solid var(--border-dim)",
            borderRadius: "8px",
            overflowX: "auto",
            overflowY: "hidden",
          }}
        >
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr style={{ background: "var(--bg-surface-2, var(--bg-surface))" }}>
                {["Name", "Dataset", "Models", "Status", "Created", "Actions"].map((h) => (
                  <th
                    key={h}
                    style={{
                      padding: "10px 16px",
                      textAlign: "left",
                      fontSize: "11px",
                      fontFamily: "var(--font-syne)",
                      fontWeight: 700,
                      letterSpacing: "0.08em",
                      textTransform: "uppercase",
                      color: "var(--text-tertiary)",
                      borderBottom: "1px solid var(--border-dim)",
                    }}
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {analyses.map((a, i) => {
                const summary = a.result_summary as { n_rows?: number; models_applied?: { task: string }[] } | null;
                const nModels = a.model_ids?.length ?? 0;
                const tasks = summary?.models_applied?.map((m) => m.task) ?? [];
                const uniqueTasks = [...new Set(tasks)];

                return (
                  <tr
                    key={a.id}
                    style={{
                      borderBottom: i < analyses.length - 1 ? "1px solid var(--border-dim)" : "none",
                      background: "transparent",
                    }}
                  >
                    <td style={{ padding: "12px 16px" }}>
                      <Link
                        href={`/analyses/${a.id}`}
                        style={{
                          color: "var(--text-primary)",
                          textDecoration: "none",
                          fontWeight: 500,
                          fontSize: "13px",
                        }}
                      >
                        {a.name || <span style={{ color: "var(--text-tertiary)" }}>{a.id.slice(0, 16)}</span>}
                      </Link>
                      {a.description && (
                        <div style={{ fontSize: "11px", color: "var(--text-tertiary)", marginTop: "2px" }}>
                          {a.description.slice(0, 60)}{a.description.length > 60 ? "…" : ""}
                        </div>
                      )}
                    </td>
                    <td style={{ padding: "12px 16px" }}>
                      {a.dataset_id ? (
                        <Link
                          href={`/datasets/${a.dataset_id}`}
                          style={{
                            fontFamily: "var(--font-jetbrains)",
                            fontSize: "11px",
                            color: "var(--text-secondary)",
                            textDecoration: "none",
                          }}
                        >
                          {a.dataset_id.slice(0, 12)}…
                          {a.dataset_version != null && (
                            <span style={{ color: "var(--text-tertiary)" }}> v{a.dataset_version}</span>
                          )}
                        </Link>
                      ) : (
                        <span style={{ color: "var(--text-tertiary)", fontSize: "12px" }}>—</span>
                      )}
                    </td>
                    <td style={{ padding: "12px 16px" }}>
                      <div style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
                        {nModels} model{nModels !== 1 ? "s" : ""}
                      </div>
                      {uniqueTasks.length > 0 && (
                        <div className="flex gap-1" style={{ marginTop: "4px", flexWrap: "wrap" }}>
                          {uniqueTasks.map((t) => (
                            <span
                              key={t}
                              style={{
                                fontSize: "10px",
                                padding: "1px 6px",
                                borderRadius: "4px",
                                background: "var(--bg-surface)",
                                border: "1px solid var(--border-dim)",
                                color: "var(--text-tertiary)",
                              }}
                            >
                              {t}
                            </span>
                          ))}
                        </div>
                      )}
                    </td>
                    <td style={{ padding: "12px 16px" }}>
                      <div className="flex items-center gap-1.5">
                        <span
                          style={{
                            width: "6px",
                            height: "6px",
                            borderRadius: "50%",
                            background: STATUS_COLORS[a.status] ?? "var(--text-tertiary)",
                            display: "inline-block",
                            flexShrink: 0,
                          }}
                        />
                        <span style={{ fontSize: "12px", color: "var(--text-secondary)", textTransform: "capitalize" }}>
                          {a.status}
                        </span>
                      </div>
                      {summary?.n_rows != null && (
                        <div style={{ fontSize: "11px", color: "var(--text-tertiary)", marginTop: "2px" }}>
                          {summary.n_rows.toLocaleString()} rows
                        </div>
                      )}
                    </td>
                    <td style={{ padding: "12px 16px", whiteSpace: "nowrap" }}>
                      <span style={{ fontSize: "12px", color: "var(--text-tertiary)" }}>
                        {fmtDate(a.created_at)}
                      </span>
                    </td>
                    <td style={{ padding: "12px 16px", minWidth: "220px" }}>
                      <div className="analyses-page__row-actions">
                        <div className={`analyses-page__row-actions-main${a.status === "completed" ? " is-complete" : ""}`}>
                          <Link
                            href={`/analyses/${a.id}`}
                            className="analyses-page__row-action analyses-page__row-action--primary"
                          >
                            Open
                          </Link>
                          {a.status === "completed" && (
                            <Link
                              href={`/analyses/${a.id}/results`}
                              className="analyses-page__row-action analyses-page__row-action--secondary"
                            >
                              Results
                            </Link>
                          )}
                          <button
                            type="button"
                            onClick={() => handleDelete(a.id)}
                            disabled={deletingId === a.id}
                            className={`analyses-page__row-action analyses-page__row-action--danger${confirmId === a.id ? " is-confirm" : ""}`}
                          >
                            {confirmId === a.id ? "Confirm Delete?" : deletingId === a.id ? "..." : "Delete"}
                          </button>
                        </div>
                        {confirmId === a.id && (
                          <div className="analyses-page__row-actions-manage">
                            <button
                              type="button"
                              onClick={() => setConfirmId(null)}
                              className="analyses-page__row-action analyses-page__row-action--plain"
                            >
                              Cancel
                            </button>
                          </div>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination */}
      {!loading && !error && total > 0 && (
        <div
          className="flex items-center justify-between gap-3"
          style={{
            display: "grid",
            gridTemplateColumns: "minmax(0, 1fr) auto minmax(0, 1fr)",
            alignItems: "center",
            marginTop: "16px",
            gap: "12px",
          }}
        >
          <div
            style={{
              fontFamily: "var(--font-jetbrains)",
              fontSize: "11px",
              color: "var(--text-tertiary)",
              justifySelf: "start",
            }}
          >
            Showing {rangeStart}-{rangeEnd} of {total}
          </div>
          <div className="flex items-center gap-2" style={{ gridColumn: 2, justifyContent: "center" }}>
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

