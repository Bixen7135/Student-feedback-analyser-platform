"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import {
  fetchAnalysisDetail,
  fetchAnalysisCluster,
  fetchAnalysisCorrelations,
  fetchAnalysisDescriptiveAnalytics,
  fetchAnalysisDiagnostics,
  fetchAnalysisStatus,
  updateAnalysis,
  deleteAnalysis,
  AnalysisClusterResponse,
  AnalysisCorrelationsResponse,
  AnalysisDescriptiveAnalytics,
  AnalysisDiagnosticsResponse,
  AnalysisRecord,
  AnalysisJob,
  AnalysisSummary,
  ModelApplied,
  getAnalysisExportUrl,
} from "@/app/lib/api";
import { buildFilterSearchParams } from "@/app/lib/filters";
import {
  buildCategoricalSections,
  buildClusterPoints,
  buildConfusionCells,
  buildCorrelationCells,
} from "@/app/lib/analytics";
import { ChartCard } from "@/app/components/charts/ChartCard";
import { BarListChart } from "@/app/components/charts/BarListChart";
import { Heatmap } from "@/app/components/charts/Heatmap";
import { ScatterPlot } from "@/app/components/charts/ScatterPlot";
import { formatLocalizedDateTime, useDateTimeLocale } from "@/app/lib/i18n/date-time";

const STATUS_COLORS: Record<string, string> = {
  completed: "var(--success)",
  running:   "var(--warning, #f59e0b)",
  pending:   "var(--text-tertiary)",
  failed:    "var(--error, #ef4444)",
};

const TASK_LABELS: Record<string, string> = {
  language: "Language Detection",
  sentiment: "Sentiment Classification",
  detail_level: "Detail Level",
};

function fmtDate(iso: string | null | undefined, locale: string): string {
  if (!iso) return "\u2014";
  return formatLocalizedDateTime(iso, locale, {
    year: "numeric", month: "short", day: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

export default function AnalysisDetailPage() {
  const params = useParams();
  const router = useRouter();
  const analysisId = params.analysisId as string;
  const dateTimeLocale = useDateTimeLocale();

  const [analysis, setAnalysis] = useState<AnalysisRecord | null>(null);
  const [job, setJob] = useState<AnalysisJob | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [descriptive, setDescriptive] = useState<AnalysisDescriptiveAnalytics | null>(null);
  const [correlations, setCorrelations] = useState<AnalysisCorrelationsResponse | null>(null);
  const [diagnostics, setDiagnostics] = useState<AnalysisDiagnosticsResponse | null>(null);
  const [clusters, setClusters] = useState<AnalysisClusterResponse | null>(null);

  // Edit mode
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState("");
  const [editDesc, setEditDesc] = useState("");
  const [editComments, setEditComments] = useState("");
  const [editTags, setEditTags] = useState("");
  const [saving, setSaving] = useState(false);

  // Delete
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);

  useEffect(() => {
    let active = true;
    let timeout: ReturnType<typeof setTimeout>;

    async function poll() {
      try {
        // Try status endpoint first (works for in-memory + DB)
        const statusData = await fetchAnalysisStatus(analysisId);
        if (!active) return;
        setJob(statusData);

        if (statusData.status === "completed" || statusData.status === "failed") {
          // Fetch full detail from DB once completed
          try {
            const detail = await fetchAnalysisDetail(analysisId);
            if (!active) return;
            setAnalysis(detail);
          } catch {
            // detail may not yet be in DB if status just changed — use job data
          }
          setLoading(false);
          return;
        }

        // Still running/pending — keep polling
        timeout = setTimeout(poll, 2000);
        setLoading(false);
      } catch {
        // Fall back to full detail endpoint
        try {
          const detail = await fetchAnalysisDetail(analysisId);
          if (!active) return;
          setAnalysis(detail);
          setLoading(false);
        } catch (e: unknown) {
          if (!active) return;
          setError(e instanceof Error ? e.message : String(e));
          setLoading(false);
        }
      }
    }

    poll();
    return () => {
      active = false;
      clearTimeout(timeout);
    };
  }, [analysisId]);

  // Populate edit form
  useEffect(() => {
    const src = analysis ?? (job ? {
      name: job.name, description: job.description,
      tags: job.tags, comments: "",
    } : null);
    if (src) {
      setEditName(src.name ?? "");
      setEditDesc((src as AnalysisRecord).description ?? "");
      setEditComments((src as AnalysisRecord).comments ?? "");
      setEditTags(((src as AnalysisRecord).tags ?? []).join(", "));
    }
  }, [analysis, job]);

  async function handleSave() {
    setSaving(true);
    try {
      const tagList = editTags.split(",").map((t) => t.trim()).filter(Boolean);
      const updated = await updateAnalysis(analysisId, {
        name: editName,
        description: editDesc,
        tags: tagList,
        comments: editComments,
      });
      setAnalysis(updated);
      setEditing(false);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  async function handleDelete() {
    if (!confirmDelete) { setConfirmDelete(true); return; }
    setDeleting(true);
    try {
      await deleteAnalysis(analysisId);
      router.push("/analyses");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      setDeleting(false);
    }
  }

  const currentStatus = job?.status ?? analysis?.status;
  const summary: AnalysisSummary | null =
    (job?.result_summary as AnalysisSummary | null) ??
    (analysis?.result_summary as AnalysisSummary | null) ?? null;
  const modelsApplied = Array.isArray(summary?.models_applied)
    ? summary.models_applied
    : [];

  const name = analysis?.name ?? job?.name ?? "";
  const description = analysis?.description ?? job?.description ?? "";
  const tags = analysis?.tags ?? job?.tags ?? [];
  const comments = (analysis as AnalysisRecord | null)?.comments ?? "";
  const createdAt = analysis?.created_at ?? job?.started_at;
  const modelIds = analysis?.model_ids ?? job?.model_ids ?? [];
  const analyticsTask =
    modelsApplied.find((item) => !item.error)?.task ?? "sentiment";

  useEffect(() => {
    if (currentStatus !== "completed") return;
    let active = true;
    Promise.allSettled([
      fetchAnalysisDescriptiveAnalytics(analysisId),
      fetchAnalysisCorrelations(analysisId),
      fetchAnalysisDiagnostics(analysisId, { task: analyticsTask }),
      fetchAnalysisCluster(analysisId, { method: "kmeans", k: 4, reuse_embeddings: true }),
    ]).then(([a, b, c, d]) => {
      if (!active) return;
      if (a.status === "fulfilled") setDescriptive(a.value);
      if (b.status === "fulfilled") setCorrelations(b.value);
      if (c.status === "fulfilled") setDiagnostics(c.value);
      if (c.status === "rejected") setDiagnostics(null);
      if (d.status === "fulfilled") setClusters(d.value);
    });
    return () => {
      active = false;
    };
  }, [analysisId, analyticsTask, currentStatus]);

  const categoricalSections = descriptive ? buildCategoricalSections(descriptive.summary, 3, 6) : [];
  const correlationCells = correlations ? buildCorrelationCells(correlations.correlations) : [];
  const confusionCells = diagnostics ? buildConfusionCells(diagnostics.diagnostics) : [];
  const clusterPoints = clusters ? buildClusterPoints(clusters) : [];

  function resultsHref(filter?: { col: string; op: "eq"; val: string }): string {
    const params = buildFilterSearchParams({
      filters: filter ? [filter] : [],
    });
    const qs = params.toString();
    return `/analyses/${analysisId}/results${qs ? `?${qs}` : ""}`;
  }

  if (loading) {
    return (
      <div className="page-shell page-standard page-shell--md" style={{ color: "var(--text-tertiary)", fontSize: "13px" }}>
        Loading analysis…
      </div>
    );
  }

  if (error && !analysis && !job) {
    return (
      <div className="page-shell page-standard page-shell--md" style={{ color: "var(--error, #ef4444)", fontSize: "13px" }}>
        {error}
      </div>
    );
  }

  return (
    <div className="page-shell page-standard page-shell--md animate-fade-up">
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

      {/* Header */}
      <div className="analysis-detail__header" style={{ marginBottom: "24px" }}>
        <div className="analysis-detail__header-main">
          {editing ? (
            <input
              type="text"
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              placeholder="Analysis name…"
              style={{
                fontSize: "20px",
                fontFamily: "var(--font-syne)",
                fontWeight: 700,
                color: "var(--text-primary)",
                background: "transparent",
                border: "none",
                borderBottom: "1px solid var(--gold)",
                outline: "none",
                width: "min(100%, 24rem)",
                padding: "2px 0",
              }}
            />
          ) : (
            <h1
              style={{
                fontFamily: "var(--font-syne)",
                fontWeight: 700,
                fontSize: "clamp(20px, 2vw, 24px)",
                color: "var(--text-primary)",
                margin: 0,
              }}
            >
              {name || <span style={{ color: "var(--text-tertiary)" }}>{analysisId.slice(0, 20)}…</span>}
            </h1>
          )}
          <div className="flex items-center gap-3" style={{ marginTop: "6px", flexWrap: "wrap" }}>
            <div className="flex items-center gap-1.5">
              <span
                style={{
                  width: "7px", height: "7px", borderRadius: "50%",
                  background: STATUS_COLORS[currentStatus ?? ""] ?? "var(--text-tertiary)",
                  display: "inline-block",
                  animation: currentStatus === "running" ? "pulse 1.5s infinite" : "none",
                }}
              />
              <span style={{ fontSize: "12px", color: "var(--text-secondary)", textTransform: "capitalize" }}>
                {currentStatus ?? "unknown"}
              </span>
            </div>
            <span style={{ fontSize: "12px", color: "var(--text-tertiary)" }}>
              {fmtDate(createdAt, dateTimeLocale)}
            </span>
            {tags.length > 0 && (
              <div className="flex gap-1">
                {tags.map((t) => (
                  <span
                    key={t}
                    style={{
                      fontSize: "10px", padding: "2px 7px", borderRadius: "var(--radius-unified)",
                      background: "var(--bg-surface)", border: "1px solid var(--border-dim)",
                      color: "var(--text-tertiary)",
                    }}
                  >
                    {t}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="analysis-detail__actions">
          {currentStatus === "completed" && !editing && (
            <div className="analysis-detail__action-group analysis-detail__action-group--primary">
              <Link
                href={`/analyses/${analysisId}/results`}
                className="analysis-detail__action analysis-detail__action--primary"
              >
                View Results
              </Link>
              <a
                href={getAnalysisExportUrl(analysisId, "csv")}
                download
                className="analysis-detail__action analysis-detail__action--secondary"
              >
                Export CSV
              </a>
            </div>
          )}
          <div className="analysis-detail__action-group analysis-detail__action-group--manage">
            {!editing ? (
            <button
              type="button"
              onClick={() => setEditing(true)}
              className="analysis-detail__action analysis-detail__action--ghost"
            >
              Edit
            </button>
          ) : (
            <>
              <button
                type="button"
                onClick={() => setEditing(false)}
                className="analysis-detail__action analysis-detail__action--ghost"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleSave}
                disabled={saving}
                className="analysis-detail__action analysis-detail__action--primary"
              >
                {saving ? "Saving…" : "Save"}
              </button>
            </>
          )}
          <button
            type="button"
            onClick={handleDelete}
            disabled={deleting}
            className={`analysis-detail__action analysis-detail__action--danger${confirmDelete ? " is-confirm" : ""}`}
          >
            {deleting ? "…" : confirmDelete ? "Confirm Delete?" : "Delete"}
          </button>
            {confirmDelete && (
            <button
              type="button"
              onClick={() => setConfirmDelete(false)}
              className="analysis-detail__action analysis-detail__action--plain"
            >
              Cancel delete
            </button>
            )}
          </div>
        </div>
      </div>

      {/* Running indicator */}
      {(currentStatus === "running" || currentStatus === "pending") && (
        <div
          style={{
            padding: "16px 20px",
            borderRadius: "var(--radius-unified)",
            border: "1px solid var(--warning, #f59e0b)",
            background: "rgba(245,158,11,0.07)",
            color: "var(--warning, #f59e0b)",
            fontSize: "13px",
            marginBottom: "24px",
          }}
        >
          Analysis {currentStatus === "pending" ? "queued" : "in progress"}…
          Results will appear here once complete. This page refreshes automatically.
        </div>
      )}

      {/* Failed indicator */}
      {currentStatus === "failed" && (
        <div
          style={{
            padding: "14px 18px",
            borderRadius: "var(--radius-unified)",
            border: "1px solid rgba(239,68,68,0.3)",
            background: "rgba(239,68,68,0.07)",
            color: "var(--error, #ef4444)",
            fontSize: "13px",
            marginBottom: "24px",
          }}
        >
          <strong>Analysis failed.</strong>
          {job?.error && <div style={{ marginTop: "6px", fontFamily: "var(--font-jetbrains)", fontSize: "11px" }}>{job.error}</div>}
        </div>
      )}

      {error && (
        <div style={{ color: "var(--error, #ef4444)", fontSize: "13px", marginBottom: "16px" }}>{error}</div>
      )}

      {/* Metadata edit form */}
      {editing && (
        <div
          style={{
            padding: "20px",
            border: "1px solid var(--gold-muted)",
            borderRadius: "var(--radius-unified)",
            background: "var(--gold-faint)",
            marginBottom: "24px",
          }}
        >
          <div className="flex flex-col gap-3">
            <div>
              <label style={{ fontSize: "12px", color: "var(--text-secondary)", display: "block", marginBottom: "4px" }}>
                Description
              </label>
              <textarea
                value={editDesc}
                onChange={(e) => setEditDesc(e.target.value)}
                rows={2}
                style={{
                  width: "100%", padding: "8px 10px",
                  border: "1px solid var(--border-dim)", borderRadius: "var(--radius-unified)",
                  background: "var(--bg-surface)", color: "var(--text-primary)",
                  fontSize: "13px", resize: "vertical", fontFamily: "inherit", boxSizing: "border-box",
                }}
              />
            </div>
            <div>
              <label style={{ fontSize: "12px", color: "var(--text-secondary)", display: "block", marginBottom: "4px" }}>
                Tags (comma-separated)
              </label>
              <input
                type="text"
                value={editTags}
                onChange={(e) => setEditTags(e.target.value)}
                style={{
                  width: "100%", padding: "7px 10px",
                  border: "1px solid var(--border-dim)", borderRadius: "var(--radius-unified)",
                  background: "var(--bg-surface)", color: "var(--text-primary)",
                  fontSize: "13px", boxSizing: "border-box",
                }}
              />
            </div>
            <div>
              <label style={{ fontSize: "12px", color: "var(--text-secondary)", display: "block", marginBottom: "4px" }}>
                Comments / Notes
              </label>
              <textarea
                value={editComments}
                onChange={(e) => setEditComments(e.target.value)}
                rows={3}
                style={{
                  width: "100%", padding: "8px 10px",
                  border: "1px solid var(--border-dim)", borderRadius: "var(--radius-unified)",
                  background: "var(--bg-surface)", color: "var(--text-primary)",
                  fontSize: "13px", resize: "vertical", fontFamily: "inherit", boxSizing: "border-box",
                }}
              />
            </div>
          </div>
        </div>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 18rem), 1fr))",
          gap: "16px",
          marginBottom: "24px",
        }}
      >
        {/* Dataset info */}
        <div
          style={{
            padding: "16px 20px",
            border: "1px solid var(--border-dim)",
            borderRadius: "var(--radius-unified)",
            background: "var(--bg-surface)",
          }}
        >
          <div style={{ fontSize: "11px", color: "var(--text-tertiary)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "8px" }}>
            Dataset
          </div>
          {analysis?.dataset_id || job?.dataset_id ? (
            <Link
              href={`/datasets/${analysis?.dataset_id ?? job?.dataset_id}`}
              style={{ color: "var(--text-primary)", textDecoration: "none", fontWeight: 500, fontSize: "13px" }}
            >
              {analysis?.dataset_id ?? job?.dataset_id}
            </Link>
          ) : (
            <span style={{ color: "var(--text-tertiary)", fontSize: "13px" }}>—</span>
          )}
          {summary?.n_rows != null && (
            <div style={{ fontSize: "12px", color: "var(--text-tertiary)", marginTop: "4px" }}>
              {summary.n_rows.toLocaleString()} rows
            </div>
          )}
        </div>

        {/* Models */}
        <div
          style={{
            padding: "16px 20px",
            border: "1px solid var(--border-dim)",
            borderRadius: "var(--radius-unified)",
            background: "var(--bg-surface)",
          }}
        >
          <div style={{ fontSize: "11px", color: "var(--text-tertiary)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "8px" }}>
            Models Applied ({modelIds.length})
          </div>
          <div className="flex flex-col gap-1">
            {modelIds.slice(0, 4).map((mid) => (
              <Link
                key={mid}
                href={`/models/${mid}`}
                style={{ fontSize: "12px", color: "var(--text-secondary)", textDecoration: "none" }}
              >
                {mid.slice(0, 20)}…
              </Link>
            ))}
            {modelIds.length > 4 && (
              <span style={{ fontSize: "11px", color: "var(--text-tertiary)" }}>+{modelIds.length - 4} more</span>
            )}
          </div>
        </div>
      </div>

      {/* Results summary */}
      {summary && modelsApplied.length > 0 && (
        <div>
          <h2 style={{ fontSize: "15px", fontWeight: 600, color: "var(--text-primary)", marginBottom: "12px" }}>
            Prediction Summary
          </h2>
          <div
            style={{
              padding: "10px 14px",
              borderRadius: "var(--radius-unified)",
              background: "var(--gold-faint)",
              border: "1px solid var(--gold-muted)",
              fontSize: "11px",
              color: "var(--text-secondary)",
              marginBottom: "16px",
            }}
          >
            Aggregate distributions only. Not for individual-level decisions. No causal claims.
          </div>

          <div className="flex flex-col gap-4">
            {modelsApplied.map((m: ModelApplied) => (
              <div
                key={m.model_id}
                style={{
                  padding: "16px 20px",
                  border: "1px solid var(--border-dim)",
                  borderRadius: "var(--radius-unified)",
                  background: "var(--bg-surface)",
                }}
              >
                <div className="flex items-start justify-between" style={{ marginBottom: "12px" }}>
                  <div>
                    <Link
                      href={`/models/${m.model_id}`}
                      style={{ fontWeight: 600, fontSize: "13px", color: "var(--text-primary)", textDecoration: "none" }}
                    >
                      {m.model_name}
                    </Link>
                    <div style={{ fontSize: "11px", color: "var(--text-tertiary)", marginTop: "2px" }}>
                      {TASK_LABELS[m.task] ?? m.task} · {m.model_type} · {m.n_predicted.toLocaleString()} predictions
                    </div>
                  </div>
                  {m.error && (
                    <span style={{ fontSize: "11px", color: "var(--error, #ef4444)", padding: "2px 8px", background: "rgba(239,68,68,0.08)", borderRadius: "var(--radius-unified)" }}>
                      Error
                    </span>
                  )}
                </div>

                {m.error ? (
                  <div style={{ fontSize: "12px", color: "var(--error, #ef4444)", fontFamily: "var(--font-jetbrains)" }}>
                    {m.error}
                  </div>
                ) : (
                  <div>
                    {/* Distribution bars */}
                    {Object.entries(m.class_distribution)
                      .sort(([, a], [, b]) => b - a)
                      .map(([cls, frac]) => (
                        <div key={cls} style={{ marginBottom: "6px" }}>
                          <div className="flex items-center justify-between" style={{ marginBottom: "3px" }}>
                            <span style={{ fontSize: "12px", color: "var(--text-secondary)" }}>{cls}</span>
                            <span style={{ fontSize: "12px", color: "var(--text-tertiary)", fontFamily: "var(--font-jetbrains)" }}>
                              {pct(frac)}
                            </span>
                          </div>
                          <div
                            style={{
                              height: "6px", borderRadius: "3px",
                              background: "var(--border-dim)", overflow: "hidden",
                            }}
                          >
                            <div
                              style={{
                                height: "100%",
                                width: pct(frac),
                                background: "var(--gold)",
                                borderRadius: "3px",
                              }}
                            />
                          </div>
                        </div>
                      ))}
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Actions */}
          <div className="flex gap-3" style={{ marginTop: "20px", flexWrap: "wrap" }}>
            <Link
              href={`/analyses/${analysisId}/results`}
              style={{
                padding: "9px 18px",
                background: "var(--gold)",
                color: "#000",
                borderRadius: "var(--radius-unified)",
                fontSize: "13px",
                fontWeight: 600,
                textDecoration: "none",
              }}
            >
              Browse Results Table
            </Link>
            <a
              href={getAnalysisExportUrl(analysisId, "csv")}
              download
              style={{
                padding: "9px 18px",
                border: "1px solid var(--border-dim)",
                borderRadius: "var(--radius-unified)",
                fontSize: "13px",
                color: "var(--text-secondary)",
                textDecoration: "none",
              }}
            >
              Export CSV
            </a>
            <a
              href={getAnalysisExportUrl(analysisId, "json")}
              download
              style={{
                padding: "9px 18px",
                border: "1px solid var(--border-dim)",
                borderRadius: "var(--radius-unified)",
                fontSize: "13px",
                color: "var(--text-secondary)",
                textDecoration: "none",
              }}
            >
              Export JSON
            </a>
          </div>
        </div>
      )}

      {currentStatus === "completed" && (
        <div style={{ marginTop: "24px" }}>
          <h2 style={{ fontSize: "15px", fontWeight: 600, color: "var(--text-primary)", marginBottom: "12px" }}>
            Analytics Snapshot
          </h2>
          <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            {categoricalSections.length > 0 && (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "16px" }}>
                {categoricalSections.map((section) => (
                  <ChartCard key={section.column} title={section.column} subtitle="Click through to open the filtered results view.">
                    <BarListChart
                      series={section.series}
                      onSelect={(value) => {
                        router.push(resultsHref({ col: section.column, op: "eq", val: value }));
                      }}
                    />
                  </ChartCard>
                ))}
              </div>
            )}

            {correlationCells.length > 0 && (
              <ChartCard title="Correlation Heatmap" subtitle="Pairwise associations across the analysis output.">
                <Heatmap cells={correlationCells} />
              </ChartCard>
            )}

            {diagnostics && (
              <ChartCard title={`${diagnostics.task} diagnostics`} subtitle={`Label column: ${diagnostics.label_col}`}>
                <Heatmap cells={confusionCells} />
              </ChartCard>
            )}

            {clusters && (
              <ChartCard title="Embeddings" subtitle="Clustered 2D projection of the result rows.">
                <ScatterPlot points={clusterPoints} />
              </ChartCard>
            )}
          </div>
        </div>
      )}

      {/* Comments */}
      {comments && !editing && (
        <div
          style={{
            marginTop: "24px",
            padding: "16px 20px",
            border: "1px solid var(--border-dim)",
            borderRadius: "var(--radius-unified)",
          }}
        >
          <div style={{ fontSize: "11px", color: "var(--text-tertiary)", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: "8px" }}>
            Notes
          </div>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)", margin: 0, whiteSpace: "pre-wrap" }}>
            {comments}
          </p>
        </div>
      )}

      {description && !editing && (
        <div style={{ marginTop: "16px", fontSize: "13px", color: "var(--text-tertiary)" }}>
          {description}
        </div>
      )}
    </div>
  );
}
