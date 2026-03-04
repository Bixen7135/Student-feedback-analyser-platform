"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { fetchModels, deleteModel, ModelSummary } from "@/app/lib/api";
import { formatLocalizedDate, useDateTimeLocale } from "@/app/lib/i18n/date-time";
import { useI18n } from "@/app/lib/i18n/provider";

const TASK_COLORS: Record<string, string> = {
  language: "var(--teal)",
  sentiment: "var(--gold)",
  detail_level: "var(--running)",
};

const TASK_LABELS: Record<string, string> = {
  language: "Language",
  sentiment: "Sentiment",
  detail_level: "Detail Level",
};

const MODEL_TYPE_LABELS: Record<string, string> = {
  tfidf: "TF-IDF",
  char_ngram: "Char N-gram",
  xlm_roberta: "XLM-RoBERTa",
};

const PER_PAGE = 20;

function formatModelType(modelType: string): string {
  return MODEL_TYPE_LABELS[modelType] ?? modelType;
}

function formatTaskLabel(task: string): string {
  return TASK_LABELS[task] ?? task;
}

function renderConfigPreview(model: ModelSummary, t: (value: string) => string): string | null {
  if (!model.config || Object.keys(model.config).length === 0) return null;
  const config = model.config as Record<string, unknown>;

  if (model.model_type === "xlm_roberta") {
    const parts = [
      typeof config.pretrained_model === "string" && config.pretrained_model
        ? config.pretrained_model
        : null,
      config.max_seq_length != null ? `${t("Max sequence length")}: ${config.max_seq_length}` : null,
      config.batch_size != null ? `${t("Batch size")}: ${config.batch_size}` : null,
      config.epochs != null ? `${t("Epochs")}: ${config.epochs}` : null,
      config.learning_rate != null ? `${t("Learning rate")}: ${config.learning_rate}` : null,
    ].filter((part): part is string => Boolean(part));
    return parts.length > 0 ? parts.join(" | ") : null;
  }

  const parts = [
    config.max_features != null ? `${t("Max features")}: ${config.max_features}` : null,
    config.C != null ? `C ${config.C}` : null,
    config.max_iter != null ? `${t("Max iterations")}: ${config.max_iter}` : null,
  ].filter((part): part is string => Boolean(part));
  return parts.length > 0 ? parts.join(" | ") : null;
}

export default function ModelsPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const dateTimeLocale = useDateTimeLocale();
  const { t } = useI18n();
  const lineageRunId = searchParams.get("run_id") ?? "";
  const includeArchived = searchParams.get("include_archived") === "true";
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [taskFilter, setTaskFilter] = useState<string>("");
  const [typeFilter, setTypeFilter] = useState<string>("");
  const [page, setPage] = useState(1);

  // Delete state
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteWarning, setDeleteWarning] = useState<{ deleted: boolean; reason?: string; dependencies?: { analyses: number } } | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    fetchModels({
      task: taskFilter || undefined,
      model_type: typeFilter || undefined,
      run_id: lineageRunId || undefined,
      include_archived: includeArchived,
      page,
      per_page: PER_PAGE,
    })
      .then((res) => {
        if (!active) return;
        setModels(res.models);
        setTotal(res.total);
        setLoading(false);
      })
      .catch((e) => {
        if (!active) return;
        setError(e.message);
        setLoading(false);
      });
    return () => { active = false; };
  }, [taskFilter, typeFilter, page, lineageRunId, includeArchived]);

  useEffect(() => {
    setPage(1);
  }, [lineageRunId, includeArchived]);

  function renderLineage(model: ModelSummary) {
    if (!model.run_id) return null;

    if (model.run_source === "pipeline") {
      return (
        <Link
          href={`/runs/${model.run_id}`}
          onClick={(e) => e.stopPropagation()}
          style={{
            color: "var(--gold)",
            textDecoration: "none",
            borderBottom: "1px solid var(--gold-muted)",
          }}
        >
          {`produced by pipeline run ${model.run_id.slice(0, 20)}...`}
        </Link>
      );
    }

    if (model.run_source === "training") {
      return <span>{`produced by training job ${model.run_id.slice(0, 20)}...`}</span>;
    }

    return <span>{`linked run ${model.run_id.slice(0, 20)}...`}</span>;
  }

  function renderSourceBadge(model: ModelSummary) {
    if (!model.run_id) return null;

    const label =
      model.run_source === "pipeline"
        ? "pipeline run"
        : model.run_source === "training"
          ? "training job"
          : "linked run";

    return (
      <span
        className="rounded"
        style={{
          background: "var(--gold-faint)",
          border: "1px solid var(--gold-muted)",
          color: "var(--gold)",
          padding: "1px 6px",
          fontSize: "9px",
          fontFamily: "var(--font-jetbrains)",
        }}
      >
        {label}
      </span>
    );
  }

  const totalPages = Math.max(1, Math.ceil(total / PER_PAGE));
  const rangeStart = total === 0 ? 0 : (page - 1) * PER_PAGE + 1;
  const rangeEnd = total === 0 ? 0 : Math.min(total, rangeStart + Math.max(models.length, 1) - 1);

  async function handleDelete(id: string) {
    setDeletingId(id);
    try {
      const result = await deleteModel(id);
      if (!result.deleted) {
        setDeleteWarning(result);
        return;
      }
      const nextTotal = Math.max(0, total - 1);
      const nextPage = Math.min(page, Math.max(1, Math.ceil(nextTotal / PER_PAGE)));
      setTotal(nextTotal);
      setConfirmId(null);
      setDeleteWarning(null);
      if (nextPage !== page) {
        setPage(nextPage);
        return;
      }
      setModels((prev) => prev.filter((m) => m.id !== id));
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Delete failed");
    } finally {
      setDeletingId(null);
    }
  }

  const selectStyle = {
    background: "var(--bg-surface)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius-unified)",
    padding: "6px 10px",
    color: "var(--text-secondary)",
    fontSize: "11px",
    fontFamily: "var(--font-jetbrains)",
  } as const;

  return (
    <div className="page-shell page-standard page-shell--md animate-fade-up">
      <div style={{ marginBottom: "6px" }}>
        <button
          type="button"
          onClick={() => router.back()}
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "0.35rem",
            color: "var(--text-tertiary)",
            background: "none",
            border: "none",
            padding: 0,
            textDecoration: "none",
            fontFamily: "var(--font-jetbrains)",
            fontSize: "11px",
            cursor: "pointer",
          }}
        >
          <span aria-hidden="true">&larr;</span>
          <span>Back</span>
        </button>
      </div>

      {/* Header */}
      <div
        className="flex items-start justify-between gap-3"
        style={{ marginBottom: "28px", flexWrap: "wrap" }}
      >
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
            Model Management
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
            Models
            {!loading && (
              <span
                style={{
                  fontFamily: "var(--font-jetbrains)",
                  fontSize: "12px",
                  fontWeight: 400,
                  color: "var(--text-tertiary)",
                  marginLeft: "10px",
                }}
              >
                {`${total} total`}
              </span>
            )}
          </h1>
        </div>
        {lineageRunId && (
          <div className="flex items-center gap-2" style={{ flexWrap: "wrap" }}>
            <Link
              href={`/runs/${lineageRunId}`}
              className="inline-flex items-center gap-2 rounded-lg"
              style={{
                background: "transparent",
                color: "var(--text-secondary)",
                padding: "8px 14px",
                fontSize: "12px",
                fontWeight: 600,
                fontFamily: "var(--font-syne)",
                letterSpacing: "0.04em",
                textDecoration: "none",
                border: "1px solid var(--border-dim)",
              }}
            >
              Open Run
            </Link>
            <Link
              href="/runs"
              className="inline-flex items-center gap-2 rounded-lg"
              style={{
                background: "transparent",
                color: "var(--text-secondary)",
                padding: "8px 14px",
                fontSize: "12px",
                fontWeight: 600,
                fontFamily: "var(--font-syne)",
                letterSpacing: "0.04em",
                textDecoration: "none",
                border: "1px solid var(--border-dim)",
              }}
            >
              Back to Runs
            </Link>
          </div>
        )}
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3" style={{ marginBottom: "16px", flexWrap: "wrap" }}>
        <select
          value={taskFilter}
          onChange={(e) => { setTaskFilter(e.target.value); setPage(1); }}
          style={selectStyle}
        >
          <option value="">All tasks</option>
          <option value="language">{TASK_LABELS.language}</option>
          <option value="sentiment">{TASK_LABELS.sentiment}</option>
          <option value="detail_level">{TASK_LABELS.detail_level}</option>
        </select>
        <select
          value={typeFilter}
          onChange={(e) => { setTypeFilter(e.target.value); setPage(1); }}
          style={selectStyle}
        >
          <option value="">All types</option>
          <option value="tfidf">TF-IDF</option>
          <option value="char_ngram">Char N-gram</option>
          <option value="xlm_roberta">XLM-RoBERTa</option>
        </select>
      </div>

      {lineageRunId && (
        <div
          className="rounded-lg flex items-center justify-between gap-3"
          style={{
            background: "var(--gold-faint)",
            border: "1px solid var(--gold-muted)",
            padding: "10px 14px",
            marginBottom: "12px",
          }}
        >
          <span
            style={{
              color: "var(--gold)",
              fontSize: "11px",
              fontFamily: "var(--font-jetbrains)",
            }}
          >
            {`Showing ${includeArchived ? "all" : "active"} models produced by run ${lineageRunId}`}
          </span>
          <button
            onClick={() => router.push("/models")}
            style={{
              background: "transparent",
              border: "1px solid var(--gold-muted)",
              borderRadius: "var(--radius-unified)",
              color: "var(--gold)",
              fontSize: "10px",
              fontFamily: "var(--font-jetbrains)",
              padding: "2px 8px",
              cursor: "pointer",
            }}
          >
            View all models
          </button>
        </div>
      )}

      {error && (
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
          {error}
        </div>
      )}

      {loading && (
        <div
          style={{
            color: "var(--text-tertiary)",
            fontFamily: "var(--font-jetbrains)",
            fontSize: "12px",
            padding: "48px 0",
          }}
        >
          Loading models...
        </div>
      )}

      {!loading && !error && models.length === 0 && (
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
            No models registered
          </div>
          <div style={{ fontSize: "13px", color: "var(--text-tertiary)" }}>
            {lineageRunId
              ? includeArchived
                ? "This run has not produced any models."
                : "This run has not produced any active models."
              : "Train a model to see it here."}
          </div>
        </div>
      )}

      {/* Model cards */}
      <div className="flex flex-col gap-3">
        {models.map((m) => {
          const taskColor = TASK_COLORS[m.task] || "var(--text-secondary)";
          const macroF1 = m.metrics?.macro_f1 ?? m.metrics?.val_macro_f1;
          const isConfirming = confirmId === m.id;
          const isArchived = m.status !== "active";
          const configPreview = renderConfigPreview(m, t);

          return (
            <div
              key={m.id}
              className="rounded-xl transition-all duration-150"
              style={{
                background: "var(--bg-surface)",
                border: "1px solid var(--border-dim)",
                borderLeft: `3px solid ${taskColor}`,
                padding: "16px 20px",
                cursor: "pointer",
              }}
              onClick={() => !isConfirming && router.push(`/models/${m.id}`)}
            >
              <div
                className="flex items-center justify-between"
                style={{ marginBottom: "6px" }}
              >
                <div className="flex items-center gap-3">
                  <span
                    style={{
                      fontFamily: "var(--font-syne)",
                      fontSize: "14px",
                      fontWeight: 600,
                      color: "var(--text-primary)",
                    }}
                  >
                    {m.name}
                  </span>
                  <span
                    className="rounded"
                    style={{
                      background: `${taskColor}22`,
                      border: `1px solid ${taskColor}66`,
                      color: taskColor,
                      padding: "1px 6px",
                      fontSize: "9px",
                      fontFamily: "var(--font-jetbrains)",
                    }}
                  >
                    {formatTaskLabel(m.task)}
                  </span>
                  <span
                    className="rounded"
                    style={{
                      background: "var(--bg-elevated)",
                      border: "1px solid var(--border)",
                      color: "var(--text-tertiary)",
                      padding: "1px 6px",
                      fontSize: "9px",
                      fontFamily: "var(--font-jetbrains)",
                    }}
                  >
                    {formatModelType(m.model_type)}
                  </span>
                  {renderSourceBadge(m)}
                  {isArchived && (
                    <span
                      className="rounded"
                      style={{
                        background: "var(--warning-dim)",
                        border: "1px solid var(--warning)",
                        color: "var(--warning)",
                        padding: "1px 6px",
                        fontSize: "9px",
                        fontFamily: "var(--font-jetbrains)",
                      }}
                    >
                      archived
                    </span>
                  )}
                </div>

                <div
                  className="flex items-center"
                  style={{ gap: "10px" }}
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
                    {formatLocalizedDate(m.created_at, dateTimeLocale)}
                  </span>

                  {isConfirming ? (
                    <div className="flex items-center" style={{ gap: "6px" }}>
                      {deleteWarning && !deleteWarning.deleted ? (
                        <>
                          <span
                            style={{
                              fontFamily: "var(--font-jetbrains)",
                              fontSize: "10px",
                              color: "var(--warning)",
                              whiteSpace: "nowrap",
                            }}
                          >
                            {`${deleteWarning.dependencies?.analyses || 0} analyses linked`}
                          </span>
                        </>
                      ) : (
                        <>
                          <span
                            style={{
                              fontFamily: "var(--font-jetbrains)",
                              fontSize: "10px",
                              color: "var(--error)",
                            }}
                          >
                            Delete?
                          </span>
                          <button
                            onClick={() => {
                              setConfirmId(null);
                              setDeleteWarning(null);
                            }}
                            style={{
                              background: "transparent",
                              border: "1px solid var(--border-dim)",
                              borderRadius: "var(--radius-unified)",
                              color: "var(--text-tertiary)",
                              fontFamily: "var(--font-jetbrains)",
                              fontSize: "10px",
                              padding: "2px 8px",
                              cursor: "pointer",
                            }}
                          >
                            Cancel
                          </button>
                          <button
                            onClick={() => handleDelete(m.id)}
                            disabled={deletingId === m.id}
                            style={{
                              background: "var(--error-dim)",
                              border: "1px solid var(--error)",
                              borderRadius: "var(--radius-unified)",
                              color: "var(--error)",
                              fontFamily: "var(--font-jetbrains)",
                              fontSize: "10px",
                              padding: "2px 8px",
                              cursor: "pointer",
                            }}
                          >
                            Confirm
                          </button>
                        </>
                      )}
                    </div>
                  ) : (
                    <button
                      onClick={() => { if (!isArchived) setConfirmId(m.id); }}
                      title={isArchived ? "Model already archived" : "Delete model"}
                      disabled={isArchived}
                      style={{
                        background: "transparent",
                        border: "none",
                        padding: "4px",
                        cursor: isArchived ? "not-allowed" : "pointer",
                        opacity: isArchived ? 0.2 : 0.4,
                        color: "var(--text-tertiary)",
                        display: "flex",
                        alignItems: "center",
                        borderRadius: "var(--radius-unified)",
                        transition: "opacity 0.15s",
                      }}
                      onMouseEnter={(e) => { if (!isArchived) e.currentTarget.style.opacity = "1"; }}
                      onMouseLeave={(e) => { if (!isArchived) e.currentTarget.style.opacity = "0.4"; }}
                    >
                      <svg
                        width="14"
                        height="14"
                        viewBox="0 0 14 14"
                        fill="none"
                        aria-label="Delete"
                      >
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

              <div
                className="flex items-center gap-5"
                style={{
                  fontFamily: "var(--font-jetbrains)",
                  fontSize: "10px",
                  color: "var(--text-tertiary)",
                }}
              >
                <span>v{m.version}</span>
                {macroF1 != null && (
                  <span>
                    F1:{" "}
                    <span style={{ color: "var(--success)" }}>
                      {(macroF1 as number).toFixed(3)}
                    </span>
                  </span>
                )}
                {renderLineage(m)}
              </div>
              {configPreview && (
                <div
                  style={{
                    marginTop: "8px",
                    fontFamily: "var(--font-jetbrains)",
                    fontSize: "10px",
                    color: "var(--text-tertiary)",
                    overflowWrap: "anywhere",
                  }}
                >
                  {configPreview}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {!loading && !error && total > 0 && (
        <div
          style={{
            marginTop: "14px",
            display: "grid",
            alignItems: "center",
            gap: "12px",
            gridTemplateColumns: "minmax(0, 1fr) auto minmax(0, 1fr)",
          }}
        >
          <div
            style={{
              fontFamily: "var(--font-jetbrains)",
              fontSize: "11px",
              color: "var(--text-tertiary)",
            }}
          >
            {`Showing ${rangeStart}-${rangeEnd} of ${total}`}
          </div>
          <div className="flex items-center justify-center gap-2">
            <button
              onClick={() => setPage((prev) => Math.max(1, prev - 1))}
              disabled={page <= 1}
              style={{
                background: "transparent",
                border: "1px solid var(--border-dim)",
                borderRadius: "var(--radius-unified)",
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
              {`Page ${page} of ${totalPages}`}
            </span>
            <button
              onClick={() => setPage((prev) => Math.min(totalPages, prev + 1))}
              disabled={page >= totalPages}
              style={{
                background: "transparent",
                border: "1px solid var(--border-dim)",
                borderRadius: "var(--radius-unified)",
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
          <div aria-hidden="true" />
        </div>
      )}
    </div>
  );
}

