"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  fetchDatasets,
  deleteDataset,
  DatasetSummary,
  DatasetDeleteResult,
} from "@/app/lib/api";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DatasetsPage() {
  const router = useRouter();
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const perPage = 20;

  // Delete state
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteWarning, setDeleteWarning] = useState<DatasetDeleteResult | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);

    fetchDatasets({ search: search || undefined, page, per_page: perPage })
      .then((res) => {
        if (!active) return;
        setDatasets(res.datasets as unknown as DatasetSummary[]);
        setTotal(res.total);
        setLoading(false);
      })
      .catch((e) => {
        if (!active) return;
        setError(e.message);
        setLoading(false);
      });

    return () => { active = false; };
  }, [search, page]);

  async function handleDelete(id: string, force: boolean) {
    setDeletingId(id);
    try {
      const result = await deleteDataset(id, force);
      if (!result.deleted) {
        setDeleteWarning(result);
        return;
      }
      setDatasets((prev) => prev.filter((d) => d.id !== id));
      setConfirmId(null);
      setDeleteWarning(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Delete failed");
    } finally {
      setDeletingId(null);
    }
  }

  const totalPages = Math.ceil(total / perPage);

  return (
    <div className="page-shell page-standard page-shell--md animate-fade-up">
      {/* Header */}
      <div
        className="flex items-center justify-between"
        style={{ marginBottom: "28px" }}
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
            Data Management
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
            Datasets
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
                {total} total
              </span>
            )}
          </h1>
        </div>
        <div className="datasets-page__header-actions">
          <Link
            href="/datasets/create"
            className="datasets-page__header-action datasets-page__header-action--secondary"
            style={{
              background: "var(--bg-elevated)",
              border: "1px solid var(--border)",
              color: "var(--text-secondary)",
            }}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
              <path d="M6 2V10M2 6H10" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
            </svg>
            Create Dataset
          </Link>
          <Link
            href="/datasets/upload"
            className="datasets-page__header-action datasets-page__header-action--primary"
            style={{
              background: "var(--gold)",
              color: "#08080B",
            }}
          >
            <svg width="12" height="12" viewBox="0 0 12 12" fill="none" aria-hidden="true">
              <path d="M6 2V10M2 6H10" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
            </svg>
            Upload Dataset
          </Link>
        </div>
      </div>

      {/* Search */}
      <div style={{ marginBottom: "16px" }}>
        <input
          type="text"
          placeholder="Search datasets by name or description…"
          value={search}
          onChange={(e) => { setSearch(e.target.value); setPage(1); }}
          style={{
            background: "var(--bg-surface)",
            border: "1px solid var(--border)",
            borderRadius: "8px",
            padding: "8px 14px",
            color: "var(--text-primary)",
            fontSize: "12px",
            fontFamily: "var(--font-jetbrains)",
            width: "100%",
            maxWidth: "400px",
          }}
        />
      </div>

      {/* Error */}
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

      {/* Loading */}
      {loading && (
        <div
          style={{
            color: "var(--text-tertiary)",
            fontFamily: "var(--font-jetbrains)",
            fontSize: "12px",
            padding: "48px 0",
          }}
        >
          Loading datasets…
        </div>
      )}

      {/* Empty */}
      {!loading && !error && datasets.length === 0 && (
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
            No datasets yet
          </div>
          <div style={{ fontSize: "13px", color: "var(--text-tertiary)" }}>
            Upload a CSV file to get started.
          </div>
        </div>
      )}

      {/* Dataset cards */}
      <div className="flex flex-col gap-3">
        {datasets.map((ds) => {
          const isConfirming = confirmId === ds.id;

          return (
            <div
              key={ds.id}
              className="rounded-xl transition-all duration-150"
              style={{
                background: "var(--bg-surface)",
                border: "1px solid var(--border-dim)",
                borderLeft: "3px solid var(--teal)",
                padding: "18px 20px",
                cursor: "pointer",
              }}
              onClick={() => router.push(`/datasets/${ds.id}`)}
            >
              {/* Top row */}
              <div
                className="flex items-center justify-between"
                style={{ marginBottom: "6px" }}
              >
                <div className="flex items-center gap-3 flex-wrap">
                  <span
                    style={{
                      fontFamily: "var(--font-syne)",
                      fontSize: "14px",
                      fontWeight: 600,
                      color: "var(--text-primary)",
                    }}
                  >
                    {ds.name}
                  </span>
                  {ds.tags.map((tag) => (
                    <span
                      key={tag}
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
                      {tag}
                    </span>
                  ))}
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
                    {new Date(ds.created_at).toLocaleDateString()}
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
                            {deleteWarning.dependencies?.models || 0} models,{" "}
                            {deleteWarning.dependencies?.analyses || 0} analyses linked
                          </span>
                          <button
                            onClick={() => handleDelete(ds.id, true)}
                            disabled={deletingId === ds.id}
                            style={{
                              background: "var(--error-dim)",
                              border: "1px solid var(--error)",
                              borderRadius: "4px",
                              color: "var(--error)",
                              fontFamily: "var(--font-jetbrains)",
                              fontSize: "10px",
                              padding: "2px 8px",
                              cursor: "pointer",
                              whiteSpace: "nowrap",
                            }}
                          >
                            Force Delete
                          </button>
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
                              borderRadius: "4px",
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
                            onClick={() => handleDelete(ds.id, false)}
                            disabled={deletingId === ds.id}
                            style={{
                              background: "var(--error-dim)",
                              border: "1px solid var(--error)",
                              borderRadius: "4px",
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
                      onClick={() => setConfirmId(ds.id)}
                      title="Delete dataset"
                      style={{
                        background: "transparent",
                        border: "none",
                        padding: "4px",
                        cursor: "pointer",
                        opacity: 0.4,
                        color: "var(--text-tertiary)",
                        display: "flex",
                        alignItems: "center",
                        borderRadius: "4px",
                        transition: "opacity 0.15s",
                      }}
                      onMouseEnter={(e) => { e.currentTarget.style.opacity = "1"; }}
                      onMouseLeave={(e) => { e.currentTarget.style.opacity = "0.4"; }}
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

              {/* Description */}
              {ds.description && (
                <div
                  style={{
                    fontSize: "12px",
                    color: "var(--text-tertiary)",
                    marginBottom: "8px",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {ds.description}
                </div>
              )}

              {/* Stats */}
              <div
                className="flex items-center gap-4"
                style={{
                  fontFamily: "var(--font-jetbrains)",
                  fontSize: "10px",
                  color: "var(--text-tertiary)",
                }}
              >
                <span>{ds.row_count.toLocaleString()} rows</span>
                <span>{ds.schema_info?.length || 0} columns</span>
                <span>{formatBytes(ds.file_size_bytes)}</span>
                <span>v{ds.current_version}</span>
                {ds.author && <span>by {ds.author}</span>}
              </div>
            </div>
          );
        })}
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div
          className="flex items-center justify-center gap-2"
          style={{ marginTop: "20px" }}
        >
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page === 1}
            style={{
              background: "var(--bg-elevated)",
              border: "1px solid var(--border)",
              borderRadius: "6px",
              padding: "4px 12px",
              color: page === 1 ? "var(--text-tertiary)" : "var(--text-secondary)",
              fontSize: "11px",
              fontFamily: "var(--font-jetbrains)",
              cursor: page === 1 ? "default" : "pointer",
              opacity: page === 1 ? 0.5 : 1,
            }}
          >
            Prev
          </button>
          <span
            style={{
              fontFamily: "var(--font-jetbrains)",
              fontSize: "11px",
              color: "var(--text-tertiary)",
            }}
          >
            {page} / {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page === totalPages}
            style={{
              background: "var(--bg-elevated)",
              border: "1px solid var(--border)",
              borderRadius: "6px",
              padding: "4px 12px",
              color:
                page === totalPages
                  ? "var(--text-tertiary)"
                  : "var(--text-secondary)",
              fontSize: "11px",
              fontFamily: "var(--font-jetbrains)",
              cursor: page === totalPages ? "default" : "pointer",
              opacity: page === totalPages ? 0.5 : 1,
            }}
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}

