"use client";

import { useEffect, useState, useCallback } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import {
  fetchAnalysisResults,
  fetchAnalysisDetail,
  fetchAnalysisAnomalies,
  fetchSavedFilters,
  createSavedFilter,
  deleteSavedFilter,
  getAnalysisExportUrl,
  getFilteredExportUrl,
  fetchAnalysisDistributions,
  fetchAnalysisSegmentStats,
  fetchCrossCompare,
  fetchAnalyses,
  AnalysisResults,
  AnalysisRecord,
  AnalysisSummary,
  FilterRule,
  SavedFilter,
  AnomalyRow,
  SegmentGroup,
  CrossCompareData,
} from "@/app/lib/api";
import { DistributionChart } from "@/app/components/charts/DistributionChart";
import { SegmentTable } from "@/app/components/charts/SegmentTable";

const PAGE_SIZE = 50;

const FILTER_OPS: { value: FilterRule["op"]; label: string }[] = [
  { value: "eq", label: "=" },
  { value: "ne", label: "≠" },
  { value: "contains", label: "contains" },
  { value: "gt", label: ">" },
  { value: "lt", label: "<" },
  { value: "gte", label: "≥" },
  { value: "lte", label: "≤" },
];

// ---------------------------------------------------------------------------
// Small reusable style helpers
// ---------------------------------------------------------------------------

const inputStyle: React.CSSProperties = {
  padding: "6px 10px",
  border: "1px solid var(--border-dim)",
  borderRadius: "5px",
  background: "var(--bg-surface)",
  color: "var(--text-primary)",
  fontSize: "12px",
};

const btnOutline: React.CSSProperties = {
  padding: "6px 12px",
  border: "1px solid var(--border-dim)",
  background: "transparent",
  borderRadius: "5px",
  fontSize: "12px",
  color: "var(--text-secondary)",
  cursor: "pointer",
};

const btnGold: React.CSSProperties = {
  padding: "6px 14px",
  background: "var(--gold)",
  color: "#000",
  border: "none",
  borderRadius: "5px",
  fontSize: "12px",
  fontWeight: 600,
  cursor: "pointer",
};

// ---------------------------------------------------------------------------
// Page component
// ---------------------------------------------------------------------------

export default function AnalysisResultsPage() {
  const params = useParams();
  const analysisId = params.analysisId as string;

  // Active tab: "table" | "analytics"
  const [activeTab, setActiveTab] = useState<"table" | "analytics">("table");

  const [analysis, setAnalysis] = useState<AnalysisRecord | null>(null);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(0);

  // Sort
  const [sortCol, setSortCol] = useState("");
  const [sortOrder, setSortOrder] = useState<"asc" | "desc">("asc");

  // Multi-filter rules
  const [filters, setFilters] = useState<FilterRule[]>([]);
  const [columns, setColumns] = useState<string[]>([]);

  // Full-text search
  const [search, setSearch] = useState("");
  const [searchInput, setSearchInput] = useState("");

  // Show anomalies panel
  const [showAnomalies, setShowAnomalies] = useState(false);
  const [anomalies, setAnomalies] = useState<AnomalyRow[] | null>(null);
  const [anomalyTotal, setAnomalyTotal] = useState(0);
  const [anomalyThreshold, setAnomalyThreshold] = useState(0.6);
  const [anomalyLoading, setAnomalyLoading] = useState(false);

  // Saved filters
  const [savedFilters, setSavedFilters] = useState<SavedFilter[]>([]);
  const [showSavedPanel, setShowSavedPanel] = useState(false);
  const [saveNameInput, setSaveNameInput] = useState("");
  const [savingFilter, setSavingFilter] = useState(false);

  // ---------------------------------------------------------------------------
  // Analytics tab state
  // ---------------------------------------------------------------------------

  // Column selector for distributions
  const [distCols, setDistCols] = useState<string[]>([]);
  const [distData, setDistData] = useState<Record<string, Record<string, number>> | null>(null);
  const [distLoading, setDistLoading] = useState(false);
  const [distError, setDistError] = useState<string | null>(null);

  // Segment stats
  const [segGroupBy, setSegGroupBy] = useState("");
  const [segMetricCol, setSegMetricCol] = useState("");
  const [segGroups, setSegGroups] = useState<SegmentGroup[] | null>(null);
  const [segLoading, setSegLoading] = useState(false);
  const [segError, setSegError] = useState<string | null>(null);

  // Cross-compare
  const [allAnalyses, setAllAnalyses] = useState<AnalysisRecord[]>([]);
  const [compareIds, setCompareIds] = useState<string[]>([]);
  const [compareCols, setCompareCols] = useState<string[]>([]);
  const [compareData, setCompareData] = useState<CrossCompareData | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareError, setCompareError] = useState<string | null>(null);

  // ---------------------------------------------------------------------------
  // Effects
  // ---------------------------------------------------------------------------

  // Get analysis metadata
  useEffect(() => {
    fetchAnalysisDetail(analysisId)
      .then(setAnalysis)
      .catch((e) => setError(e.message));
  }, [analysisId]);

  // Load saved filters for this analysis scope
  useEffect(() => {
    fetchSavedFilters("analysis_results")
      .then(setSavedFilters)
      .catch(() => {}); // non-critical
  }, [analysisId]);

  // Load other completed analyses for cross-compare
  useEffect(() => {
    fetchAnalyses({ status: "completed", per_page: 100 })
      .then((res) => setAllAnalyses(res.analyses))
      .catch(() => {});
  }, []);

  const loadResults = useCallback(async () => {
    setLoading(true);
    try {
      const activeFilters = filters.filter((f) => f.col && f.val);
      const data = await fetchAnalysisResults(analysisId, {
        offset: page * PAGE_SIZE,
        limit: PAGE_SIZE,
        sort_col: sortCol || undefined,
        sort_order: sortOrder,
        filters: activeFilters.length > 0 ? activeFilters : undefined,
        search: search || undefined,
      });
      setResults(data);
      if (columns.length === 0 && data.columns.length > 0) {
        setColumns(data.columns);
        // Pre-select prediction columns for distributions
        const predCols = data.columns.filter(
          (c) => c.endsWith("_pred") || c.endsWith("_conf")
        );
        if (distCols.length === 0 && predCols.length > 0) {
          setDistCols(predCols.filter((c) => c.endsWith("_pred")).slice(0, 3));
        }
        if (!segGroupBy && predCols.length > 0) {
          setSegGroupBy(predCols.find((c) => c.endsWith("_pred")) ?? "");
        }
        if (!segMetricCol) {
          const confCol = data.columns.find((c) => c.endsWith("_conf"));
          if (confCol) setSegMetricCol(confCol);
        }
        if (compareCols.length === 0 && predCols.length > 0) {
          setCompareCols(predCols.filter((c) => c.endsWith("_pred")).slice(0, 2));
        }
      }
      setError(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [analysisId, page, sortCol, sortOrder, filters, search, columns.length]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    loadResults();
  }, [loadResults]);

  // Keep columns list in sync once we have results
  useEffect(() => {
    if (results && results.columns.length > 0) {
      setColumns(results.columns);
    }
  }, [results]);

  // ---------------------------------------------------------------------------
  // Derived
  // ---------------------------------------------------------------------------

  const summary = analysis?.result_summary as AnalysisSummary | null;
  const predCols = summary?.models_applied
    ?.filter((m) => m.pred_col)
    .map((m) => m.pred_col!) ?? [];
  const confCols = summary?.models_applied
    ?.filter((m) => m.conf_col)
    .map((m) => m.conf_col!) ?? [];

  const colStyle = (col: string): React.CSSProperties => ({
    background: predCols.includes(col)
      ? "rgba(var(--gold-rgb, 201,155,64),0.06)"
      : "transparent",
  });

  const totalPages = results ? Math.ceil(results.total_rows / PAGE_SIZE) : 0;

  // ---------------------------------------------------------------------------
  // Table tab handlers
  // ---------------------------------------------------------------------------

  function handleSort(col: string) {
    if (sortCol === col) {
      setSortOrder((o) => (o === "asc" ? "desc" : "asc"));
    } else {
      setSortCol(col);
      setSortOrder("asc");
    }
    setPage(0);
  }

  function handleSearchApply() {
    setSearch(searchInput);
    setPage(0);
  }

  function handleClearAll() {
    setFilters([]);
    setSearch("");
    setSearchInput("");
    setSortCol("");
    setSortOrder("asc");
    setPage(0);
  }

  function addFilterRule() {
    setFilters((prev) => [
      ...prev,
      { col: columns[0] ?? "", op: "eq", val: "" },
    ]);
  }

  function updateFilterRule(idx: number, patch: Partial<FilterRule>) {
    setFilters((prev) =>
      prev.map((f, i) => (i === idx ? { ...f, ...patch } : f))
    );
  }

  function removeFilterRule(idx: number) {
    setFilters((prev) => prev.filter((_, i) => i !== idx));
    setPage(0);
  }

  function handleApplyFilters() {
    setPage(0);
    loadResults();
  }

  // Anomaly detection
  async function loadAnomalies() {
    setAnomalyLoading(true);
    try {
      const res = await fetchAnalysisAnomalies(analysisId, anomalyThreshold);
      setAnomalies(res.anomalies);
      setAnomalyTotal(res.total);
    } catch {
      setAnomalies([]);
      setAnomalyTotal(0);
    } finally {
      setAnomalyLoading(false);
    }
  }

  useEffect(() => {
    if (showAnomalies) loadAnomalies();
  }, [showAnomalies, anomalyThreshold]); // eslint-disable-line react-hooks/exhaustive-deps

  // Saved filters
  async function handleSaveFilter() {
    if (!saveNameInput.trim()) return;
    setSavingFilter(true);
    try {
      const activeFilters = filters.filter((f) => f.col && f.val);
      const saved = await createSavedFilter("analysis_results", saveNameInput.trim(), {
        filters: activeFilters,
        search,
        sort_col: sortCol || undefined,
        sort_order: sortOrder,
      });
      setSavedFilters((prev) => [saved, ...prev]);
      setSaveNameInput("");
    } catch {
      // ignore
    } finally {
      setSavingFilter(false);
    }
  }

  async function handleDeleteSavedFilter(filterId: string) {
    try {
      await deleteSavedFilter(filterId);
      setSavedFilters((prev) => prev.filter((f) => f.id !== filterId));
    } catch {
      // ignore
    }
  }

  function handleApplySavedFilter(sf: SavedFilter) {
    const cfg = sf.filter_config;
    setFilters(cfg.filters ?? []);
    setSearch(cfg.search ?? "");
    setSearchInput(cfg.search ?? "");
    setSortCol(cfg.sort_col ?? "");
    setSortOrder((cfg.sort_order as "asc" | "desc") ?? "asc");
    setPage(0);
    setShowSavedPanel(false);
  }

  // ---------------------------------------------------------------------------
  // Analytics tab handlers
  // ---------------------------------------------------------------------------

  async function loadDistributions() {
    if (distCols.length === 0) return;
    setDistLoading(true);
    setDistError(null);
    try {
      const res = await fetchAnalysisDistributions(analysisId, distCols);
      setDistData(res.distributions);
    } catch (e: unknown) {
      setDistError(e instanceof Error ? e.message : String(e));
    } finally {
      setDistLoading(false);
    }
  }

  async function loadSegmentStats() {
    if (!segGroupBy || !segMetricCol) return;
    setSegLoading(true);
    setSegError(null);
    try {
      const res = await fetchAnalysisSegmentStats(analysisId, segGroupBy, segMetricCol);
      setSegGroups(res.groups);
    } catch (e: unknown) {
      setSegError(e instanceof Error ? e.message : String(e));
    } finally {
      setSegLoading(false);
    }
  }

  async function loadCrossCompare() {
    if (compareIds.length < 1 || compareCols.length === 0) return;
    setCompareLoading(true);
    setCompareError(null);
    try {
      const allIds = [analysisId, ...compareIds.filter((id) => id !== analysisId)];
      const res = await fetchCrossCompare(allIds, compareCols);
      setCompareData(res);
    } catch (e: unknown) {
      setCompareError(e instanceof Error ? e.message : String(e));
    } finally {
      setCompareLoading(false);
    }
  }

  function toggleDistCol(col: string) {
    setDistCols((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  }

  function toggleCompareCol(col: string) {
    setCompareCols((prev) =>
      prev.includes(col) ? prev.filter((c) => c !== col) : [...prev, col]
    );
  }

  function toggleCompareId(id: string) {
    setCompareIds((prev) =>
      prev.includes(id) ? prev.filter((i) => i !== id) : [...prev, id]
    );
  }

  // Filtered export URL
  const activeFilters = filters.filter((f) => f.col && f.val);
  const hasActiveFilter = activeFilters.length > 0 || !!search;

  // ---------------------------------------------------------------------------
  // Tab nav style
  // ---------------------------------------------------------------------------

  const tabStyle = (active: boolean): React.CSSProperties => ({
    padding: "8px 18px",
    border: "none",
    borderBottom: active ? "2px solid var(--gold)" : "2px solid transparent",
    background: "transparent",
    color: active ? "var(--gold)" : "var(--text-secondary)",
    fontWeight: active ? 600 : 400,
    fontSize: "13px",
    cursor: "pointer",
    transition: "color 0.15s",
  });

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div style={{ padding: "28px 40px", maxWidth: "1400px", margin: "0 auto" }}>
      {/* Breadcrumb */}
      <div style={{ fontSize: "12px", color: "var(--text-tertiary)", marginBottom: "16px" }}>
        <Link href="/analyses" style={{ color: "var(--text-tertiary)", textDecoration: "none" }}>
          Analyses
        </Link>
        {" / "}
        <Link
          href={`/analyses/${analysisId}`}
          style={{ color: "var(--text-tertiary)", textDecoration: "none" }}
        >
          {analysis?.name || analysisId.slice(0, 20)}
        </Link>
        {" / "}
        <span style={{ color: "var(--text-secondary)" }}>
          {activeTab === "table" ? "Results" : "Analytics"}
        </span>
      </div>

      {/* Header */}
      <div className="flex items-center justify-between" style={{ marginBottom: "16px" }}>
        <div>
          <h1
            style={{
              fontFamily: "var(--font-syne)",
              fontWeight: 700,
              fontSize: "20px",
              color: "var(--text-primary)",
              margin: 0,
            }}
          >
            {analysis?.name || analysisId.slice(0, 20)}
          </h1>
          <p style={{ color: "var(--text-tertiary)", fontSize: "12px", marginTop: "4px" }}>
            {results?.total_rows != null
              ? `${results.total_rows.toLocaleString()} rows`
              : "Loading…"}
            {predCols.length > 0 && (
              <span> · Prediction cols: {predCols.join(", ")}</span>
            )}
            {hasActiveFilter && (
              <span style={{ color: "var(--gold)", marginLeft: "8px" }}>● filtered</span>
            )}
          </p>
        </div>

        {/* Export buttons (visible on table tab) */}
        {activeTab === "table" && (
          <div className="flex gap-2" style={{ flexWrap: "wrap" }}>
            {hasActiveFilter ? (
              <>
                <a
                  href={getFilteredExportUrl(
                    analysisId,
                    "csv",
                    activeFilters.length > 0 ? activeFilters : undefined,
                    search || undefined,
                    sortCol || undefined,
                    sortOrder
                  )}
                  download
                  style={{ ...btnOutline, color: "var(--gold)", borderColor: "var(--gold-muted)" }}
                >
                  Export Filtered CSV
                </a>
                <a
                  href={getFilteredExportUrl(
                    analysisId,
                    "json",
                    activeFilters.length > 0 ? activeFilters : undefined,
                    search || undefined,
                    sortCol || undefined,
                    sortOrder
                  )}
                  download
                  style={btnOutline}
                >
                  Export Filtered JSON
                </a>
              </>
            ) : (
              <>
                <a href={getAnalysisExportUrl(analysisId, "csv")} download style={btnOutline}>
                  Export CSV
                </a>
                <a href={getAnalysisExportUrl(analysisId, "json")} download style={btnOutline}>
                  Export JSON
                </a>
              </>
            )}
            <button
              onClick={() => setShowAnomalies((v) => !v)}
              style={{
                ...btnOutline,
                borderColor: showAnomalies ? "var(--gold)" : undefined,
                color: showAnomalies ? "var(--gold)" : "var(--text-secondary)",
              }}
            >
              Anomalies {anomalyTotal > 0 ? `(${anomalyTotal})` : ""}
            </button>
            <button
              onClick={() => setShowSavedPanel((v) => !v)}
              style={{
                ...btnOutline,
                borderColor: showSavedPanel ? "var(--gold)" : undefined,
                color: showSavedPanel ? "var(--gold)" : "var(--text-secondary)",
              }}
            >
              Saved Filters {savedFilters.length > 0 ? `(${savedFilters.length})` : ""}
            </button>
          </div>
        )}
      </div>

      {/* Disclaimer */}
      <div
        style={{
          padding: "8px 14px",
          borderRadius: "6px",
          background: "var(--gold-faint)",
          border: "1px solid var(--gold-muted)",
          fontSize: "11px",
          color: "var(--text-secondary)",
          marginBottom: "16px",
        }}
      >
        Batch predictions for aggregate analysis. Not for individual-level decisions.
      </div>

      {/* Tab bar */}
      <div
        style={{
          display: "flex",
          borderBottom: "1px solid var(--border-dim)",
          marginBottom: "20px",
        }}
      >
        <button style={tabStyle(activeTab === "table")} onClick={() => setActiveTab("table")}>
          Results Table
        </button>
        <button
          style={tabStyle(activeTab === "analytics")}
          onClick={() => setActiveTab("analytics")}
        >
          Analytics
        </button>
      </div>

      {/* ================================================================ */}
      {/* TABLE TAB                                                        */}
      {/* ================================================================ */}
      {activeTab === "table" && (
        <>
          {/* Saved Filters Panel */}
          {showSavedPanel && (
            <div
              style={{
                padding: "16px",
                background: "var(--bg-surface)",
                border: "1px solid var(--border-dim)",
                borderRadius: "8px",
                marginBottom: "16px",
              }}
            >
              <div className="flex items-center justify-between" style={{ marginBottom: "12px" }}>
                <span style={{ fontSize: "13px", fontWeight: 600, color: "var(--text-primary)" }}>
                  Saved Filters
                </span>
                <button
                  onClick={() => setShowSavedPanel(false)}
                  style={{ ...btnOutline, padding: "3px 8px", fontSize: "11px" }}
                >
                  ✕
                </button>
              </div>

              <div className="flex gap-2 items-center" style={{ marginBottom: "12px" }}>
                <input
                  type="text"
                  placeholder="Name for this filter config…"
                  value={saveNameInput}
                  onChange={(e) => setSaveNameInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleSaveFilter()}
                  style={{ ...inputStyle, flex: 1 }}
                />
                <button
                  onClick={handleSaveFilter}
                  disabled={savingFilter || !saveNameInput.trim()}
                  style={{
                    ...btnGold,
                    opacity: savingFilter || !saveNameInput.trim() ? 0.5 : 1,
                  }}
                >
                  Save current
                </button>
              </div>

              {savedFilters.length === 0 ? (
                <p style={{ fontSize: "12px", color: "var(--text-tertiary)" }}>
                  No saved filters yet.
                </p>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                  {savedFilters.map((sf) => (
                    <div
                      key={sf.id}
                      className="flex items-center justify-between"
                      style={{
                        padding: "8px 12px",
                        border: "1px solid var(--border-dim)",
                        borderRadius: "6px",
                        background: "var(--bg-base)",
                      }}
                    >
                      <div>
                        <span style={{ fontSize: "13px", color: "var(--text-primary)" }}>
                          {sf.name}
                        </span>
                        <span
                          style={{
                            fontSize: "11px",
                            color: "var(--text-tertiary)",
                            marginLeft: "8px",
                          }}
                        >
                          {sf.filter_config.filters?.length ?? 0} rules
                          {sf.filter_config.search
                            ? `, search: "${sf.filter_config.search}"`
                            : ""}
                        </span>
                      </div>
                      <div className="flex gap-2">
                        <button
                          onClick={() => handleApplySavedFilter(sf)}
                          style={{ ...btnOutline, padding: "4px 10px", fontSize: "11px" }}
                        >
                          Apply
                        </button>
                        <button
                          onClick={() => handleDeleteSavedFilter(sf.id)}
                          style={{
                            ...btnOutline,
                            padding: "4px 10px",
                            fontSize: "11px",
                            color: "var(--error, #ef4444)",
                          }}
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Anomalies Panel */}
          {showAnomalies && (
            <div
              style={{
                padding: "16px",
                background: "var(--bg-surface)",
                border: "1px solid var(--border-dim)",
                borderRadius: "8px",
                marginBottom: "16px",
              }}
            >
              <div className="flex items-center justify-between" style={{ marginBottom: "12px" }}>
                <span style={{ fontSize: "13px", fontWeight: 600, color: "var(--text-primary)" }}>
                  Anomaly Detection
                </span>
                <button
                  onClick={() => setShowAnomalies(false)}
                  style={{ ...btnOutline, padding: "3px 8px", fontSize: "11px" }}
                >
                  ✕
                </button>
              </div>

              <div className="flex gap-3 items-center" style={{ marginBottom: "12px" }}>
                <label style={{ fontSize: "12px", color: "var(--text-tertiary)" }}>
                  Confidence threshold:
                </label>
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.05"
                  value={anomalyThreshold}
                  onChange={(e) =>
                    setAnomalyThreshold(parseFloat(e.target.value) || 0.6)
                  }
                  style={{ ...inputStyle, width: "80px" }}
                />
                <button onClick={loadAnomalies} style={btnGold} disabled={anomalyLoading}>
                  {anomalyLoading ? "Scanning…" : "Scan"}
                </button>
              </div>

              {anomalyLoading ? (
                <p style={{ fontSize: "12px", color: "var(--text-tertiary)" }}>
                  Scanning for anomalies…
                </p>
              ) : anomalies !== null ? (
                anomalies.length === 0 ? (
                  <p style={{ fontSize: "12px", color: "var(--text-tertiary)" }}>
                    No anomalies found (threshold: {anomalyThreshold}).
                  </p>
                ) : (
                  <div>
                    <p
                      style={{
                        fontSize: "12px",
                        color: "var(--text-secondary)",
                        marginBottom: "8px",
                      }}
                    >
                      {anomalyTotal} anomalous row{anomalyTotal !== 1 ? "s" : ""} found
                      {anomalyTotal > 500 ? " (showing first 500)" : ""}.
                      Reasons: low confidence below {anomalyThreshold} or empty text column.
                    </p>
                    <div
                      style={{
                        maxHeight: "260px",
                        overflowY: "auto",
                        borderRadius: "6px",
                        border: "1px solid var(--border-dim)",
                      }}
                    >
                      <table
                        style={{ width: "100%", borderCollapse: "collapse", fontSize: "11px" }}
                      >
                        <thead>
                          <tr style={{ background: "var(--bg-base)" }}>
                            <th
                              style={{
                                padding: "7px 10px",
                                textAlign: "left",
                                color: "var(--text-tertiary)",
                                borderBottom: "1px solid var(--border-dim)",
                                whiteSpace: "nowrap",
                              }}
                            >
                              Row #
                            </th>
                            <th
                              style={{
                                padding: "7px 10px",
                                textAlign: "left",
                                color: "var(--text-tertiary)",
                                borderBottom: "1px solid var(--border-dim)",
                              }}
                            >
                              Reasons
                            </th>
                            {confCols.slice(0, 3).map((cc) => (
                              <th
                                key={cc}
                                style={{
                                  padding: "7px 10px",
                                  textAlign: "left",
                                  color: "var(--gold)",
                                  borderBottom: "1px solid var(--border-dim)",
                                  whiteSpace: "nowrap",
                                }}
                              >
                                {cc}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {anomalies.map((a) => (
                            <tr
                              key={a.row_index}
                              style={{ borderBottom: "1px solid var(--border-dim)" }}
                            >
                              <td
                                style={{
                                  padding: "6px 10px",
                                  color: "var(--text-tertiary)",
                                  fontFamily: "var(--font-jetbrains)",
                                }}
                              >
                                {a.row_index}
                              </td>
                              <td
                                style={{ padding: "6px 10px", color: "var(--text-secondary)" }}
                              >
                                {a.reasons.join("; ")}
                              </td>
                              {confCols.slice(0, 3).map((cc) => (
                                <td
                                  key={cc}
                                  style={{
                                    padding: "6px 10px",
                                    color: "var(--error, #ef4444)",
                                    fontFamily: "var(--font-jetbrains)",
                                  }}
                                >
                                  {a.data[cc] ?? "—"}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )
              ) : null}
            </div>
          )}

          {/* Search bar */}
          <div className="flex gap-2 items-center" style={{ marginBottom: "12px" }}>
            <input
              type="text"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearchApply()}
              placeholder="Search across all text columns…"
              style={{ ...inputStyle, flex: 1, maxWidth: "400px" }}
            />
            <button onClick={handleSearchApply} style={btnGold}>
              Search
            </button>
            {search && (
              <button
                onClick={() => {
                  setSearch("");
                  setSearchInput("");
                  setPage(0);
                }}
                style={btnOutline}
              >
                Clear search
              </button>
            )}
          </div>

          {/* Multi-filter builder */}
          <div
            style={{
              padding: "12px 16px",
              background: "var(--bg-surface)",
              border: "1px solid var(--border-dim)",
              borderRadius: "7px",
              marginBottom: "16px",
            }}
          >
            <div className="flex items-center justify-between" style={{ marginBottom: "10px" }}>
              <span style={{ fontSize: "12px", fontWeight: 600, color: "var(--text-secondary)" }}>
                Filters{" "}
                {filters.length > 0
                  ? `(${filters.length} rule${filters.length > 1 ? "s" : ""})`
                  : ""}
              </span>
              <div className="flex gap-2">
                <button
                  onClick={addFilterRule}
                  disabled={columns.length === 0}
                  style={{ ...btnOutline, padding: "4px 10px", fontSize: "11px" }}
                >
                  + Add rule
                </button>
                {filters.length > 0 && (
                  <>
                    <button
                      onClick={handleApplyFilters}
                      style={{ ...btnGold, padding: "4px 12px", fontSize: "11px" }}
                    >
                      Apply
                    </button>
                    <button
                      onClick={handleClearAll}
                      style={{ ...btnOutline, padding: "4px 10px", fontSize: "11px" }}
                    >
                      Clear all
                    </button>
                  </>
                )}
              </div>
            </div>

            {filters.length === 0 ? (
              <p style={{ fontSize: "11px", color: "var(--text-tertiary)", margin: 0 }}>
                No filters active. Click &quot;+ Add rule&quot; to filter by column value.
              </p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
                {filters.map((f, idx) => (
                  <div key={idx} className="flex gap-2 items-center">
                    <select
                      value={f.col}
                      onChange={(e) => updateFilterRule(idx, { col: e.target.value })}
                      style={{ ...inputStyle, minWidth: "150px" }}
                    >
                      {columns.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                    <select
                      value={f.op}
                      onChange={(e) =>
                        updateFilterRule(idx, { op: e.target.value as FilterRule["op"] })
                      }
                      style={{ ...inputStyle, width: "100px" }}
                    >
                      {FILTER_OPS.map((o) => (
                        <option key={o.value} value={o.value}>
                          {o.label}
                        </option>
                      ))}
                    </select>
                    <input
                      type="text"
                      value={f.val}
                      onChange={(e) => updateFilterRule(idx, { val: e.target.value })}
                      onKeyDown={(e) => e.key === "Enter" && handleApplyFilters()}
                      placeholder="value…"
                      style={{ ...inputStyle, width: "160px" }}
                    />
                    <button
                      onClick={() => removeFilterRule(idx)}
                      style={{
                        ...btnOutline,
                        padding: "5px 8px",
                        fontSize: "11px",
                        color: "var(--text-tertiary)",
                      }}
                    >
                      ✕
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Error */}
          {error && (
            <div
              style={{ color: "var(--error, #ef4444)", fontSize: "13px", marginBottom: "12px" }}
            >
              {error}
            </div>
          )}

          {/* Results Table */}
          {loading && !results ? (
            <div
              style={{
                color: "var(--text-tertiary)",
                fontSize: "13px",
                padding: "40px 0",
                textAlign: "center",
              }}
            >
              Loading results…
            </div>
          ) : results && results.columns.length > 0 ? (
            <div
              style={{
                overflowX: "auto",
                borderRadius: "8px",
                border: "1px solid var(--border-dim)",
              }}
            >
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "12px" }}>
                <thead>
                  <tr style={{ background: "var(--bg-surface)" }}>
                    {results.columns.map((col) => (
                      <th
                        key={col}
                        onClick={() => handleSort(col)}
                        style={{
                          padding: "9px 12px",
                          textAlign: "left",
                          fontSize: "11px",
                          fontFamily: "var(--font-jetbrains)",
                          color: predCols.includes(col) ? "var(--gold)" : "var(--text-tertiary)",
                          fontWeight: predCols.includes(col) ? 600 : 400,
                          borderBottom: "1px solid var(--border-dim)",
                          cursor: "pointer",
                          whiteSpace: "nowrap",
                          userSelect: "none",
                          ...colStyle(col),
                        }}
                      >
                        {col}
                        {sortCol === col && (
                          <span style={{ marginLeft: "4px" }}>
                            {sortOrder === "asc" ? "↑" : "↓"}
                          </span>
                        )}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {results.rows.map((row, ri) => (
                    <tr
                      key={ri}
                      style={{
                        borderBottom: "1px solid var(--border-dim)",
                        opacity: loading ? 0.5 : 1,
                      }}
                    >
                      {row.map((cell, ci) => {
                        const col = results.columns[ci];
                        const isPred = predCols.includes(col);
                        const isConf = confCols.includes(col);
                        const isLowConf =
                          isConf &&
                          parseFloat(cell) < anomalyThreshold &&
                          !isNaN(parseFloat(cell));
                        return (
                          <td
                            key={ci}
                            style={{
                              padding: "7px 12px",
                              color: isLowConf
                                ? "var(--error, #ef4444)"
                                : isPred
                                ? "var(--text-primary)"
                                : "var(--text-secondary)",
                              fontFamily:
                                isPred || isConf ? undefined : "var(--font-jetbrains)",
                              fontSize: isPred ? "12px" : "11px",
                              fontWeight: isPred ? 500 : 400,
                              maxWidth: "200px",
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                              whiteSpace: "nowrap",
                              ...colStyle(col),
                            }}
                            title={cell}
                          >
                            {cell}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div
              style={{
                color: "var(--text-tertiary)",
                fontSize: "13px",
                padding: "40px 0",
                textAlign: "center",
              }}
            >
              No results found.
            </div>
          )}

          {/* Pagination */}
          {totalPages > 1 && (
            <div className="flex items-center justify-between" style={{ marginTop: "16px" }}>
              <span style={{ fontSize: "12px", color: "var(--text-tertiary)" }}>
                Page {page + 1} of {totalPages} ·{" "}
                {results?.total_rows.toLocaleString()} rows ·{" "}
                showing {page * PAGE_SIZE + 1}–
                {Math.min((page + 1) * PAGE_SIZE, results?.total_rows ?? 0)}
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => setPage((p) => Math.max(0, p - 1))}
                  disabled={page === 0}
                  style={{
                    ...btnOutline,
                    color: page === 0 ? "var(--text-tertiary)" : "var(--text-secondary)",
                    cursor: page === 0 ? "not-allowed" : "pointer",
                  }}
                >
                  Prev
                </button>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                  disabled={page === totalPages - 1}
                  style={{
                    ...btnOutline,
                    color:
                      page === totalPages - 1
                        ? "var(--text-tertiary)"
                        : "var(--text-secondary)",
                    cursor: page === totalPages - 1 ? "not-allowed" : "pointer",
                  }}
                >
                  Next
                </button>
              </div>
            </div>
          )}
        </>
      )}

      {/* ================================================================ */}
      {/* ANALYTICS TAB                                                    */}
      {/* ================================================================ */}
      {activeTab === "analytics" && (
        <div style={{ display: "flex", flexDirection: "column", gap: "28px" }}>
          {/* ---- Section 1: Distributions ---- */}
          <section>
            <h2
              style={{
                fontFamily: "var(--font-syne)",
                fontSize: "15px",
                fontWeight: 700,
                color: "var(--text-primary)",
                marginBottom: "14px",
              }}
            >
              Value Distributions
            </h2>

            {/* Column selector */}
            <div
              style={{
                padding: "12px 16px",
                background: "var(--bg-surface)",
                border: "1px solid var(--border-dim)",
                borderRadius: "7px",
                marginBottom: "14px",
              }}
            >
              <div style={{ fontSize: "12px", color: "var(--text-secondary)", marginBottom: "8px" }}>
                Select columns to visualise:
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "6px", marginBottom: "10px" }}>
                {columns.map((col) => (
                  <button
                    key={col}
                    onClick={() => toggleDistCol(col)}
                    style={{
                      padding: "3px 10px",
                      borderRadius: "4px",
                      border: "1px solid",
                      borderColor: distCols.includes(col) ? "var(--gold)" : "var(--border-dim)",
                      background: distCols.includes(col) ? "rgba(201,155,64,0.12)" : "transparent",
                      color: distCols.includes(col) ? "var(--gold)" : "var(--text-tertiary)",
                      fontSize: "11px",
                      fontFamily: "var(--font-jetbrains)",
                      cursor: "pointer",
                    }}
                  >
                    {col}
                  </button>
                ))}
              </div>
              <button
                onClick={loadDistributions}
                disabled={distCols.length === 0 || distLoading}
                style={{
                  ...btnGold,
                  opacity: distCols.length === 0 || distLoading ? 0.5 : 1,
                }}
              >
                {distLoading ? "Loading…" : "Load distributions"}
              </button>
            </div>

            {distError && (
              <div style={{ color: "var(--error, #ef4444)", fontSize: "12px", marginBottom: "12px" }}>
                {distError}
              </div>
            )}

            {distData && (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))",
                  gap: "20px",
                }}
              >
                {Object.entries(distData).map(([col, data]) => (
                  <div
                    key={col}
                    style={{
                      padding: "16px",
                      background: "var(--bg-surface)",
                      border: "1px solid var(--border-dim)",
                      borderRadius: "8px",
                    }}
                  >
                    <DistributionChart column={col} data={data} />
                  </div>
                ))}
              </div>
            )}
          </section>

          {/* ---- Section 2: Segment Stats ---- */}
          <section>
            <h2
              style={{
                fontFamily: "var(--font-syne)",
                fontSize: "15px",
                fontWeight: 700,
                color: "var(--text-primary)",
                marginBottom: "14px",
              }}
            >
              Segment Statistics
            </h2>

            <div
              style={{
                padding: "12px 16px",
                background: "var(--bg-surface)",
                border: "1px solid var(--border-dim)",
                borderRadius: "7px",
                marginBottom: "14px",
              }}
            >
              <div
                style={{ display: "flex", gap: "12px", alignItems: "flex-end", flexWrap: "wrap" }}
              >
                <div>
                  <div
                    style={{
                      fontSize: "11px",
                      color: "var(--text-tertiary)",
                      marginBottom: "4px",
                    }}
                  >
                    Group by (categorical)
                  </div>
                  <select
                    value={segGroupBy}
                    onChange={(e) => setSegGroupBy(e.target.value)}
                    style={{ ...inputStyle, minWidth: "160px" }}
                  >
                    <option value="">— select —</option>
                    {columns.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <div
                    style={{
                      fontSize: "11px",
                      color: "var(--text-tertiary)",
                      marginBottom: "4px",
                    }}
                  >
                    Metric column (numeric)
                  </div>
                  <select
                    value={segMetricCol}
                    onChange={(e) => setSegMetricCol(e.target.value)}
                    style={{ ...inputStyle, minWidth: "160px" }}
                  >
                    <option value="">— select —</option>
                    {columns.map((c) => (
                      <option key={c} value={c}>
                        {c}
                      </option>
                    ))}
                  </select>
                </div>
                <button
                  onClick={loadSegmentStats}
                  disabled={!segGroupBy || !segMetricCol || segLoading}
                  style={{
                    ...btnGold,
                    opacity: !segGroupBy || !segMetricCol || segLoading ? 0.5 : 1,
                  }}
                >
                  {segLoading ? "Loading…" : "Compute stats"}
                </button>
              </div>
            </div>

            {segError && (
              <div style={{ color: "var(--error, #ef4444)", fontSize: "12px", marginBottom: "12px" }}>
                {segError}
              </div>
            )}

            {segGroups !== null && (
              <SegmentTable
                groups={segGroups}
                groupBy={segGroupBy}
                metricCol={segMetricCol}
              />
            )}
          </section>

          {/* ---- Section 3: Cross-compare ---- */}
          <section>
            <h2
              style={{
                fontFamily: "var(--font-syne)",
                fontSize: "15px",
                fontWeight: 700,
                color: "var(--text-primary)",
                marginBottom: "14px",
              }}
            >
              Cross-Analysis Comparison
            </h2>

            <div
              style={{
                padding: "12px 16px",
                background: "var(--bg-surface)",
                border: "1px solid var(--border-dim)",
                borderRadius: "7px",
                marginBottom: "14px",
              }}
            >
              {/* Analysis selector */}
              <div style={{ marginBottom: "12px" }}>
                <div
                  style={{
                    fontSize: "12px",
                    color: "var(--text-secondary)",
                    marginBottom: "6px",
                  }}
                >
                  Compare with other completed analyses:
                </div>
                {allAnalyses.filter((a) => a.id !== analysisId).length === 0 ? (
                  <p style={{ fontSize: "11px", color: "var(--text-tertiary)" }}>
                    No other completed analyses available.
                  </p>
                ) : (
                  <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                    {allAnalyses
                      .filter((a) => a.id !== analysisId)
                      .map((a) => (
                        <button
                          key={a.id}
                          onClick={() => toggleCompareId(a.id)}
                          style={{
                            padding: "3px 10px",
                            borderRadius: "4px",
                            border: "1px solid",
                            borderColor: compareIds.includes(a.id)
                              ? "var(--gold)"
                              : "var(--border-dim)",
                            background: compareIds.includes(a.id)
                              ? "rgba(201,155,64,0.12)"
                              : "transparent",
                            color: compareIds.includes(a.id)
                              ? "var(--gold)"
                              : "var(--text-tertiary)",
                            fontSize: "11px",
                            cursor: "pointer",
                          }}
                        >
                          {a.name || a.id.slice(0, 16)}
                        </button>
                      ))}
                  </div>
                )}
              </div>

              {/* Column selector */}
              <div style={{ marginBottom: "12px" }}>
                <div
                  style={{
                    fontSize: "12px",
                    color: "var(--text-secondary)",
                    marginBottom: "6px",
                  }}
                >
                  Columns to compare:
                </div>
                <div style={{ display: "flex", flexWrap: "wrap", gap: "6px" }}>
                  {columns.map((col) => (
                    <button
                      key={col}
                      onClick={() => toggleCompareCol(col)}
                      style={{
                        padding: "3px 10px",
                        borderRadius: "4px",
                        border: "1px solid",
                        borderColor: compareCols.includes(col)
                          ? "var(--gold)"
                          : "var(--border-dim)",
                        background: compareCols.includes(col)
                          ? "rgba(201,155,64,0.12)"
                          : "transparent",
                        color: compareCols.includes(col)
                          ? "var(--gold)"
                          : "var(--text-tertiary)",
                        fontSize: "11px",
                        fontFamily: "var(--font-jetbrains)",
                        cursor: "pointer",
                      }}
                    >
                      {col}
                    </button>
                  ))}
                </div>
              </div>

              <button
                onClick={loadCrossCompare}
                disabled={compareIds.length < 1 || compareCols.length === 0 || compareLoading}
                style={{
                  ...btnGold,
                  opacity:
                    compareIds.length < 1 || compareCols.length === 0 || compareLoading
                      ? 0.5
                      : 1,
                }}
              >
                {compareLoading ? "Comparing…" : "Compare"}
              </button>
            </div>

            {compareError && (
              <div style={{ color: "var(--error, #ef4444)", fontSize: "12px", marginBottom: "12px" }}>
                {compareError}
              </div>
            )}

            {compareData && (
              <div style={{ display: "flex", flexDirection: "column", gap: "20px" }}>
                {/* Disagreement rates summary */}
                {Object.keys(compareData.disagreement_rates).length > 0 && (
                  <div
                    style={{
                      padding: "14px 16px",
                      background: "var(--bg-surface)",
                      border: "1px solid var(--border-dim)",
                      borderRadius: "8px",
                    }}
                  >
                    <div
                      style={{
                        fontSize: "12px",
                        fontWeight: 600,
                        color: "var(--text-secondary)",
                        marginBottom: "10px",
                      }}
                    >
                      Disagreement Rates (fraction of rows where analyses differ)
                    </div>
                    <div style={{ display: "flex", flexWrap: "wrap", gap: "12px" }}>
                      {Object.entries(compareData.disagreement_rates).map(([col, rate]) => (
                        <div
                          key={col}
                          style={{
                            padding: "8px 14px",
                            borderRadius: "6px",
                            border: "1px solid var(--border-dim)",
                            background: "var(--bg-base)",
                          }}
                        >
                          <div
                            style={{
                              fontSize: "11px",
                              fontFamily: "var(--font-jetbrains)",
                              color: "var(--text-secondary)",
                            }}
                          >
                            {col}
                          </div>
                          <div
                            style={{
                              fontSize: "18px",
                              fontWeight: 700,
                              color:
                                rate > 0.2
                                  ? "var(--error, #ef4444)"
                                  : rate > 0.1
                                  ? "var(--gold)"
                                  : "var(--text-primary)",
                              fontFamily: "var(--font-jetbrains)",
                            }}
                          >
                            {(rate * 100).toFixed(1)}%
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Per-analysis distribution charts per column */}
                {compareData.columns.map((col) => (
                  <div key={col}>
                    <div
                      style={{
                        fontSize: "13px",
                        fontWeight: 600,
                        color: "var(--text-secondary)",
                        fontFamily: "var(--font-jetbrains)",
                        marginBottom: "10px",
                      }}
                    >
                      {col}
                      {compareData.disagreement_rates[col] != null && (
                        <span
                          style={{
                            marginLeft: "10px",
                            fontSize: "11px",
                            fontWeight: 400,
                            color:
                              compareData.disagreement_rates[col] > 0.2
                                ? "var(--error, #ef4444)"
                                : "var(--text-tertiary)",
                          }}
                        >
                          {(compareData.disagreement_rates[col] * 100).toFixed(1)}% disagreement
                        </span>
                      )}
                    </div>
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: `repeat(${compareData.analysis_ids.length}, 1fr)`,
                        gap: "14px",
                      }}
                    >
                      {compareData.analysis_ids.map((aid) => {
                        const dist = compareData.per_analysis[aid]?.[col] ?? {};
                        const label =
                          aid === analysisId
                            ? (analysis?.name || `This (${aid.slice(9, 17)})`)
                            : (allAnalyses.find((a) => a.id === aid)?.name || aid.slice(9, 17));
                        return (
                          <div
                            key={aid}
                            style={{
                              padding: "14px",
                              background: "var(--bg-surface)",
                              border: "1px solid var(--border-dim)",
                              borderRadius: "8px",
                            }}
                          >
                            <div
                              style={{
                                fontSize: "11px",
                                color: "var(--text-tertiary)",
                                marginBottom: "10px",
                                fontWeight: 600,
                              }}
                            >
                              {label}
                            </div>
                            <DistributionChart column={col} data={dist} />
                          </div>
                        );
                      })}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
