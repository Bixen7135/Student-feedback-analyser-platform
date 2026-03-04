"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";

import {
  type AnalysisClusterResponse,
  type AnalysisCorrelationsResponse,
  type AnalysisDescriptiveAnalytics,
  type AnalysisDiagnosticsResponse,
  type AnalysisRecord,
  type AnalysisResults,
  type AnalysisSummary,
  type SavedFilter,
  createSavedFilter,
  deleteSavedFilter,
  fetchAnalysisCluster,
  fetchAnalysisCorrelations,
  fetchAnalysisDescriptiveAnalytics,
  fetchAnalysisDetail,
  fetchAnalysisDiagnostics,
  fetchAnalysisResults,
  fetchSavedFilters,
  getAnalysisExportUrl,
  getFilteredExportUrl,
} from "@/app/lib/api";
import type { FilterRule } from "@/app/lib/filters";
import {
  buildCalibrationBins,
  buildCategoricalSections,
  buildClusterPoints,
  buildConfusionCells,
  buildCorrelationCells,
  buildNumericSections,
  buildTextLengthSections,
} from "@/app/lib/analytics";
import { FilterProvider, useFilterContext } from "@/app/components/filters/FilterProvider";
import { BarListChart } from "@/app/components/charts/BarListChart";
import { ChartCard } from "@/app/components/charts/ChartCard";
import { Heatmap } from "@/app/components/charts/Heatmap";
import { Histogram } from "@/app/components/charts/Histogram";
import { ScatterPlot } from "@/app/components/charts/ScatterPlot";

const PAGE_SIZE = 50;

const FILTER_OPS: Array<{ value: FilterRule["op"]; label: string }> = [
  { value: "eq", label: "=" },
  { value: "contains", label: "contains" },
  { value: "gte", label: ">=" },
  { value: "lte", label: "<=" },
];

const inputStyle: React.CSSProperties = {
  padding: "7px 10px",
  border: "1px solid var(--border-dim)",
  borderRadius: "var(--radius-unified)",
  background: "var(--bg-surface)",
  color: "var(--text-primary)",
  fontSize: "12px",
};

const buttonStyle: React.CSSProperties = {
  padding: "7px 12px",
  border: "1px solid var(--border-dim)",
  borderRadius: "var(--radius-unified)",
  background: "transparent",
  color: "var(--text-secondary)",
  fontSize: "12px",
  cursor: "pointer",
};

const primaryButtonStyle: React.CSSProperties = {
  ...buttonStyle,
  background: "var(--gold)",
  border: "none",
  color: "#08080b",
  fontWeight: 600,
};

export default function AnalysisResultsPage() {
  const params = useParams();
  const analysisId = params.analysisId as string;

  return (
    <FilterProvider>
      <AnalysisResultsContent analysisId={analysisId} />
    </FilterProvider>
  );
}

function AnalysisResultsContent({ analysisId }: { analysisId: string }) {
  const {
    filters,
    search,
    sortCol,
    sortOrder,
    setFilters,
    setSearch,
    setSortCol,
    setSortOrder,
    replaceAll,
    clearAll,
  } = useFilterContext();

  const [tab, setTab] = useState<"table" | "analytics">("table");
  const [analysis, setAnalysis] = useState<AnalysisRecord | null>(null);
  const [results, setResults] = useState<AnalysisResults | null>(null);
  const [columns, setColumns] = useState<string[]>([]);
  const [page, setPage] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [searchDraft, setSearchDraft] = useState("");
  const [savedFilters, setSavedFilters] = useState<SavedFilter[]>([]);
  const [saveName, setSaveName] = useState("");

  const [descriptive, setDescriptive] = useState<AnalysisDescriptiveAnalytics | null>(null);
  const [correlations, setCorrelations] = useState<AnalysisCorrelationsResponse | null>(null);
  const [diagnostics, setDiagnostics] = useState<AnalysisDiagnosticsResponse | null>(null);
  const [clusters, setClusters] = useState<AnalysisClusterResponse | null>(null);

  const backendFilters = useMemo(
    () => filters.filter((rule) => rule.col && rule.val && rule.col !== "cluster_id"),
    [filters],
  );
  const filterKey = useMemo(() => JSON.stringify(filters), [filters]);
  const clusterFilter = filters.find((rule) => rule.col === "cluster_id") ?? null;
  const selectedCluster = clusterFilter?.val ?? null;
  const summary = analysis?.result_summary as AnalysisSummary | null;
  const taskForDiagnostics = summary?.models_applied.find((item) => !item.error)?.task ?? "sentiment";

  useEffect(() => {
    setSearchDraft(search);
  }, [search]);

  useEffect(() => {
    fetchAnalysisDetail(analysisId).then(setAnalysis).catch((err: unknown) => {
      setError(err instanceof Error ? err.message : String(err));
    });
    fetchSavedFilters("analysis_results").then(setSavedFilters).catch(() => {});
  }, [analysisId]);

  useEffect(() => {
    setPage(0);
  }, [filterKey, search, sortCol, sortOrder]);

  useEffect(() => {
    let active = true;
    setLoading(true);
    fetchAnalysisResults(analysisId, {
      offset: page * PAGE_SIZE,
      limit: PAGE_SIZE,
      sort_col: sortCol || undefined,
      sort_order: sortOrder,
      filters: backendFilters.length > 0 ? backendFilters : undefined,
      search: search || undefined,
    })
      .then((data) => {
        if (!active) return;
        setResults(data);
        setColumns(data.columns);
        setError(null);
      })
      .catch((err: unknown) => {
        if (!active) return;
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [analysisId, backendFilters, page, search, selectedCluster, sortCol, sortOrder]);

  useEffect(() => {
    if (tab !== "analytics") return;
    let active = true;
    Promise.allSettled([
      fetchAnalysisDescriptiveAnalytics(analysisId, {
        filters: backendFilters.length > 0 ? backendFilters : undefined,
        search: search || undefined,
      }),
      fetchAnalysisCorrelations(analysisId, {
        filters: backendFilters.length > 0 ? backendFilters : undefined,
        search: search || undefined,
      }),
      fetchAnalysisDiagnostics(analysisId, {
        task: taskForDiagnostics,
        filters: backendFilters.length > 0 ? backendFilters : undefined,
        search: search || undefined,
      }),
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
  }, [analysisId, backendFilters, search, selectedCluster, tab, taskForDiagnostics]);

  function setEqFilter(column: string, value: string) {
    setFilters((current) => {
      const exists = current.some(
        (rule) => rule.col === column && rule.op === "eq" && rule.val === value,
      );
      const next = current.filter((rule) => !(rule.col === column && rule.op === "eq"));
      return exists ? next : [...next, { col: column, op: "eq", val: value }];
    });
  }

  function setRangeFilter(column: string, min: number, max: number) {
    setFilters((current) => {
      const next = current.filter(
        (rule) => !(rule.col === column && (rule.op === "gte" || rule.op === "lte")),
      );
      return [
        ...next,
        { col: column, op: "gte", val: String(min) },
        { col: column, op: "lte", val: String(max) },
      ];
    });
  }

  function toggleCluster(clusterId: string) {
    setFilters((current) => {
      const next = current.filter((rule) => rule.col !== "cluster_id");
      return selectedCluster === clusterId
        ? next
        : [...next, { col: "cluster_id", op: "eq", val: clusterId }];
    });
  }

  function addFilter() {
    setFilters((current) => [...current, { col: columns[0] ?? "", op: "eq", val: "" }]);
  }

  function updateFilter(index: number, patch: Partial<FilterRule>) {
    const editable = filters.filter((rule) => rule.col !== "cluster_id");
    const next = editable.map((rule, i) => (i === index ? { ...rule, ...patch } : rule));
    if (clusterFilter) next.push(clusterFilter);
    setFilters(next);
  }

  function removeFilter(index: number) {
    const editable = filters.filter((rule) => rule.col !== "cluster_id");
    const next = editable.filter((_, i) => i !== index);
    if (clusterFilter) next.push(clusterFilter);
    setFilters(next);
  }

  async function saveCurrentFilter() {
    if (!saveName.trim()) return;
    try {
      const saved = await createSavedFilter(saveName.trim(), "analysis_results", {
        filters: backendFilters,
        search,
        sort_col: sortCol || undefined,
        sort_order: sortOrder,
      });
      setSavedFilters((current) => [saved, ...current]);
      setSaveName("");
    } catch {
      // Keep the current UI state if saving fails.
    }
  }

  function applySavedFilter(item: SavedFilter) {
    replaceAll({
      filters: item.filter_config.filters ?? [],
      search: item.filter_config.search ?? "",
      sortCol: item.filter_config.sort_col ?? "",
      sortOrder: item.filter_config.sort_order === "desc" ? "desc" : "asc",
    });
  }

  const editableFilters = filters.filter((rule) => rule.col !== "cluster_id");
  const totalPages = results ? Math.max(1, Math.ceil(results.total_rows / PAGE_SIZE)) : 1;
  const filteredCsv = getFilteredExportUrl(
    analysisId,
    "csv",
    backendFilters.length > 0 ? backendFilters : undefined,
    search || undefined,
    sortCol || undefined,
    sortOrder,
  );
  const filteredJson = getFilteredExportUrl(
    analysisId,
    "json",
    backendFilters.length > 0 ? backendFilters : undefined,
    search || undefined,
    sortCol || undefined,
    sortOrder,
  );

  const categoricalSections = descriptive ? buildCategoricalSections(descriptive.summary) : [];
  const numericSections = descriptive ? buildNumericSections(descriptive.summary) : [];
  const textSections = descriptive ? buildTextLengthSections(descriptive.summary) : [];
  const correlationCells = correlations ? buildCorrelationCells(correlations.correlations) : [];
  const confusionCells = diagnostics ? buildConfusionCells(diagnostics.diagnostics) : [];
  const calibrationBins = diagnostics ? buildCalibrationBins(diagnostics.diagnostics) : [];
  const clusterPoints = clusters ? buildClusterPoints(clusters) : [];
  const clusterCounts = clusters
    ? Object.entries(clusters.cluster_counts).map(([label, value]) => ({ label, value }))
    : [];

  const rowIdxIndex = results ? results.columns.indexOf("row_idx") : -1;
  const clusterRows =
    selectedCluster && clusters
      ? new Set(
          clusters.clusters
            .filter((item) => String(item.cluster) === selectedCluster)
            .map((item) => String(item.row_idx)),
        )
      : null;
  const visibleRows =
    results && rowIdxIndex >= 0 && clusterRows
      ? results.rows.filter((row) => clusterRows.has(String(row[rowIdxIndex])))
      : results?.rows ?? [];

  return (
    <div className="page-shell page-standard page-shell--xl animate-fade-up">
      <div style={{ fontSize: "12px", color: "var(--text-tertiary)", marginBottom: "14px" }}>
        <Link href="/analyses" style={{ color: "inherit", textDecoration: "none" }}>
          Analyses
        </Link>
        {" / "}
        <Link href={`/analyses/${analysisId}`} style={{ color: "inherit", textDecoration: "none" }}>
          {analysis?.name || analysisId.slice(0, 16)}
        </Link>
        {" / "}
        <span style={{ color: "var(--text-secondary)" }}>Results</span>
      </div>

      <div className="flex items-start justify-between gap-3" style={{ marginBottom: "16px", flexWrap: "wrap" }}>
        <div>
          <h1 style={{ margin: 0, fontFamily: "var(--font-syne)", fontSize: "clamp(20px, 2vw, 24px)", color: "var(--text-primary)" }}>
            {analysis?.name || analysisId}
          </h1>
          <div style={{ marginTop: "6px", fontSize: "12px", color: "var(--text-tertiary)" }}>
            {results ? `${results.total_rows.toLocaleString()} rows` : "Loading"}
            {backendFilters.length > 0 || search ? " | filtered" : ""}
            {selectedCluster ? ` | cluster ${selectedCluster}` : ""}
          </div>
        </div>
        <div className="flex gap-2" style={{ flexWrap: "wrap" }}>
          {backendFilters.length > 0 || search ? (
            <>
              <a href={filteredCsv} download style={buttonStyle}>Export Filtered CSV</a>
              <a href={filteredJson} download style={buttonStyle}>Export Filtered JSON</a>
            </>
          ) : (
            <>
              <a href={getAnalysisExportUrl(analysisId, "csv")} download style={buttonStyle}>Export CSV</a>
              <a href={getAnalysisExportUrl(analysisId, "json")} download style={buttonStyle}>Export JSON</a>
            </>
          )}
        </div>
      </div>

      <div style={{ display: "flex", gap: "8px", borderBottom: "1px solid var(--border-dim)", marginBottom: "16px", flexWrap: "wrap" }}>
        <button type="button" onClick={() => setTab("table")} style={{ ...buttonStyle, border: "none", borderBottom: tab === "table" ? "2px solid var(--gold)" : "2px solid transparent" }}>Table</button>
        <button type="button" onClick={() => setTab("analytics")} style={{ ...buttonStyle, border: "none", borderBottom: tab === "analytics" ? "2px solid var(--gold)" : "2px solid transparent" }}>Analytics</button>
      </div>

      <section style={{ padding: "16px", border: "1px solid var(--border-dim)", borderRadius: "var(--radius-unified)", background: "var(--bg-surface)", marginBottom: "16px" }}>
        <div className="flex items-center gap-2" style={{ flexWrap: "wrap", marginBottom: "10px" }}>
          <input value={searchDraft} onChange={(e) => setSearchDraft(e.target.value)} placeholder="Search" style={{ ...inputStyle, flex: "1 1 14rem", minWidth: "12rem" }} />
          <button type="button" onClick={() => setSearch(searchDraft.trim())} style={primaryButtonStyle}>Apply</button>
          <button type="button" onClick={() => { clearAll(); setSearchDraft(""); }} style={buttonStyle}>Clear</button>
          <button type="button" onClick={addFilter} style={buttonStyle}>Add Filter</button>
          <input value={saveName} onChange={(e) => setSaveName(e.target.value)} placeholder="Save current view" style={{ ...inputStyle, flex: "1 1 11rem", minWidth: "10rem" }} />
          <button type="button" onClick={saveCurrentFilter} disabled={!saveName.trim()} style={{ ...buttonStyle, opacity: saveName.trim() ? 1 : 0.6 }}>Save</button>
        </div>

        {editableFilters.map((rule, index) => (
          <div key={`${index}-${rule.col}-${rule.op}`} className="flex items-center gap-2" style={{ flexWrap: "wrap", marginBottom: "8px" }}>
            <select value={rule.col} onChange={(e) => updateFilter(index, { col: e.target.value })} style={inputStyle}>
              <option value="">Column</option>
              {columns.map((column) => <option key={column} value={column}>{column}</option>)}
            </select>
            <select value={rule.op} onChange={(e) => updateFilter(index, { op: e.target.value as FilterRule["op"] })} style={inputStyle}>
              {FILTER_OPS.map((op) => <option key={op.value} value={op.value}>{op.label}</option>)}
            </select>
            <input value={rule.val} onChange={(e) => updateFilter(index, { val: e.target.value })} placeholder="Value" style={inputStyle} />
            <button type="button" onClick={() => removeFilter(index)} style={buttonStyle}>Remove</button>
          </div>
        ))}

        {savedFilters.length > 0 && (
          <div className="flex items-center gap-2" style={{ flexWrap: "wrap", marginTop: "8px" }}>
            {savedFilters.slice(0, 6).map((item) => (
              <span key={item.id} style={{ display: "inline-flex", gap: "6px", alignItems: "center", padding: "4px 8px", border: "1px solid var(--border-dim)", borderRadius: "999px", fontSize: "11px" }}>
                <button type="button" onClick={() => applySavedFilter(item)} style={{ border: "none", background: "transparent", cursor: "pointer", color: "var(--text-secondary)" }}>{item.name}</button>
                <button type="button" onClick={() => { void deleteSavedFilter(item.id).then(() => setSavedFilters((current) => current.filter((saved) => saved.id !== item.id))).catch(() => {}); }} style={{ border: "none", background: "transparent", cursor: "pointer", color: "var(--text-tertiary)" }}>x</button>
              </span>
            ))}
          </div>
        )}
      </section>

      {error && <div style={{ color: "var(--error, #ef4444)", fontSize: "12px", marginBottom: "12px" }}>{error}</div>}

      {tab === "table" ? (
        <section style={{ border: "1px solid var(--border-dim)", borderRadius: "var(--radius-unified)", background: "var(--bg-surface)", overflow: "hidden" }}>
          {loading ? (
            <div style={{ padding: "18px", fontSize: "12px", color: "var(--text-tertiary)" }}>Loading results</div>
          ) : !results ? (
            <div style={{ padding: "18px", fontSize: "12px", color: "var(--text-tertiary)" }}>No results.</div>
          ) : (
            <>
              <div className="overflow-x-auto">
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      {results.columns.map((column) => (
                        <th key={column} style={{ padding: "10px", borderBottom: "1px solid var(--border-dim)", textAlign: "left", fontSize: "11px", color: "var(--text-secondary)", background: "var(--bg-base)" }}>
                          <button type="button" onClick={() => {
                            if (sortCol === column) setSortOrder(sortOrder === "asc" ? "desc" : "asc");
                            else { setSortCol(column); setSortOrder("asc"); }
                          }} style={{ border: "none", background: "transparent", cursor: "pointer", color: "inherit", padding: 0 }}>
                            {column}{sortCol === column ? (sortOrder === "asc" ? " ^" : " v") : ""}
                          </button>
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {visibleRows.map((row, rowIndex) => (
                      <tr key={`${page}-${rowIndex}`}>
                        {row.map((value, cellIndex) => (
                          <td key={`${rowIndex}-${cellIndex}`} title={value} style={{ padding: "10px", borderBottom: "1px solid var(--border-dim)", fontSize: "12px", color: "var(--text-secondary)", maxWidth: "320px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                            {value}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="flex items-center justify-between gap-2" style={{ padding: "12px 16px", flexWrap: "wrap" }}>
                <span style={{ fontSize: "11px", color: "var(--text-tertiary)" }}>Page {page + 1} of {totalPages}</span>
                <div className="flex gap-2">
                  <button type="button" onClick={() => setPage((current) => Math.max(0, current - 1))} disabled={page === 0} style={{ ...buttonStyle, opacity: page === 0 ? 0.5 : 1 }}>Prev</button>
                  <button type="button" onClick={() => setPage((current) => Math.min(totalPages - 1, current + 1))} disabled={page >= totalPages - 1} style={{ ...buttonStyle, opacity: page >= totalPages - 1 ? 0.5 : 1 }}>Next</button>
                </div>
              </div>
            </>
          )}
        </section>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
          {categoricalSections.length > 0 && (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))", gap: "16px" }}>
              {categoricalSections.map((section) => (
                <ChartCard key={section.column} title={section.column} subtitle="Click a bar to add or remove an equality filter.">
                  <BarListChart series={section.series} activeLabel={backendFilters.find((rule) => rule.col === section.column && rule.op === "eq")?.val ?? null} onSelect={(value) => setEqFilter(section.column, value)} />
                </ChartCard>
              ))}
            </div>
          )}

          {(textSections.length > 0 || numericSections.length > 0) && (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: "16px" }}>
              {textSections.map((section) => (
                <ChartCard key={`text-${section.column}`} title={`${section.column} length`} subtitle="Derived from quartiles. Click bins to set range filters.">
                  <Histogram bins={section.charBins} onRangeSelect={(range) => setRangeFilter(section.column, range.min, range.max)} />
                </ChartCard>
              ))}
              {numericSections.map((section) => (
                <ChartCard key={`num-${section.column}`} title={`${section.column} distribution`} subtitle="Derived from quartiles.">
                  <Histogram bins={section.bins} onRangeSelect={(range) => setRangeFilter(section.column, range.min, range.max)} />
                </ChartCard>
              ))}
            </div>
          )}

          {correlationCells.length > 0 && (
            <ChartCard title="Correlation Heatmap" subtitle="Associations in the current filtered slice.">
              <Heatmap cells={correlationCells} />
            </ChartCard>
          )}

          {diagnostics && (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 20rem), 1fr))",
                gap: "16px",
              }}
            >
              <ChartCard title={`${diagnostics.task} diagnostics`} subtitle={`Label column: ${diagnostics.label_col}`}>
                <Heatmap cells={confusionCells} />
              </ChartCard>
              <ChartCard title="Calibration" subtitle={`ECE ${diagnostics.diagnostics.ece.toFixed(3)}`}>
                <Histogram bins={calibrationBins} />
              </ChartCard>
            </div>
          )}

          {clusters && (
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 20rem), 1fr))",
                gap: "16px",
              }}
            >
              <ChartCard title="Embeddings" subtitle="Clustered 2D projection. Selecting a cluster updates the shared URL state.">
                <ScatterPlot points={clusterPoints} selectedGroup={selectedCluster} onSelectGroup={toggleCluster} />
              </ChartCard>
              <ChartCard title="Cluster Summary" subtitle="Click a cluster to focus the scatter and narrow the table when row indices are available.">
                <BarListChart series={clusterCounts} activeLabel={selectedCluster} onSelect={toggleCluster} />
              </ChartCard>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
