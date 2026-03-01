"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  fetchDatasets,
  fetchDatasetBranches,
  fetchDatasetVersions,
  fetchDatasetSchema,
  fetchColumnRoles,
  fetchModelDetail,
  fetchModelCompatibility,
  fetchModels,
  startAnalysis,
  DatasetSummary,
  DatasetBranch,
  DatasetVersion,
  ModelCompatibility,
  ModelSummary,
} from "@/app/lib/api";

const STEP_LABELS = ["Select Dataset", "Select Models", "Name & Launch"];

const SYSTEM_ROLES: { value: string; label: string }[] = [
  { value: "", label: "(not used)" },
  { value: "text", label: "text — feedback text" },
  { value: "sentiment", label: "sentiment — sentiment label" },
  { value: "language", label: "language — language label" },
  { value: "detail_level", label: "detail_level — detail level label" },
  { value: "survey_id", label: "survey_id — record ID" },
  ...Array.from({ length: 9 }, (_, i) => ({
    value: `item_${i + 1}`,
    label: `item_${i + 1} — ordinal survey item`,
  })),
];

const SYSTEM_ROLE_VALUES = new Set(
  SYSTEM_ROLES.map((r) => r.value).filter(Boolean)
);

const TASK_LABELS: Record<string, string> = {
  language: "Language Detection",
  sentiment: "Sentiment Classification",
  detail_level: "Detail Level",
};

export default function NewAnalysisPage() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const [step, setStep] = useState(0);

  // Data
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [modelCompatibility, setModelCompatibility] = useState<
    Record<string, ModelCompatibility>
  >({});
  const [compatibleModelIds, setCompatibleModelIds] = useState<Set<string>>(new Set());
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [modelsLoading, setModelsLoading] = useState(false);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [modelsError, setModelsError] = useState<string | null>(null);

  // Selections
  const [selectedDataset, setSelectedDataset] = useState<DatasetSummary | null>(null);
  const [dsBranches, setDsBranches] = useState<DatasetBranch[]>([]);
  const [selectedBranch, setSelectedBranch] = useState<string | null>(null);
  const [dsVersions, setDsVersions] = useState<DatasetVersion[]>([]);
  const [selectedVersion, setSelectedVersion] = useState<number | null>(null);
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);

  // Metadata
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [tags, setTags] = useState("");

  // Column role mapping: { datasetColumnName -> systemRole }
  const [datasetColumns, setDatasetColumns] = useState<string[]>([]);
  const [columnRoles, setColumnRoles] = useState<Record<string, string>>({});

  // Derive text_col from the mapping table
  const textCol =
    Object.entries(columnRoles).find(([, r]) => r === "text")?.[0] ?? "";

  // Launch
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [datasetPrefillApplied, setDatasetPrefillApplied] = useState(false);
  const [modelPrefillApplied, setModelPrefillApplied] = useState(false);

  const selectedVersionMeta =
    selectedVersion == null
      ? dsVersions[0] ?? null
      : dsVersions.find((v) => v.version === selectedVersion) ?? null;
  const selectedVersionId = selectedVersionMeta?.id;

  function inferFallbackRole(columnName: string): string {
    const normalized = columnName.trim().toLowerCase();
    if (SYSTEM_ROLE_VALUES.has(normalized)) return normalized;
    if (normalized === "text_feedback") return "text";
    if (normalized === "sentiment_class") return "sentiment";
    return "";
  }

  function summarizeCompatibility(report?: ModelCompatibility): string {
    if (!report) return "Checking compatibility...";
    if (report.ok) return "Compatible with selected dataset";
    if (report.reasons.length === 0) return "Incompatible with selected dataset";
    return report.reasons
      .map((reason) => {
        const message =
          typeof reason.message === "string" && reason.message.trim()
            ? reason.message.trim()
            : typeof reason.code === "string" && reason.code.trim()
            ? reason.code.trim()
            : "Incompatible with selected dataset";
        const fix =
          typeof reason.suggested_fix === "string" && reason.suggested_fix.trim()
            ? ` Fix: ${reason.suggested_fix.trim()}`
            : "";
        return `${message}${fix}`;
      })
      .join(" | ");
  }

  // Fetch branches when dataset changes
  useEffect(() => {
    if (!selectedDataset) {
      setDsBranches([]);
      setSelectedBranch(null);
      setModels([]);
      setModelCompatibility({});
      setCompatibleModelIds(new Set());
      setSelectedModelIds([]);
      return;
    }
    fetchDatasetBranches(selectedDataset.id)
      .then((branches) => {
        setDsBranches(branches);
        setSelectedBranch(null);
      })
      .catch(() => {});
  }, [selectedDataset]);

  // Fetch versions when dataset or branch changes
  useEffect(() => {
    if (!selectedDataset) {
      setDsVersions([]);
      setSelectedVersion(null);
      return;
    }
    const branchToFetch = selectedBranch ?? selectedDataset.default_branch_id ?? undefined;
    fetchDatasetVersions(selectedDataset.id, branchToFetch)
      .then((vs) => {
        setDsVersions(vs);
        setSelectedVersion(null);
      })
      .catch(() => {});
  }, [selectedDataset, selectedBranch]);

  // Fetch dataset schema and auto-populate column roles when dataset/version changes.
  useEffect(() => {
    if (!selectedDataset) {
      setDatasetColumns([]);
      setColumnRoles({});
      return;
    }
    fetchDatasetSchema(
      selectedDataset.id,
      selectedVersionId ? { version_id: selectedVersionId } : undefined
    )
      .then((res) => {
        const cols = res.columns.map((c: { name: string }) => c.name);
        setDatasetColumns(cols);
        return fetchColumnRoles(
          selectedDataset.id,
          selectedVersionId ? { version_id: selectedVersionId } : undefined
        )
          .then(({ column_roles }) => {
            const mapped: Record<string, string> = {};
            for (const col of cols) {
              mapped[col] = column_roles[col] ?? inferFallbackRole(col);
            }
            setColumnRoles(mapped);
          })
          .catch(() => {
            const auto: Record<string, string> = {};
            for (const col of cols) {
              auto[col] = inferFallbackRole(col);
            }
            setColumnRoles(auto);
          });
      })
      .catch(() => {/* best-effort */});
  }, [selectedDataset, selectedVersionId]);

  // Load datasets on mount
  useEffect(() => {
    fetchDatasets({ sort: "created_at", order: "desc", per_page: 100 })
      .then((r) => {
        setDatasets(r.datasets.filter((d) => d.status === "active"));
        setDatasetsLoading(false);
      })
      .catch((e) => {
        setDatasetsError(e.message);
        setDatasetsLoading(false);
      });
  }, []);

  useEffect(() => {
    if (datasetPrefillApplied || datasets.length === 0) return;

    const datasetIdFromQuery = searchParams.get("dataset_id");
    if (datasetIdFromQuery) {
      const ds = datasets.find((d) => d.id === datasetIdFromQuery) ?? null;
      if (ds) {
        setSelectedDataset(ds);
        setStep(1);
      }
      setDatasetPrefillApplied(true);
      return;
    }

    const modelIdFromQuery = searchParams.get("model_id");
    if (!modelIdFromQuery) {
      setDatasetPrefillApplied(true);
      return;
    }

    fetchModelDetail(modelIdFromQuery)
      .then((model) => {
        if (!model.dataset_id) return;
        const ds = datasets.find((d) => d.id === model.dataset_id) ?? null;
        if (ds) {
          setSelectedDataset(ds);
          setStep(1);
        }
      })
      .catch(() => {})
      .finally(() => setDatasetPrefillApplied(true));
  }, [datasets, datasetPrefillApplied, searchParams]);

  // Load models when dataset is selected (step 1→2)
  useEffect(() => {
    if (step !== 1 || !selectedDataset) return;
    setModelsLoading(true);
    setModelsError(null);
    fetchModels({ per_page: 100, sort: "created_at", order: "desc" })
      .then(async (allResp) => {
        const allActive = allResp.models.filter((m) => m.status === "active");
        const compatibilityEntries = await Promise.all(
          allActive.map(async (model) => {
            try {
              const report = await fetchModelCompatibility(model.id, {
                dataset_id: selectedDataset.id,
                dataset_version: selectedVersion,
                branch_id: selectedBranch,
              });
              return [model.id, report] as const;
            } catch (error: unknown) {
              return [
                model.id,
                {
                  ok: false,
                  reasons: [
                    {
                      code: "compatibility_check_failed",
                      message:
                        error instanceof Error
                          ? error.message
                          : "Compatibility check failed.",
                      suggested_fix:
                        "Retry the compatibility check or verify the selected dataset version.",
                    },
                  ],
                  resolved_columns: {},
                  required_roles: [],
                  preprocess_spec_id: null,
                  label_schema: {},
                  schema_columns: [],
                  text_col_used: null,
                  label_col_used: null,
                } as ModelCompatibility,
              ] as const;
            }
          })
        );

        const nextCompatibility: Record<string, ModelCompatibility> = {};
        const compatible = new Set<string>();
        for (const [modelId, report] of compatibilityEntries) {
          nextCompatibility[modelId] = report;
          if (report.ok) compatible.add(modelId);
        }

        setModelCompatibility(nextCompatibility);
        setCompatibleModelIds(compatible);
        setModels(allActive);
        setSelectedModelIds((prev) => prev.filter((id) => compatible.has(id)));
        setModelsLoading(false);
      })
      .catch((e) => {
        setModelsError(e.message);
        setModelCompatibility({});
        setModelsLoading(false);
      });
  }, [step, selectedDataset, selectedVersion, selectedBranch]);

  useEffect(() => {
    if (modelPrefillApplied || models.length === 0) return;
    const modelIdFromQuery = searchParams.get("model_id");
    if (!modelIdFromQuery) {
      setModelPrefillApplied(true);
      return;
    }
    if (compatibleModelIds.has(modelIdFromQuery)) {
      setSelectedModelIds((prev) =>
        prev.includes(modelIdFromQuery) ? prev : [...prev, modelIdFromQuery]
      );
    }
    setModelPrefillApplied(true);
  }, [models, compatibleModelIds, modelPrefillApplied, searchParams]);

  function toggleModel(modelId: string) {
    if (!compatibleModelIds.has(modelId)) return;
    setSelectedModelIds((prev) =>
      prev.includes(modelId)
        ? prev.filter((id) => id !== modelId)
        : [...prev, modelId]
    );
  }

  async function handleLaunch() {
    if (!selectedDataset || selectedModelIds.length === 0) return;
    setLaunching(true);
    setLaunchError(null);
    try {
      const tagList = tags
        .split(",")
        .map((t) => t.trim())
        .filter(Boolean);
      const job = await startAnalysis({
        dataset_id: selectedDataset.id,
        model_ids: selectedModelIds,
        name: name.trim() || undefined,
        description: description.trim() || undefined,
        tags: tagList.length > 0 ? tagList : undefined,
        dataset_version: selectedVersion,
        branch_id: selectedBranch,
        text_col: textCol || undefined,
      });
      router.push(`/analyses/${job.job_id}`);
    } catch (e: unknown) {
      setLaunchError(e instanceof Error ? e.message : String(e));
      setLaunching(false);
    }
  }

  return (
    <div className="page-shell page-standard page-shell--sm animate-fade-up">
      {/* Back link */}
      <div style={{ marginBottom: "6px" }}>
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
      <h1
        style={{
          fontFamily: "var(--font-syne)",
          fontWeight: 700,
          fontSize: "22px",
          color: "var(--text-primary)",
          marginBottom: "6px",
        }}
      >
        New Analysis
      </h1>
      <p style={{ color: "var(--text-tertiary)", fontSize: "13px", marginBottom: "28px" }}>
        Apply registered models to a dataset in batch. Results are aggregate summaries only.
      </p>

      {/* Step indicators */}
      <div className="flex flex-wrap gap-y-3" style={{ marginBottom: "32px" }}>
        {STEP_LABELS.map((label, i) => (
          <div key={i} className="flex items-center">
            <div className="flex items-center gap-2">
              <div
                style={{
                  width: "24px",
                  height: "24px",
                  borderRadius: "50%",
                  background:
                    i < step
                      ? "var(--success)"
                      : i === step
                      ? "var(--gold)"
                      : "var(--bg-surface)",
                  border: `1px solid ${
                    i < step
                      ? "var(--success)"
                      : i === step
                      ? "var(--gold)"
                      : "var(--border-dim)"
                  }`,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontSize: "11px",
                  fontWeight: 700,
                  color: i <= step ? "#000" : "var(--text-tertiary)",
                  flexShrink: 0,
                }}
              >
                {i < step ? "✓" : i + 1}
              </div>
              <span
                style={{
                  fontSize: "12px",
                  color: i === step ? "var(--text-primary)" : "var(--text-tertiary)",
                  fontWeight: i === step ? 500 : 400,
                  whiteSpace: "nowrap",
                }}
              >
                {label}
              </span>
            </div>
            {i < STEP_LABELS.length - 1 && (
              <div
                style={{
                  width: "32px",
                  height: "1px",
                  background: "var(--border-dim)",
                  margin: "0 8px",
                  flexShrink: 0,
                }}
              />
            )}
          </div>
        ))}
      </div>

      {/* Step 0: Select Dataset */}
      {step === 0 && (
        <div className="space-y-5">
          <div
            className="rounded-xl"
            style={{
              background: "var(--bg-surface)",
              border: "1px solid var(--border-dim)",
              padding: "24px",
            }}
          >
            <div style={{ marginBottom: "20px" }}>
              <label
                style={{
                  fontSize: "11px",
                  color: "var(--text-tertiary)",
                  fontFamily: "var(--font-syne)",
                  fontWeight: 700,
                  letterSpacing: "0.1em",
                  textTransform: "uppercase",
                  marginBottom: "6px",
                  display: "block",
                }}
              >
                Dataset
              </label>
              {datasetsLoading ? (
                <div
                  style={{
                    fontSize: "12px",
                    color: "var(--text-tertiary)",
                    fontFamily: "var(--font-jetbrains)",
                  }}
                >
                  Loading datasets...
                </div>
              ) : datasetsError ? (
                <div
                  style={{
                    fontSize: "12px",
                    color: "var(--error, #ef4444)",
                    fontFamily: "var(--font-jetbrains)",
                  }}
                >
                  {datasetsError}
                </div>
	              ) : datasets.length === 0 ? (
	                <div style={{ color: "var(--text-tertiary)", fontSize: "13px" }}>
	                  No active datasets. <Link href="/datasets/upload" style={{ color: "var(--gold)" }}>Upload one first.</Link>
	                </div>
              ) : (
                <select
                  value={selectedDataset?.id ?? ""}
                  onChange={(e) => {
                    const ds = datasets.find((d) => d.id === e.target.value) ?? null;
                    setSelectedDataset(ds);
                  }}
                  style={{
                    width: "100%",
                    background: "var(--bg-elevated)",
                    border: "1px solid var(--border)",
                    borderRadius: "6px",
                    padding: "7px 12px",
                    fontFamily: "var(--font-jetbrains)",
                    fontSize: "12px",
                    color: "var(--text-primary)",
                    outline: "none",
                  }}
                >
                  <option value="">- select a dataset -</option>
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name} ({d.row_count.toLocaleString()} rows, v{d.current_version})
                    </option>
                  ))}
                </select>
              )}
            </div>

            {selectedDataset && dsBranches.length > 0 && (
              <div style={{ marginBottom: "20px" }}>
                <label
                  style={{
                    fontSize: "11px",
                    color: "var(--text-tertiary)",
                    fontFamily: "var(--font-syne)",
                    fontWeight: 700,
                    letterSpacing: "0.1em",
                    textTransform: "uppercase",
                    marginBottom: "6px",
                    display: "block",
                  }}
                >
                  Branch
                </label>
                <select
                  value={selectedBranch ?? ""}
                  onChange={(e) => setSelectedBranch(e.target.value || null)}
                  style={{
                    width: "100%",
                    background: "var(--bg-elevated)",
                    border: "1px solid var(--border)",
                    borderRadius: "6px",
                    padding: "7px 12px",
                    fontFamily: "var(--font-jetbrains)",
                    fontSize: "12px",
                    color: "var(--text-primary)",
                    outline: "none",
                  }}
                >
                  <option value="">- default branch -</option>
                  {dsBranches.map((b) => (
                    <option key={b.id} value={b.id}>
                      {b.name}
                      {b.is_default ? " (default)" : ""}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {selectedDataset && dsVersions.length > 0 && (
              <div>
                <label
                  style={{
                    fontSize: "11px",
                    color: "var(--text-tertiary)",
                    fontFamily: "var(--font-syne)",
                    fontWeight: 700,
                    letterSpacing: "0.1em",
                    textTransform: "uppercase",
                    marginBottom: "6px",
                    display: "block",
                  }}
                >
                  Version
                </label>
                <select
                  value={selectedVersion ?? ""}
                  onChange={(e) => setSelectedVersion(e.target.value ? Number(e.target.value) : null)}
                  style={{
                    width: "100%",
                    background: "var(--bg-elevated)",
                    border: "1px solid var(--border)",
                    borderRadius: "6px",
                    padding: "7px 12px",
                    fontFamily: "var(--font-jetbrains)",
                    fontSize: "12px",
                    color: "var(--text-primary)",
                    outline: "none",
                  }}
                >
                  <option value="">- latest -</option>
                  {dsVersions.map((v) => (
                    <option key={v.id} value={v.version}>
                      v{v.version} - {v.reason || "no reason"} ({v.row_count.toLocaleString()} rows)
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          <div className="flex justify-end">
            <button
              onClick={() => setStep(1)}
              disabled={!selectedDataset}
              style={{
                padding: "9px 20px",
                background: selectedDataset ? "var(--gold)" : "var(--border-dim)",
                color: selectedDataset ? "#000" : "var(--text-tertiary)",
                border: "none",
                borderRadius: "6px",
                fontSize: "13px",
                fontWeight: 600,
                cursor: selectedDataset ? "pointer" : "not-allowed",
              }}
            >
              Next: Select Models {"->"}
            </button>
          </div>
        </div>
      )}
      {/* Step 1: Select Models */}
      {step === 1 && (
        <div>
          <h2 style={{ fontSize: "15px", fontWeight: 600, marginBottom: "6px", color: "var(--text-primary)" }}>
            Select Models
          </h2>
          <p style={{ fontSize: "12px", color: "var(--text-tertiary)", marginBottom: "16px" }}>
            Select one or more registered models to apply. Compatibility is checked against the selected dataset schema before launch.
          </p>
          {modelsError && (
            <div style={{ color: "var(--error, #ef4444)", fontSize: "13px", marginBottom: "12px" }}>
              {modelsError}
            </div>
          )}
          {modelsLoading ? (
            <div style={{ color: "var(--text-tertiary)", fontSize: "13px" }}>Loading models…</div>
	          ) : models.length === 0 ? (
	            <div style={{ color: "var(--text-tertiary)", fontSize: "13px" }}>
	              No registered models. <Link href="/training" style={{ color: "var(--gold)" }}>Train one first.</Link>
	            </div>
          ) : (
            <div className="flex flex-col gap-2">
	              {models.map((m) => {
	                const compatible = compatibleModelIds.has(m.id);
	                const selected = selectedModelIds.includes(m.id);
                    const compatibilityReport = modelCompatibility[m.id];
                    const compatibilityHint = summarizeCompatibility(compatibilityReport);
                const f1 = (m.metrics as { val?: { macro_f1?: number } })?.val?.macro_f1;
                return (
	                  <button
	                    key={m.id}
	                    onClick={() => toggleModel(m.id)}
	                    disabled={!compatible}
	                    title={compatibilityHint}
	                    style={{
                      textAlign: "left",
                      padding: "14px 16px",
	                      border: `1px solid ${selected ? "var(--gold)" : compatible ? "var(--border-dim)" : "var(--border)"}`,
                      borderRadius: "8px",
	                      background: selected ? "var(--gold-faint)" : compatible ? "var(--bg-surface)" : "var(--bg-base)",
	                      cursor: compatible ? "pointer" : "not-allowed",
	                      opacity: compatible ? 1 : 0.6,
                      display: "flex",
                      alignItems: "flex-start",
                      gap: "12px",
                    }}
                  >
                    <div
                      style={{
                        width: "16px",
                        height: "16px",
                        borderRadius: "3px",
                        border: `1px solid ${selected ? "var(--gold)" : "var(--border-dim)"}`,
                        background: selected ? "var(--gold)" : "transparent",
                        flexShrink: 0,
                        marginTop: "1px",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      {selected && <span style={{ fontSize: "10px", color: "#000", fontWeight: 700 }}>✓</span>}
                    </div>
                    <div style={{ flex: 1, minWidth: 0 }}>
	                      <div style={{ fontWeight: 500, fontSize: "13px", color: compatible ? "var(--text-primary)" : "var(--text-tertiary)" }}>
                        {m.name}
                      </div>
                      <div style={{ fontSize: "11px", color: "var(--text-tertiary)", marginTop: "3px" }}>
                        {TASK_LABELS[m.task] ?? m.task} · {m.model_type} · v{m.version}
                        {f1 != null && ` · F1 ${(f1 * 100).toFixed(1)}%`}
                        {!compatible && " · trained on a different dataset"}
                      </div>
                      {!compatible && (
                        <div
                          style={{
                            fontSize: "11px",
                            color: "var(--error, #ef4444)",
                            marginTop: "6px",
                          }}
                        >
                          {compatibilityHint}
                        </div>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>
          )}
          <div className="flex justify-between" style={{ marginTop: "24px" }}>
            <button
              onClick={() => setStep(0)}
              style={{
                padding: "9px 16px",
                border: "1px solid var(--border-dim)",
                borderRadius: "6px",
                background: "transparent",
                color: "var(--text-secondary)",
                fontSize: "13px",
                cursor: "pointer",
              }}
            >
              ← Back
            </button>
            <button
              onClick={() => setStep(2)}
              disabled={selectedModelIds.length === 0}
              style={{
                padding: "9px 20px",
                background: selectedModelIds.length > 0 ? "var(--gold)" : "var(--border-dim)",
                color: selectedModelIds.length > 0 ? "#000" : "var(--text-tertiary)",
                border: "none",
                borderRadius: "6px",
                fontSize: "13px",
                fontWeight: 600,
                cursor: selectedModelIds.length > 0 ? "pointer" : "not-allowed",
              }}
            >
              Next: Review & Launch →
            </button>
          </div>
        </div>
      )}

      {/* Step 2: Name & Launch */}
      {step === 2 && (
        <div>
          <h2 style={{ fontSize: "15px", fontWeight: 600, marginBottom: "16px", color: "var(--text-primary)" }}>
            Review & Launch
          </h2>

          {/* Summary card */}
          <div
            style={{
              padding: "16px",
              border: "1px solid var(--border-dim)",
              borderRadius: "8px",
              background: "var(--bg-surface)",
              marginBottom: "20px",
              fontSize: "13px",
            }}
          >
            <div style={{ marginBottom: "8px" }}>
              <span style={{ color: "var(--text-tertiary)", fontSize: "12px" }}>Dataset</span>
              <div style={{ fontWeight: 500, color: "var(--text-primary)" }}>
                {selectedDataset?.name}
                <span style={{ color: "var(--text-tertiary)", fontWeight: 400, marginLeft: "8px" }}>
                  {selectedDataset?.row_count.toLocaleString()} rows
                  {selectedBranch && dsBranches.length > 0
                    ? ` · ${dsBranches.find((b) => b.id === selectedBranch)?.name || selectedBranch}`
                    : ""}
                  {selectedVersion ? ` · v${selectedVersion}` : " · (latest)"}
                </span>
              </div>
            </div>
            <div>
              <span style={{ color: "var(--text-tertiary)", fontSize: "12px" }}>
                Models ({selectedModelIds.length})
              </span>
              <div className="flex flex-col gap-1" style={{ marginTop: "4px" }}>
                {models
                  .filter((m) => selectedModelIds.includes(m.id))
                  .map((m) => (
                    <div key={m.id} style={{ color: "var(--text-primary)", fontSize: "12px" }}>
                      {m.name}
                      <span style={{ color: "var(--text-tertiary)", marginLeft: "8px" }}>
                        {TASK_LABELS[m.task] ?? m.task} · {m.model_type}
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          </div>

          {/* Column role mapping */}
          {datasetColumns.length > 0 && (
            <div style={{ marginBottom: "20px" }}>
              <label style={{ fontSize: "12px", color: "var(--text-secondary)", display: "block", marginBottom: "4px" }}>
                Column Roles
              </label>
              <p style={{ fontSize: "11px", color: "var(--text-tertiary)", marginBottom: "8px", fontFamily: "var(--font-jetbrains)", margin: "0 0 8px 0" }}>
                For each column, select what it represents. Unneeded columns can be left as &quot;(not used)&quot;.
              </p>
              <div style={{ border: "1px solid var(--border-dim)", borderRadius: "6px", overflow: "hidden" }}>
                {/* Header */}
                <div className="flex" style={{ background: "var(--bg-base)", padding: "7px 12px", borderBottom: "1px solid var(--border-dim)", flexWrap: "wrap", gap: "4px 12px" }}>
                  <span style={{ flex: "1 1 12rem", fontSize: "10px", color: "var(--text-tertiary)", fontFamily: "var(--font-syne)", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase" }}>
                    Dataset Column
                  </span>
                  <span style={{ flex: "1 1 12rem", fontSize: "10px", color: "var(--text-tertiary)", fontFamily: "var(--font-syne)", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase" }}>
                    System Role
                  </span>
                </div>
                {/* Rows */}
                {datasetColumns.map((col, idx) => (
                  <div
                    key={col}
                    className="flex items-center"
                    style={{
                      padding: "6px 12px",
                      borderBottom: idx < datasetColumns.length - 1 ? "1px solid var(--border-dim)" : "none",
                      background: columnRoles[col] === "text" ? "var(--gold-faint)" : "transparent",
                      flexWrap: "wrap",
                      gap: "8px 12px",
                    }}
                  >
                    <span style={{ flex: "1 1 12rem", fontSize: "12px", color: "var(--text-primary)", fontFamily: "var(--font-jetbrains)", paddingRight: "8px", wordBreak: "break-all" }}>
                      {col}
                    </span>
                    <div style={{ flex: "1 1 12rem", minWidth: "min(100%, 12rem)" }}>
                      <select
                        value={columnRoles[col] ?? ""}
                        onChange={(e) => setColumnRoles((prev) => ({ ...prev, [col]: e.target.value }))}
                        style={{ width: "100%", padding: "5px 8px", border: "1px solid var(--border-dim)", borderRadius: "4px", background: "var(--bg-base)", color: "var(--text-primary)", fontSize: "12px", fontFamily: "var(--font-jetbrains)" }}
                      >
                        {SYSTEM_ROLES.map((r) => (
                          <option key={r.value} value={r.value}>{r.label}</option>
                        ))}
                      </select>
                    </div>
                  </div>
                ))}
              </div>
              <div style={{ marginTop: "6px", fontSize: "11px", color: "var(--text-tertiary)", fontFamily: "var(--font-jetbrains)" }}>
                Text:{" "}
                <span style={{ color: textCol ? "var(--success)" : "var(--error, #ef4444)", fontWeight: 600 }}>
                  {textCol || "none — assign text role to a column"}
                </span>
              </div>
            </div>
          )}

          {/* Metadata form */}
          <div className="flex flex-col gap-4" style={{ marginBottom: "20px" }}>
            <div>
              <label style={{ fontSize: "12px", color: "var(--text-secondary)", display: "block", marginBottom: "6px" }}>
                Name <span style={{ color: "var(--text-tertiary)" }}>(optional)</span>
              </label>
              <input
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="e.g. Q4 2025 sentiment analysis"
                style={{
                  width: "100%",
                  padding: "8px 12px",
                  border: "1px solid var(--border-dim)",
                  borderRadius: "6px",
                  background: "var(--bg-surface)",
                  color: "var(--text-primary)",
                  fontSize: "13px",
                  boxSizing: "border-box",
                }}
              />
            </div>
            <div>
              <label style={{ fontSize: "12px", color: "var(--text-secondary)", display: "block", marginBottom: "6px" }}>
                Description <span style={{ color: "var(--text-tertiary)" }}>(optional)</span>
              </label>
              <textarea
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Brief description of what this analysis covers…"
                rows={3}
                style={{
                  width: "100%",
                  padding: "8px 12px",
                  border: "1px solid var(--border-dim)",
                  borderRadius: "6px",
                  background: "var(--bg-surface)",
                  color: "var(--text-primary)",
                  fontSize: "13px",
                  resize: "vertical",
                  fontFamily: "inherit",
                  boxSizing: "border-box",
                }}
              />
            </div>
            <div>
              <label style={{ fontSize: "12px", color: "var(--text-secondary)", display: "block", marginBottom: "6px" }}>
                Tags <span style={{ color: "var(--text-tertiary)" }}>(comma-separated, optional)</span>
              </label>
              <input
                type="text"
                value={tags}
                onChange={(e) => setTags(e.target.value)}
                placeholder="e.g. q4, production, sentiment"
                style={{
                  width: "100%",
                  padding: "8px 12px",
                  border: "1px solid var(--border-dim)",
                  borderRadius: "6px",
                  background: "var(--bg-surface)",
                  color: "var(--text-primary)",
                  fontSize: "13px",
                  boxSizing: "border-box",
                }}
              />
            </div>
          </div>

          {launchError && (
            <div
              style={{
                padding: "10px 14px",
                borderRadius: "6px",
                background: "rgba(239,68,68,0.08)",
                border: "1px solid rgba(239,68,68,0.3)",
                color: "var(--error, #ef4444)",
                fontSize: "13px",
                marginBottom: "16px",
              }}
            >
              {launchError}
            </div>
          )}

          <div className="flex justify-between">
            <button
              onClick={() => setStep(1)}
              style={{
                padding: "9px 16px",
                border: "1px solid var(--border-dim)",
                borderRadius: "6px",
                background: "transparent",
                color: "var(--text-secondary)",
                fontSize: "13px",
                cursor: "pointer",
              }}
            >
              ← Back
            </button>
            <button
              onClick={handleLaunch}
              disabled={launching}
              style={{
                padding: "9px 24px",
                background: launching ? "var(--border-dim)" : "var(--gold)",
                color: launching ? "var(--text-tertiary)" : "#000",
                border: "none",
                borderRadius: "6px",
                fontSize: "13px",
                fontWeight: 600,
                cursor: launching ? "not-allowed" : "pointer",
              }}
            >
              {launching ? "Launching…" : "Launch Analysis"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}


