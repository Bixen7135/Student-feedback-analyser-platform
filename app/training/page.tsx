"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  fetchDatasets,
  fetchDatasetSchema,
  fetchColumnRoles,
  fetchDatasetVersions,
  fetchDatasetBranches,
  fetchModels,
  startTraining,
  DatasetSummary,
  DatasetVersion,
  DatasetBranch,
  ModelSummary,
  StartTrainingRequest,
} from "@/app/lib/api";

type Task = "language" | "sentiment" | "detail_level";
type ModelType = "tfidf" | "char_ngram";
type Balancing = "none" | "class_weight" | "oversample";

const TASK_LABELS: Record<Task, string> = {
  language: "Language Detection",
  sentiment: "Sentiment Classification",
  detail_level: "Detail Level",
};

const TASK_LABEL_COLS: Record<Task, string> = {
  language: "language",
  sentiment: "sentiment",
  detail_level: "detail_level",
};

const MODEL_LABELS: Record<ModelType, string> = {
  tfidf: "TF-IDF + Logistic Regression",
  char_ngram: "Char N-gram + Logistic Regression",
};

const STEP_LABELS = [
  "Select Dataset",
  "Select Task",
  "Configure",
  "Review & Launch",
];

// All known system field names that a column can be mapped to.
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

export default function TrainingPage() {
  const router = useRouter();
  const searchParams = useSearchParams();

  // Wizard state
  const [step, setStep] = useState(0);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);

  // Form values
  const [selectedDataset, setSelectedDataset] = useState<DatasetSummary | null>(null);
  const [dsVersions, setDsVersions] = useState<DatasetVersion[]>([]);
  const [selectedVersion, setSelectedVersion] = useState<number | null>(null); // null = latest
  const [dsBranches, setDsBranches] = useState<DatasetBranch[]>([]);
  const [selectedBranch, setSelectedBranch] = useState<string | null>(null); // null = default
  const [selectedTask, setSelectedTask] = useState<Task>("sentiment");
  const [selectedModel, setSelectedModel] = useState<ModelType>("tfidf");
  const [trainName, setTrainName] = useState("");
  const [seed, setSeed] = useState(42);
  const [balancing, setBalancing] = useState<Balancing>("class_weight");
  const [trainRatio, setTrainRatio] = useState(0.80);
  const [valRatio, setValRatio] = useState(0.10);

  // Column role mapping: { datasetColumnName -> systemRole }
  const [datasetColumns, setDatasetColumns] = useState<string[]>([]);
  const [columnRoles, setColumnRoles] = useState<Record<string, string>>({});

  // Fine-tuning state
  const [fineTuneEnabled, setFineTuneEnabled] = useState(false);
  const [baseModelId, setBaseModelId] = useState<string | null>(null);
  const [availableBaseModels, setAvailableBaseModels] = useState<ModelSummary[]>([]);
  const [baseModelsLoading, setBaseModelsLoading] = useState(false);

  // Launch state
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [queryPrefillApplied, setQueryPrefillApplied] = useState(false);

  const selectedVersionMeta =
    selectedVersion == null
      ? dsVersions[0] ?? null
      : dsVersions.find((v) => v.version === selectedVersion) ?? null;
  const selectedVersionId = selectedVersionMeta?.id;

  // Derive text_col and label_col from the mapping table
  const textCol =
    Object.entries(columnRoles).find(([, r]) => r === "text")?.[0] ?? "";
  const labelCol =
    Object.entries(columnRoles).find(
      ([, r]) => r === TASK_LABEL_COLS[selectedTask]
    )?.[0] ?? "";

  function inferFallbackRole(columnName: string): string {
    const normalized = columnName.trim().toLowerCase();
    if (SYSTEM_ROLE_VALUES.has(normalized)) return normalized;
    if (normalized === "text_feedback") return "text";
    if (normalized === "sentiment_class") return "sentiment";
    return "";
  }

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
    if (queryPrefillApplied || datasets.length === 0) return;
    const datasetIdFromQuery = searchParams.get("dataset_id");
    if (!datasetIdFromQuery) {
      setQueryPrefillApplied(true);
      return;
    }
    const ds = datasets.find((d) => d.id === datasetIdFromQuery) ?? null;
    if (ds) setSelectedDataset(ds);
    setQueryPrefillApplied(true);
  }, [datasets, queryPrefillApplied, searchParams]);

  // Fetch versions when dataset changes; auto-select latest
  useEffect(() => {
    if (!selectedDataset) {
      setDsVersions([]);
      setSelectedVersion(null);
      return;
    }
    // When selectedBranch is null, use default_branch_id to get only default branch versions
    const branchToFetch = selectedBranch ?? selectedDataset.default_branch_id ?? undefined;
    fetchDatasetVersions(selectedDataset.id, branchToFetch)
      .then((vs) => {
        setDsVersions(vs);
        setSelectedVersion(null); // default to latest
      })
      .catch(() => {});
  }, [selectedDataset, selectedBranch]);

  // Fetch branches when dataset changes; auto-select default
  useEffect(() => {
    if (!selectedDataset) {
      setDsBranches([]);
      setSelectedBranch(null);
      return;
    }
    fetchDatasetBranches(selectedDataset.id)
      .then((branches) => {
        setDsBranches(branches);
        setSelectedBranch(null); // default to default branch
      })
      .catch(() => {});
  }, [selectedDataset]);

  // Fetch dataset schema and auto-populate column roles whenever dataset/version changes.
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
      .catch(() => {/* schema fetch is best-effort */});
  }, [selectedDataset, selectedVersionId]);

  // Fetch models available for fine-tuning (same task + model type)
  useEffect(() => {
    if (!fineTuneEnabled) return;
    setBaseModelsLoading(true);
    setBaseModelId(null);
    fetchModels({ task: selectedTask, model_type: selectedModel, per_page: 100 })
      .then((r) => {
        setAvailableBaseModels(r.models);
        setBaseModelsLoading(false);
      })
      .catch(() => setBaseModelsLoading(false));
  }, [fineTuneEnabled, selectedTask, selectedModel]);

  // Reset fine-tune state when toggled off
  useEffect(() => {
    if (!fineTuneEnabled) {
      setBaseModelId(null);
      setAvailableBaseModels([]);
    }
  }, [fineTuneEnabled]);

  const testRatio = Math.max(0.02, parseFloat((1 - trainRatio - valRatio).toFixed(3)));

  async function handleLaunch() {
    if (!selectedDataset) return;
    setLaunching(true);
    setLaunchError(null);
    try {
      const req: StartTrainingRequest = {
        dataset_id: selectedDataset.id,
        dataset_version: selectedVersion,
        branch_id: selectedBranch,
        task: selectedTask,
        model_type: selectedModel,
        seed,
        name: trainName.trim() || undefined,
        base_model_id: fineTuneEnabled ? baseModelId : undefined,
        config: {
          train_ratio: trainRatio,
          val_ratio: valRatio,
          test_ratio: testRatio,
          class_balancing: balancing,
          text_col: textCol || undefined,
          label_col: labelCol || undefined,
        },
      };
      const job = await startTraining(req);
      router.push(`/training/${job.job_id}`);
    } catch (e: unknown) {
      setLaunchError(e instanceof Error ? e.message : String(e));
      setLaunching(false);
    }
  }

  const card = {
    background: "var(--bg-surface)",
    border: "1px solid var(--border)",
    borderRadius: "10px",
    padding: "24px",
    marginBottom: "16px",
  } as const;

  const inputStyle = {
    background: "var(--bg-base)",
    border: "1px solid var(--border)",
    borderRadius: "6px",
    padding: "8px 12px",
    color: "var(--text-primary)",
    fontSize: "13px",
    fontFamily: "var(--font-jetbrains)",
    width: "100%",
    boxSizing: "border-box" as const,
  } as const;

  const selectStyle = { ...inputStyle };

  const datasetStepSelectStyle = {
    width: "100%",
    background: "var(--bg-elevated)",
    border: "1px solid var(--border)",
    borderRadius: "6px",
    padding: "7px 12px",
    fontFamily: "var(--font-jetbrains)",
    fontSize: "12px",
    color: "var(--text-primary)",
    outline: "none",
  } as const;

  const btnPrimary = {
    background: "var(--gold)",
    color: "#000",
    border: "none",
    borderRadius: "6px",
    padding: "9px 20px",
    fontSize: "13px",
    fontWeight: 600,
    cursor: "pointer",
    fontFamily: "var(--font-jetbrains)",
  } as const;

  const btnGhost = {
    background: "transparent",
    color: "var(--text-secondary)",
    border: "1px solid var(--border)",
    borderRadius: "6px",
    padding: "9px 20px",
    fontSize: "13px",
    cursor: "pointer",
    fontFamily: "var(--font-jetbrains)",
  } as const;

  const label = {
    fontSize: "11px",
    color: "var(--text-tertiary)",
    fontFamily: "var(--font-syne)",
    fontWeight: 700,
    letterSpacing: "0.1em",
    textTransform: "uppercase" as const,
    marginBottom: "6px",
    display: "block",
  } as const;

  return (
    <div style={{ padding: "32px", maxWidth: "720px" }} className="animate-fade-up">
      {/* Header */}
      <div style={{ marginBottom: "28px" }}>
        <h1
          style={{
            fontFamily: "var(--font-syne)",
            fontSize: "20px",
            fontWeight: 700,
            color: "var(--text-primary)",
            marginBottom: "6px",
          }}
        >
          Train a Classifier
        </h1>
        <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
          Train a text classifier on an uploaded dataset and register it in the
          model registry.
        </p>
      </div>

      {/* Step indicator */}
      <div className="flex items-center gap-0" style={{ marginBottom: "28px" }}>
        {STEP_LABELS.map((stepLabel, i) => (
          <div key={i} className="flex items-center">
            <div
              className="flex items-center gap-2"
              style={{ cursor: i < step ? "pointer" : "default" }}
              onClick={() => i < step && setStep(i)}
            >
              <div
                className="flex items-center justify-center rounded-full flex-shrink-0"
                style={{
                  width: "24px",
                  height: "24px",
                  background:
                    i === step
                      ? "var(--gold)"
                      : i < step
                      ? "var(--success)"
                      : "var(--bg-surface)",
                  border:
                    i === step
                      ? "none"
                      : i < step
                      ? "none"
                      : "1px solid var(--border)",
                  color: i <= step ? "#000" : "var(--text-tertiary)",
                  fontSize: "11px",
                  fontWeight: 700,
                  fontFamily: "var(--font-jetbrains)",
                }}
              >
                {i < step ? "✓" : i + 1}
              </div>
              <span
                style={{
                  fontSize: "11px",
                  color:
                    i === step
                      ? "var(--text-primary)"
                      : "var(--text-tertiary)",
                  fontFamily: "var(--font-syne)",
                  fontWeight: i === step ? 700 : 400,
                  whiteSpace: "nowrap",
                }}
              >
                {stepLabel}
              </span>
            </div>
            {i < STEP_LABELS.length - 1 && (
              <div
                style={{
                  width: "32px",
                  height: "1px",
                  background:
                    i < step ? "var(--success)" : "var(--border-dim)",
                  margin: "0 8px",
                  flexShrink: 0,
                }}
              />
            )}
          </div>
        ))}
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Step 0: Select Dataset */}
      {/* ------------------------------------------------------------------ */}
      {step === 0 && (
        <div className="space-y-5 animate-fade-up">
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
                  fontFamily: "var(--font-syne)",
                  fontSize: "9.5px",
                  fontWeight: 700,
                  letterSpacing: "0.14em",
                  textTransform: "uppercase",
                  color: "var(--text-tertiary)",
                  marginBottom: "8px",
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
                    color: "var(--error)",
                    fontFamily: "var(--font-jetbrains)",
                  }}
                >
                  {datasetsError}
                </div>
              ) : datasets.length === 0 ? (
                <p style={{ fontSize: "13px", color: "var(--text-tertiary)" }}>
                  No datasets uploaded yet.{" "}
                  <Link
                    href="/datasets/upload"
                    style={{ color: "var(--gold)", textDecoration: "none" }}
                  >
                    Upload one first.
                  </Link>
                </p>
              ) : (
                <select
                  style={datasetStepSelectStyle}
                  value={selectedDataset?.id ?? ""}
                  onChange={(e) => {
                    const ds = datasets.find((d) => d.id === e.target.value) ?? null;
                    setSelectedDataset(ds);
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
                    fontFamily: "var(--font-syne)",
                    fontSize: "9.5px",
                    fontWeight: 700,
                    letterSpacing: "0.14em",
                    textTransform: "uppercase",
                    color: "var(--text-tertiary)",
                    marginBottom: "8px",
                    display: "block",
                  }}
                >
                  Branch
                </label>
                <select
                  style={datasetStepSelectStyle}
                  value={selectedBranch ?? ""}
                  onChange={(e) => setSelectedBranch(e.target.value || null)}
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
                    fontFamily: "var(--font-syne)",
                    fontSize: "9.5px",
                    fontWeight: 700,
                    letterSpacing: "0.14em",
                    textTransform: "uppercase",
                    color: "var(--text-tertiary)",
                    marginBottom: "8px",
                    display: "block",
                  }}
                >
                  Version
                </label>
                <select
                  style={datasetStepSelectStyle}
                  value={selectedVersion ?? ""}
                  onChange={(e) =>
                    setSelectedVersion(e.target.value ? Number(e.target.value) : null)
                  }
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

          {!selectedDataset && !datasetsLoading && datasets.length > 0 && (
            <div
              style={{
                fontSize: "11px",
                color: "var(--text-tertiary)",
                fontFamily: "var(--font-jetbrains)",
                textAlign: "center",
              }}
            >
              Select a dataset to continue to task configuration.
            </div>
          )}

          <div>
            <button
              className="w-full rounded-lg flex items-center justify-center gap-2 transition-all duration-150"
              style={{
                background: !selectedDataset ? "var(--gold-muted)" : "var(--gold)",
                color: "#08080B",
                padding: "11px 24px",
                fontSize: "13px",
                fontWeight: 600,
                fontFamily: "var(--font-syne)",
                letterSpacing: "0.05em",
                border: "none",
                cursor: !selectedDataset ? "not-allowed" : "pointer",
                opacity: !selectedDataset ? 0.7 : 1,
              }}
              disabled={!selectedDataset}
              onClick={() => setStep(1)}
            >
              Next
              <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
                <path
                  d="M3 6.5H10M7.5 4L10 6.5L7.5 9"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Step 1: Select Task */}
      {/* ------------------------------------------------------------------ */}
      {step === 1 && (
        <div style={card}>
          <span style={label}>Task</span>
          <div className="flex flex-col gap-2" style={{ marginBottom: "20px" }}>
            {(Object.keys(TASK_LABELS) as Task[]).map((t) => (
              <label
                key={t}
                className="flex items-center gap-3"
                style={{
                  padding: "12px 14px",
                  borderRadius: "8px",
                  border: `1px solid ${selectedTask === t ? "var(--gold)" : "var(--border)"}`,
                  background:
                    selectedTask === t
                      ? "var(--gold-faint)"
                      : "var(--bg-base)",
                  cursor: "pointer",
                }}
              >
                <input
                  type="radio"
                  name="task"
                  value={t}
                  checked={selectedTask === t}
                  onChange={() => setSelectedTask(t)}
                  style={{ accentColor: "var(--gold)" }}
                />
                <div>
                  <div
                    style={{
                      fontSize: "13px",
                      fontWeight: 600,
                      color: "var(--text-primary)",
                    }}
                  >
                    {TASK_LABELS[t]}
                  </div>
                  <div
                    style={{
                      fontSize: "11px",
                      color: "var(--text-tertiary)",
                      fontFamily: "var(--font-jetbrains)",
                    }}
                  >
                    label column: {TASK_LABEL_COLS[t]}
                  </div>
                </div>
              </label>
            ))}
          </div>

          <span style={{ ...label, marginTop: "16px" }}>Model Type</span>
          <div className="flex flex-col gap-2" style={{ marginBottom: "20px" }}>
            {(Object.keys(MODEL_LABELS) as ModelType[]).map((m) => (
              <label
                key={m}
                className="flex items-center gap-3"
                style={{
                  padding: "12px 14px",
                  borderRadius: "8px",
                  border: `1px solid ${selectedModel === m ? "var(--gold)" : "var(--border)"}`,
                  background:
                    selectedModel === m
                      ? "var(--gold-faint)"
                      : "var(--bg-base)",
                  cursor: "pointer",
                }}
              >
                <input
                  type="radio"
                  name="modelType"
                  value={m}
                  checked={selectedModel === m}
                  onChange={() => setSelectedModel(m)}
                  style={{ accentColor: "var(--gold)" }}
                />
                <div>
                  <div
                    style={{
                      fontSize: "13px",
                      fontWeight: 600,
                      color: "var(--text-primary)",
                    }}
                  >
                    {MODEL_LABELS[m]}
                  </div>
                </div>
              </label>
            ))}
          </div>

          <div className="flex justify-between">
            <button style={btnGhost} onClick={() => setStep(0)}>
              ← Back
            </button>
            <button style={btnPrimary} onClick={() => setStep(2)}>
              Next →
            </button>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Step 2: Configure */}
      {/* ------------------------------------------------------------------ */}
      {step === 2 && (
        <div style={card}>
          <div style={{ marginBottom: "16px" }}>
            <span style={label}>Model Name (optional)</span>
            <input
              style={inputStyle}
              placeholder="e.g. sentiment_tfidf_v1"
              value={trainName}
              onChange={(e) => setTrainName(e.target.value)}
            />
          </div>

          <div className="flex gap-4" style={{ marginBottom: "16px" }}>
            <div style={{ flex: 1 }}>
              <span style={label}>Train ratio</span>
              <input
                type="number"
                style={inputStyle}
                min={0.5}
                max={0.9}
                step={0.05}
                value={trainRatio}
                onChange={(e) => setTrainRatio(parseFloat(e.target.value))}
              />
            </div>
            <div style={{ flex: 1 }}>
              <span style={label}>Val ratio</span>
              <input
                type="number"
                style={inputStyle}
                min={0.02}
                max={0.3}
                step={0.05}
                value={valRatio}
                onChange={(e) => setValRatio(parseFloat(e.target.value))}
              />
            </div>
            <div style={{ flex: 1 }}>
              <span style={label}>Test ratio</span>
              <div
                style={{
                  ...inputStyle,
                  color: "var(--text-tertiary)",
                  display: "flex",
                  alignItems: "center",
                }}
              >
                {testRatio.toFixed(2)} (auto)
              </div>
            </div>
          </div>

          <div style={{ marginBottom: "16px" }}>
            <span style={label}>Class balancing</span>
            <select
              style={selectStyle}
              value={balancing}
              onChange={(e) => setBalancing(e.target.value as Balancing)}
            >
              <option value="class_weight">
                class_weight (balanced weights in LR)
              </option>
              <option value="oversample">
                oversample (random oversampling of minority classes)
              </option>
              <option value="none">none (raw class frequencies)</option>
            </select>
          </div>

          <div style={{ marginBottom: "20px" }}>
            <span style={label}>Random seed</span>
            <input
              type="number"
              style={inputStyle}
              min={0}
              value={seed}
              onChange={(e) => setSeed(parseInt(e.target.value, 10))}
            />
          </div>

          {/* Fine-tuning toggle */}
          <div
            style={{
              marginBottom: "20px",
              border: `1px solid ${fineTuneEnabled ? "var(--gold)" : "var(--border)"}`,
              borderRadius: "8px",
              padding: "14px",
              background: fineTuneEnabled ? "var(--gold-faint)" : "var(--bg-base)",
            }}
          >
            <label
              className="flex items-center gap-3"
              style={{ cursor: "pointer", marginBottom: fineTuneEnabled ? "12px" : 0 }}
            >
              <input
                type="checkbox"
                checked={fineTuneEnabled}
                onChange={(e) => setFineTuneEnabled(e.target.checked)}
                style={{ accentColor: "var(--gold)", width: "16px", height: "16px" }}
              />
              <div>
                <span
                  style={{
                    fontSize: "13px",
                    fontWeight: 600,
                    color: "var(--text-primary)",
                  }}
                >
                  Fine-tune from existing model
                </span>
                <div
                  style={{
                    fontSize: "11px",
                    color: "var(--text-tertiary)",
                    fontFamily: "var(--font-jetbrains)",
                  }}
                >
                  Warm-start LR weights from a previously trained model of the same task
                  and type.
                </div>
              </div>
            </label>

            {fineTuneEnabled && (
              <div>
                <span style={label}>Base model</span>
                {baseModelsLoading && (
                  <p style={{ fontSize: "12px", color: "var(--text-tertiary)" }}>
                    Loading models…
                  </p>
                )}
                {!baseModelsLoading && availableBaseModels.length === 0 && (
                  <p style={{ fontSize: "12px", color: "var(--text-tertiary)" }}>
                    No compatible models found for task &quot;{selectedTask}&quot; /
                    type &quot;{selectedModel}&quot;. Train a model first.
                  </p>
                )}
                {!baseModelsLoading && availableBaseModels.length > 0 && (
                  <select
                    style={selectStyle}
                    value={baseModelId ?? ""}
                    onChange={(e) =>
                      setBaseModelId(e.target.value || null)
                    }
                  >
                    <option value="">— select base model —</option>
                    {availableBaseModels.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name} · v{m.version} ·{" "}
                        {new Date(m.created_at).toLocaleDateString()}
                        {m.metrics &&
                        typeof (m.metrics as Record<string, unknown>).val === "object"
                          ? ` · val F1 ${(
                              (
                                (m.metrics as Record<string, unknown>).val as Record<
                                  string,
                                  number
                                >
                              ).macro_f1 ?? 0
                            ).toFixed(3)}`
                          : ""}
                      </option>
                    ))}
                  </select>
                )}
              </div>
            )}
          </div>

          {/* Column role mapping table */}
          {datasetColumns.length > 0 && (
            <div style={{ marginBottom: "16px" }}>
              <span style={label}>Column Roles</span>
              <p style={{ fontSize: "11px", color: "var(--text-tertiary)", marginBottom: "8px", fontFamily: "var(--font-jetbrains)" }}>
                For each column in the dataset, select what it represents in the system.
                Unneeded columns can be left as &quot;(not used)&quot;.
              </p>
              <div
                style={{
                  border: "1px solid var(--border)",
                  borderRadius: "6px",
                  overflow: "hidden",
                }}
              >
                {/* Header */}
                <div
                  className="flex"
                  style={{
                    background: "var(--bg-base)",
                    padding: "7px 12px",
                    borderBottom: "1px solid var(--border)",
                  }}
                >
                  <span
                    style={{
                      flex: 1,
                      fontSize: "10px",
                      color: "var(--text-tertiary)",
                      fontFamily: "var(--font-syne)",
                      fontWeight: 700,
                      letterSpacing: "0.1em",
                      textTransform: "uppercase",
                    }}
                  >
                    Dataset Column
                  </span>
                  <span
                    style={{
                      flex: 1,
                      fontSize: "10px",
                      color: "var(--text-tertiary)",
                      fontFamily: "var(--font-syne)",
                      fontWeight: 700,
                      letterSpacing: "0.1em",
                      textTransform: "uppercase",
                    }}
                  >
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
                      borderBottom:
                        idx < datasetColumns.length - 1
                          ? "1px solid var(--border-dim)"
                          : "none",
                      background:
                        columnRoles[col] === "text" ||
                        columnRoles[col] === TASK_LABEL_COLS[selectedTask]
                          ? "var(--gold-faint)"
                          : "transparent",
                    }}
                  >
                    <span
                      style={{
                        flex: 1,
                        fontSize: "12px",
                        color: "var(--text-primary)",
                        fontFamily: "var(--font-jetbrains)",
                        paddingRight: "8px",
                        wordBreak: "break-all",
                      }}
                    >
                      {col}
                    </span>
                    <div style={{ flex: 1 }}>
                      <select
                        value={columnRoles[col] ?? ""}
                        onChange={(e) =>
                          setColumnRoles((prev) => ({
                            ...prev,
                            [col]: e.target.value,
                          }))
                        }
                        style={{
                          ...selectStyle,
                          fontSize: "12px",
                          padding: "5px 8px",
                        }}
                      >
                        {SYSTEM_ROLES.map((r) => (
                          <option key={r.value} value={r.value}>
                            {r.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                ))}
              </div>

              {/* Derived summary */}
              <div
                style={{
                  marginTop: "8px",
                  fontSize: "11px",
                  color: "var(--text-tertiary)",
                  fontFamily: "var(--font-jetbrains)",
                  display: "flex",
                  gap: "16px",
                }}
              >
                <span>
                  Text:{" "}
                  <span
                    style={{
                      color: textCol
                        ? "var(--success)"
                        : "var(--error, #ef4444)",
                      fontWeight: 600,
                    }}
                  >
                    {textCol || "none — assign text role"}
                  </span>
                </span>
                <span>
                  Label ({selectedTask}):{" "}
                  <span
                    style={{
                      color: labelCol
                        ? "var(--success)"
                        : "var(--error, #ef4444)",
                      fontWeight: 600,
                    }}
                  >
                    {labelCol ||
                      `none — assign ${TASK_LABEL_COLS[selectedTask]} role`}
                  </span>
                </span>
              </div>
            </div>
          )}

          <div className="flex justify-between">
            <button style={btnGhost} onClick={() => setStep(1)}>
              ← Back
            </button>
            <button style={btnPrimary} onClick={() => setStep(3)}>
              Next →
            </button>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Step 3: Review & Launch */}
      {/* ------------------------------------------------------------------ */}
      {step === 3 && (
        <div style={card}>
          <div style={{ marginBottom: "20px" }}>
            <div
              style={{
                fontFamily: "var(--font-syne)",
                fontSize: "13px",
                fontWeight: 700,
                color: "var(--text-secondary)",
                marginBottom: "14px",
              }}
            >
              Training Summary
            </div>

            {[
              ["Dataset", selectedDataset?.name ?? "—"],
              ["Branch", (() => {
                const branch = dsBranches.find(b => b.id === selectedBranch);
                return branch ? branch.name : (dsBranches.find(b => b.is_default)?.name || "default");
              })()],
              ["Version", selectedVersion ? `v${selectedVersion}` : "(latest on branch)"],
              ["Task", TASK_LABELS[selectedTask]],
              ["Model type", MODEL_LABELS[selectedModel]],
              ["Name", trainName || "(auto-generated)"],
              ...(fineTuneEnabled
                ? [
                    [
                      "Fine-tune from",
                      baseModelId
                        ? (availableBaseModels.find((m) => m.id === baseModelId)?.name ??
                            baseModelId)
                        : "(none selected — fresh training)",
                    ],
                  ]
                : []),
              ["Text column", textCol || "(auto-detect)"],
              ["Label column", labelCol || "(auto-detect)"],
              [
                "Split",
                `${(trainRatio * 100).toFixed(0)} / ${(valRatio * 100).toFixed(0)} / ${(testRatio * 100).toFixed(0)}`,
              ],
              ["Balancing", balancing],
              ["Seed", String(seed)],
            ].map(([k, v]) => (
              <div
                key={k}
                className="flex"
                style={{
                  padding: "7px 0",
                  borderBottom: "1px solid var(--border-dim)",
                  fontSize: "13px",
                }}
              >
                <span
                  style={{
                    width: "140px",
                    color: "var(--text-tertiary)",
                    flexShrink: 0,
                    fontFamily: "var(--font-jetbrains)",
                    fontSize: "12px",
                  }}
                >
                  {k}
                </span>
                <span style={{ color: "var(--text-primary)" }}>{v}</span>
              </div>
            ))}
          </div>

          <div
            style={{
              background: "var(--bg-base)",
              borderRadius: "6px",
              padding: "10px 14px",
              fontSize: "12px",
              color: "var(--text-tertiary)",
              fontFamily: "var(--font-jetbrains)",
              marginBottom: "20px",
              lineHeight: 1.6,
            }}
          >
            Batch only. Training runs in background. Not for real-time or
            individual-level decisions. No causal claims.
          </div>

          {launchError && (
            <div
              style={{
                background: "rgba(239,68,68,0.08)",
                border: "1px solid rgba(239,68,68,0.3)",
                borderRadius: "6px",
                padding: "10px 14px",
                fontSize: "13px",
                color: "var(--error, #ef4444)",
                marginBottom: "16px",
              }}
            >
              {launchError}
            </div>
          )}

          <div className="flex justify-between">
            <button style={btnGhost} onClick={() => setStep(2)}>
              ← Back
            </button>
            <button
              style={{
                ...btnPrimary,
                opacity: launching ? 0.6 : 1,
                cursor: launching ? "not-allowed" : "pointer",
              }}
              disabled={launching}
              onClick={handleLaunch}
            >
              {launching ? "Launching…" : "Launch Training"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

