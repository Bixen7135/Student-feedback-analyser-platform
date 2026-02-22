"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  fetchDatasets,
  fetchDatasetSchema,
  fetchDatasetVersions,
  fetchDatasetBranches,
  startTraining,
  DatasetSummary,
  DatasetVersion,
  DatasetBranch,
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
  sentiment: "sentiment_class",
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
  { value: "text_feedback", label: "text_feedback — feedback text" },
  { value: "sentiment_class", label: "sentiment_class — sentiment label" },
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

  // Launch state
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);

  // Derive text_col and label_col from the mapping table
  const textCol =
    Object.entries(columnRoles).find(([, r]) => r === "text_feedback")?.[0] ?? "";
  const labelCol =
    Object.entries(columnRoles).find(
      ([, r]) => r === TASK_LABEL_COLS[selectedTask]
    )?.[0] ?? "";

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

  // Fetch dataset schema and auto-populate column roles whenever dataset changes
  useEffect(() => {
    if (!selectedDataset) {
      setDatasetColumns([]);
      setColumnRoles({});
      return;
    }
    fetchDatasetSchema(selectedDataset.id)
      .then((res) => {
        const cols = res.columns.map((c: { name: string }) => c.name);
        setDatasetColumns(cols);
        // Auto-assign role = column name when it matches a known system role,
        // otherwise leave it unmapped (not used).
        const auto: Record<string, string> = {};
        for (const col of cols) {
          auto[col] = SYSTEM_ROLE_VALUES.has(col) ? col : "";
        }
        setColumnRoles(auto);
      })
      .catch(() => {/* schema fetch is best-effort */});
  }, [selectedDataset]);

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
        <div style={card}>
          <div style={{ marginBottom: "16px" }}>
            <span style={label}>Dataset</span>
            {datasetsLoading && (
              <p style={{ fontSize: "13px", color: "var(--text-tertiary)" }}>
                Loading datasets…
              </p>
            )}
            {datasetsError && (
              <p style={{ fontSize: "13px", color: "var(--error)" }}>
                {datasetsError}
              </p>
            )}
            {!datasetsLoading && datasets.length === 0 && (
              <p style={{ fontSize: "13px", color: "var(--text-tertiary)" }}>
                No datasets uploaded yet.{" "}
                <a
                  href="/datasets/upload"
                  style={{ color: "var(--gold)", textDecoration: "none" }}
                >
                  Upload one first.
                </a>
              </p>
            )}
            {!datasetsLoading && datasets.length > 0 && (
              <select
                style={selectStyle}
                value={selectedDataset?.id ?? ""}
                onChange={(e) => {
                  const ds = datasets.find((d) => d.id === e.target.value) ?? null;
                  setSelectedDataset(ds);
                }}
              >
                <option value="">— select a dataset —</option>
                {datasets.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name} ({d.row_count.toLocaleString()} rows)
                  </option>
                ))}
              </select>
            )}
          </div>

          {selectedDataset && (
            <>
              <div
                style={{
                  background: "var(--bg-base)",
                  borderRadius: "6px",
                  padding: "10px 14px",
                  fontSize: "12px",
                  color: "var(--text-secondary)",
                  fontFamily: "var(--font-jetbrains)",
                  marginBottom: "12px",
                }}
              >
                <strong style={{ color: "var(--text-primary)" }}>
                  {selectedDataset.name}
                </strong>
                {"  "}·{"  "}
                {selectedDataset.row_count.toLocaleString()} rows{"  "}·{"  "}v
                {selectedDataset.current_version}
                {selectedDataset.description && (
                  <div style={{ marginTop: "4px", color: "var(--text-tertiary)" }}>
                    {selectedDataset.description}
                  </div>
                )}
              </div>

              {/* Branch picker */}
              {dsBranches.length > 1 && (
                <div style={{ marginBottom: "12px" }}>
                  <span style={label}>Branch</span>
                  <select
                    style={selectStyle}
                    value={selectedBranch ?? "default"}
                    onChange={(e) =>
                      setSelectedBranch(
                        e.target.value === "default" ? null : e.target.value
                      )
                    }
                  >
                    <option value="default">
                      {dsBranches.find(b => b.is_default)?.name || "main"} — default branch
                    </option>
                    {dsBranches
                      .filter((b) => !b.is_default)
                      .map((b) => (
                        <option key={b.id} value={b.id}>
                          {b.name} {b.description ? `— ${b.description}` : ""}
                        </option>
                      ))}
                  </select>
                </div>
              )}

              {/* Version picker */}
              {dsVersions.length > 0 && (
                <div style={{ marginBottom: "16px" }}>
                  <span style={label}>Version</span>
                  {(() => {
                    // Get versions for the current branch (dsVersions already filtered)
                    const branchVersions = dsVersions;
                    const headVersion = branchVersions[0]; // API returns latest first
                    const currentBranch = dsBranches.find(b => b.id === selectedBranch) || dsBranches.find(b => b.is_default);

                    return (
                      <select
                        style={selectStyle}
                        value={selectedVersion ?? "latest"}
                        onChange={(e) =>
                          setSelectedVersion(
                            e.target.value === "latest" ? null : parseInt(e.target.value)
                          )
                        }
                      >
                        <option value="latest">
                          {headVersion
                            ? `v${headVersion.version} — latest on ${currentBranch?.name || "default"} (${headVersion.row_count.toLocaleString()} rows)`
                            : "Latest"
                          }
                        </option>
                        {branchVersions.slice(1).map((v) => (
                          <option key={v.version} value={v.version}>
                            v{v.version} — {new Date(v.created_at).toLocaleDateString()} ({v.row_count.toLocaleString()} rows){v.reason ? ` · ${v.reason}` : ""}
                          </option>
                        ))}
                      </select>
                    );
                  })()}
                </div>
              )}
            </>
          )}

          <div className="flex justify-end">
            <button
              style={btnPrimary}
              disabled={!selectedDataset}
              onClick={() => setStep(1)}
            >
              Next →
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
                        columnRoles[col] === "text_feedback" ||
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
                    {textCol || "none — assign text_feedback role"}
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
