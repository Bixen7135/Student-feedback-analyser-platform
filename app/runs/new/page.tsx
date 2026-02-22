"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import {
  createRun,
  startStage,
  fetchRunDetail,
  fetchDatasets,
  fetchDatasetBranches,
  fetchDatasetVersions,
  RunDetail,
  DatasetSummary,
  DatasetBranch,
  DatasetVersion,
} from "@/app/lib/api";
import { Disclaimer } from "@/app/components/Disclaimer";
import { StageProgress } from "@/app/components/StageProgress";

// ─── Stage constants ────────────────────────────────────────────────────────

const STAGE_NAMES = [
  "ingest_preprocess",
  "psychometrics",
  "splits",
  "text_tasks",
  "fusion",
  "contradiction",
  "evaluation",
  "reporting",
];

const STAGE_LABELS: Record<string, string> = {
  ingest_preprocess: "Ingest & Preprocess",
  psychometrics:     "Psychometrics (CFA)",
  splits:            "Stratified Split",
  text_tasks:        "Text Classifiers",
  fusion:            "Fusion",
  contradiction:     "Contradiction Monitor",
  evaluation:        "Evaluation",
  reporting:         "Reports & Model Cards",
};

const STAGE_ESTIMATES: Record<string, number> = {
  ingest_preprocess: 30,
  psychometrics:     60,
  splits:             5,
  text_tasks:       300,
  fusion:           120,
  contradiction:     30,
  evaluation:        60,
  reporting:         30,
};

const TERMINAL = new Set(["completed", "failed", "skipped"]);

const STEP_LABELS = ["Select Dataset", "Configure", "Review & Launch"];

// ─── Helpers ────────────────────────────────────────────────────────────────

function stageEtaLabel(
  name: string,
  status: string,
  startedAt: string | null,
  durationSeconds: number | null,
  nowMs: number
): string {
  if (status === "completed") return durationSeconds != null ? `${durationSeconds.toFixed(1)}s` : "";
  if (status === "failed") return "failed";
  if (status === "skipped") return "skipped";
  if (status === "running") {
    if (!startedAt) return "running…";
    const elapsed = Math.max(0, Math.floor((nowMs - new Date(startedAt).getTime()) / 1000));
    const est = STAGE_ESTIMATES[name] ?? 60;
    const remaining = est - elapsed;
    if (remaining > 5) return `${elapsed}s · ~${remaining}s left`;
    if (remaining > 0) return `${elapsed}s · almost done`;
    return `${elapsed}s elapsed`;
  }
  const est = STAGE_ESTIMATES[name];
  return est != null ? `~${est}s est.` : "";
}

function allDone(run: RunDetail): boolean {
  if (STAGE_NAMES.every((s) => { const st = run.stages[s]; return st && TERMINAL.has(st.status); })) return true;
  if (STAGE_NAMES.some((s) => run.stages[s]?.status === "failed")) return true;
  return false;
}

function dotColor(status: string): string {
  if (status === "completed") return "var(--success)";
  if (status === "failed")    return "var(--error)";
  if (status === "running")   return "var(--running)";
  return "var(--text-tertiary)";
}

// ─── Shared heading style ────────────────────────────────────────────────────

const labelStyle: React.CSSProperties = {
  fontFamily: "var(--font-syne)",
  fontSize: "9.5px",
  fontWeight: 700,
  letterSpacing: "0.14em",
  textTransform: "uppercase" as const,
  color: "var(--text-tertiary)",
  marginBottom: "8px",
  display: "block",
};

// ─── Step indicator ─────────────────────────────────────────────────────────

function StepIndicator({ step }: { step: number }) {
  return (
    <div className="flex items-center gap-0" style={{ marginBottom: "28px" }}>
      {STEP_LABELS.map((label, i) => (
        <div key={label} className="flex items-center">
          <div className="flex items-center gap-2">
            <span
              style={{
                width: "20px",
                height: "20px",
                borderRadius: "50%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                fontSize: "10px",
                fontFamily: "var(--font-jetbrains)",
                fontWeight: 700,
                background: i < step ? "var(--gold)" : i === step ? "var(--gold)" : "var(--bg-elevated)",
                color: i <= step ? "#08080B" : "var(--text-tertiary)",
                border: i > step ? "1px solid var(--border)" : "none",
                flexShrink: 0,
              }}
            >
              {i < step ? "✓" : String(i + 1)}
            </span>
            <span
              style={{
                fontSize: "11px",
                fontFamily: "var(--font-syne)",
                color: i === step ? "var(--text-primary)" : "var(--text-tertiary)",
                fontWeight: i === step ? 600 : 400,
              }}
            >
              {label}
            </span>
          </div>
          {i < STEP_LABELS.length - 1 && (
            <div
              style={{
                width: "24px",
                height: "1px",
                background: i < step ? "var(--gold)" : "var(--border-dim)",
                margin: "0 8px",
                flexShrink: 0,
              }}
            />
          )}
        </div>
      ))}
    </div>
  );
}

// ─── Step 1: Dataset selection ───────────────────────────────────────────────

function SelectDatasetStep({
  datasets,
  loading,
  error,
  selectedDataset,
  onSelect,
  branches,
  selectedBranch,
  onSelectBranch,
  versions,
  selectedVersion,
  onSelectVersion,
  onNext,
}: {
  datasets: DatasetSummary[];
  loading: boolean;
  error: string | null;
  selectedDataset: DatasetSummary | null;
  onSelect: (ds: DatasetSummary | null) => void;
  branches: DatasetBranch[];
  selectedBranch: string | null;
  onSelectBranch: (id: string | null) => void;
  versions: DatasetVersion[];
  selectedVersion: number | null;
  onSelectVersion: (v: number | null) => void;
  onNext: () => void;
}) {
  const selectStyle: React.CSSProperties = {
    width: "100%",
    background: "var(--bg-elevated)",
    border: "1px solid var(--border)",
    borderRadius: "6px",
    padding: "7px 12px",
    fontFamily: "var(--font-jetbrains)",
    fontSize: "12px",
    color: "var(--text-primary)",
    outline: "none",
  };

  return (
    <div className="space-y-5 animate-fade-up">
      <div
        className="rounded-xl"
        style={{ background: "var(--bg-surface)", border: "1px solid var(--border-dim)", padding: "24px" }}
      >
        {/* Dataset dropdown */}
        <div style={{ marginBottom: "20px" }}>
          <label style={labelStyle}>Dataset</label>
          {loading ? (
            <div style={{ fontSize: "12px", color: "var(--text-tertiary)", fontFamily: "var(--font-jetbrains)" }}>
              Loading datasets…
            </div>
          ) : error ? (
            <div style={{ fontSize: "12px", color: "var(--error)", fontFamily: "var(--font-jetbrains)" }}>{error}</div>
          ) : (
            <select
              value={selectedDataset?.id ?? ""}
              onChange={(e) => onSelect(datasets.find((d) => d.id === e.target.value) ?? null)}
              style={selectStyle}
            >
              <option value="">— select a dataset —</option>
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.row_count.toLocaleString()} rows, v{d.current_version})
                </option>
              ))}
            </select>
          )}
        </div>

        {/* Branch dropdown */}
        {selectedDataset && branches.length > 0 && (
          <div style={{ marginBottom: "20px" }}>
            <label style={labelStyle}>Branch</label>
            <select
              value={selectedBranch ?? ""}
              onChange={(e) => onSelectBranch(e.target.value || null)}
              style={selectStyle}
            >
              <option value="">— default branch —</option>
              {branches.map((b) => (
                <option key={b.id} value={b.id}>
                  {b.name}{b.is_default ? " (default)" : ""}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Version dropdown */}
        {selectedDataset && versions.length > 0 && (
          <div>
            <label style={labelStyle}>Version</label>
            <select
              value={selectedVersion ?? ""}
              onChange={(e) => onSelectVersion(e.target.value ? Number(e.target.value) : null)}
              style={selectStyle}
            >
              <option value="">— latest —</option>
              {versions.map((v) => (
                <option key={v.id} value={v.version}>
                  v{v.version} — {v.reason || "no reason"} ({v.row_count.toLocaleString()} rows)
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      {/* Use default dataset fallback hint */}
      {!selectedDataset && !loading && (
        <div
          style={{
            fontSize: "11px",
            color: "var(--text-tertiary)",
            fontFamily: "var(--font-jetbrains)",
            textAlign: "center" as const,
          }}
        >
          No dataset selected — the pipeline will use the default{" "}
          <span style={{ color: "var(--text-secondary)" }}>/mnt/data/dataset.csv</span>
        </div>
      )}

      <button
        onClick={onNext}
        className="w-full rounded-lg flex items-center justify-center gap-2 transition-all duration-150"
        style={{
          background: "var(--gold)",
          color: "#08080B",
          padding: "11px 24px",
          fontSize: "13px",
          fontWeight: 600,
          fontFamily: "var(--font-syne)",
          letterSpacing: "0.05em",
          border: "none",
          cursor: "pointer",
        }}
      >
        Next
        <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
          <path d="M3 6.5H10M7.5 4L10 6.5L7.5 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </button>
    </div>
  );
}

// ─── Step 2: Configure ────────────────────────────────────────────────────────

function ConfigureStep({
  seed,
  setSeed,
  name,
  setName,
  onBack,
  onNext,
}: {
  seed: number;
  setSeed: (v: number) => void;
  name: string;
  setName: (v: string) => void;
  onBack: () => void;
  onNext: () => void;
}) {
  const inputStyle: React.CSSProperties = {
    background: "var(--bg-elevated)",
    border: "1px solid var(--border)",
    borderRadius: "6px",
    padding: "7px 12px",
    fontFamily: "var(--font-jetbrains)",
    fontSize: "13px",
    color: "var(--text-primary)",
    outline: "none",
  };

  return (
    <div className="space-y-5 animate-fade-up">
      <div
        className="rounded-xl"
        style={{ background: "var(--bg-surface)", border: "1px solid var(--border-dim)", padding: "24px" }}
      >
        {/* Name */}
        <div style={{ marginBottom: "20px" }}>
          <label style={labelStyle}>Run Name (optional)</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Baseline v1"
            style={{ ...inputStyle, width: "100%" }}
          />
        </div>

        {/* Seed */}
        <div>
          <label style={labelStyle}>Random Seed</label>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
            style={{ ...inputStyle, width: "120px" }}
          />
          <div style={{ fontSize: "11px", color: "var(--text-tertiary)", marginTop: "6px", fontFamily: "var(--font-jetbrains)" }}>
            Fixed seed ensures reproducibility across runs.
          </div>
        </div>
      </div>

      <div className="flex gap-3">
        <button
          onClick={onBack}
          className="rounded-lg flex items-center justify-center"
          style={{
            background: "var(--bg-surface)",
            border: "1px solid var(--border)",
            color: "var(--text-secondary)",
            padding: "11px 20px",
            fontSize: "13px",
            fontFamily: "var(--font-syne)",
            cursor: "pointer",
          }}
        >
          Back
        </button>
        <button
          onClick={onNext}
          className="flex-1 rounded-lg flex items-center justify-center gap-2 transition-all duration-150"
          style={{
            background: "var(--gold)",
            color: "#08080B",
            padding: "11px 24px",
            fontSize: "13px",
            fontWeight: 600,
            fontFamily: "var(--font-syne)",
            letterSpacing: "0.05em",
            border: "none",
            cursor: "pointer",
          }}
        >
          Next
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
            <path d="M3 6.5H10M7.5 4L10 6.5L7.5 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      </div>
    </div>
  );
}

// ─── Step 3: Review & Launch ─────────────────────────────────────────────────

const PIPELINE_STEPS = [
  "Ingest & preprocess dataset",
  "Run ordinal CFA psychometrics",
  "Stratified 80/10/10 split",
  "Train text classifiers (language, sentiment, detail level)",
  "Run fusion experiments",
  "Run contradiction monitoring",
  "Evaluate all models",
  "Generate reports & model cards",
];

function ReviewStep({
  selectedDataset,
  selectedBranch,
  branches,
  selectedVersion,
  seed,
  name,
  loading,
  error,
  onBack,
  onLaunch,
}: {
  selectedDataset: DatasetSummary | null;
  selectedBranch: string | null;
  branches: DatasetBranch[];
  selectedVersion: number | null;
  seed: number;
  name: string;
  loading: boolean;
  error: string | null;
  onBack: () => void;
  onLaunch: () => void;
}) {
  const branchName = branches.find((b) => b.id === selectedBranch)?.name ?? "default";

  const rowStyle: React.CSSProperties = {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "8px 0",
    borderBottom: "1px solid var(--border-dim)",
    fontSize: "12px",
  };

  return (
    <div className="space-y-5 animate-fade-up">
      <Disclaimer />

      <div
        className="rounded-xl"
        style={{ background: "var(--bg-surface)", border: "1px solid var(--border-dim)", padding: "24px" }}
      >
        {/* Summary table */}
        <div style={{ marginBottom: "20px" }}>
          <div style={labelStyle as React.CSSProperties}>Run Summary</div>
          <div>
            <div style={rowStyle}>
              <span style={{ color: "var(--text-tertiary)", fontFamily: "var(--font-syne)" }}>Dataset</span>
              <span style={{ color: "var(--text-primary)", fontFamily: "var(--font-jetbrains)" }}>
                {selectedDataset ? selectedDataset.name : "default (/mnt/data/dataset.csv)"}
              </span>
            </div>
            {selectedDataset && (
              <>
                <div style={rowStyle}>
                  <span style={{ color: "var(--text-tertiary)", fontFamily: "var(--font-syne)" }}>Branch</span>
                  <span style={{ color: "var(--text-primary)", fontFamily: "var(--font-jetbrains)" }}>{branchName}</span>
                </div>
                <div style={rowStyle}>
                  <span style={{ color: "var(--text-tertiary)", fontFamily: "var(--font-syne)" }}>Version</span>
                  <span style={{ color: "var(--text-primary)", fontFamily: "var(--font-jetbrains)" }}>
                    {selectedVersion != null ? `v${selectedVersion}` : "latest"}
                  </span>
                </div>
              </>
            )}
            <div style={rowStyle}>
              <span style={{ color: "var(--text-tertiary)", fontFamily: "var(--font-syne)" }}>Name</span>
              <span style={{ color: "var(--text-primary)", fontFamily: "var(--font-jetbrains)" }}>
                {name || "(auto-generated)"}
              </span>
            </div>
            <div style={{ ...rowStyle, borderBottom: "none" }}>
              <span style={{ color: "var(--text-tertiary)", fontFamily: "var(--font-syne)" }}>Seed</span>
              <span style={{ color: "var(--text-primary)", fontFamily: "var(--font-jetbrains)" }}>{seed}</span>
            </div>
          </div>
        </div>

        {/* Pipeline steps */}
        <div>
          <div style={labelStyle as React.CSSProperties}>Pipeline Stages</div>
          <ol style={{ listStyle: "none", padding: 0, margin: 0, display: "flex", flexDirection: "column", gap: "6px" }}>
            {PIPELINE_STEPS.map((step, i) => (
              <li key={step} className="flex items-start gap-3" style={{ fontSize: "12px", color: "var(--text-secondary)" }}>
                <span style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--gold)", flexShrink: 0, marginTop: "1px", minWidth: "16px" }}>
                  {String(i + 1).padStart(2, "0")}
                </span>
                {step}
              </li>
            ))}
          </ol>
        </div>
      </div>

      {error && (
        <div
          className="rounded-lg"
          style={{ background: "var(--error-dim)", border: "1px solid var(--error)", padding: "12px 16px", color: "var(--error)", fontSize: "12px", fontFamily: "var(--font-jetbrains)" }}
        >
          {error}
        </div>
      )}

      <div className="flex gap-3">
        <button
          onClick={onBack}
          disabled={loading}
          className="rounded-lg flex items-center justify-center"
          style={{
            background: "var(--bg-surface)",
            border: "1px solid var(--border)",
            color: "var(--text-secondary)",
            padding: "11px 20px",
            fontSize: "13px",
            fontFamily: "var(--font-syne)",
            cursor: loading ? "not-allowed" : "pointer",
            opacity: loading ? 0.6 : 1,
          }}
        >
          Back
        </button>
        <button
          onClick={onLaunch}
          disabled={loading}
          className="flex-1 rounded-lg flex items-center justify-center gap-2 transition-all duration-150"
          style={{
            background: loading ? "var(--gold-muted)" : "var(--gold)",
            color: "#08080B",
            padding: "11px 24px",
            fontSize: "13px",
            fontWeight: 600,
            fontFamily: "var(--font-syne)",
            letterSpacing: "0.05em",
            border: "none",
            cursor: loading ? "not-allowed" : "pointer",
            opacity: loading ? 0.7 : 1,
          }}
        >
          {loading ? "Launching…" : "Launch Full Pipeline"}
          {!loading && (
            <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
              <path d="M3 6.5H10M7.5 4L10 6.5L7.5 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          )}
        </button>
      </div>
    </div>
  );
}

// ─── Progress view ────────────────────────────────────────────────────────────

function ProgressView({
  runId,
  run,
  phase,
  pollError,
}: {
  runId: string;
  run: RunDetail | null;
  phase: "running" | "done";
  pollError: string | null;
}) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (phase !== "running") return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [phase]);

  const completedCount = run ? STAGE_NAMES.filter((s) => TERMINAL.has(run.stages[s]?.status ?? "")).length : 0;
  const hasFailed = run ? STAGE_NAMES.some((s) => run.stages[s]?.status === "failed") : false;
  const isDone = phase === "done";

  const heading = isDone ? (hasFailed ? "Run Encountered Errors" : "Pipeline Complete") : "Pipeline Running…";

  return (
    <div style={{ padding: "32px", maxWidth: "640px" }} className="animate-fade-up">
      <div style={{ marginBottom: "28px" }}>
        <div style={{ fontFamily: "var(--font-syne)", fontSize: "9.5px", fontWeight: 700, letterSpacing: "0.18em", textTransform: "uppercase", color: "var(--text-tertiary)", marginBottom: "6px" }}>
          Pipeline
        </div>
        <h1 style={{ fontFamily: "var(--font-syne)", fontWeight: 700, fontSize: "22px", color: hasFailed ? "var(--error)" : isDone ? "var(--success)" : "var(--text-primary)", letterSpacing: "-0.01em" }}>
          {heading}
        </h1>
      </div>

      <div className="space-y-4">
        {/* Run ID */}
        <div className="rounded-xl" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-dim)", padding: "14px 20px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ fontFamily: "var(--font-syne)", fontSize: "9px", fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--text-tertiary)" }}>Run ID</div>
          <div style={{ fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--text-secondary)" }}>{runId}</div>
        </div>

        {/* Stage progress */}
        <div className="rounded-xl" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-dim)", padding: "20px 24px" }}>
          <div className="flex items-center justify-between" style={{ marginBottom: "14px" }}>
            <div style={{ fontFamily: "var(--font-syne)", fontSize: "9.5px", fontWeight: 700, letterSpacing: "0.16em", textTransform: "uppercase", color: "var(--text-tertiary)" }}>Stage Progress</div>
            <div style={{ fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--text-tertiary)" }}>{completedCount} / {STAGE_NAMES.length}</div>
          </div>

          {pollError && !run && (
            <div className="rounded-lg" style={{ background: "var(--error-dim)", border: "1px solid var(--error)", padding: "10px 14px", marginBottom: "12px", color: "var(--error)", fontSize: "11px", fontFamily: "var(--font-jetbrains)" }}>
              {pollError}
            </div>
          )}

          {run ? (
            <>
              <StageProgress stages={run.stages} />
              <div style={{ marginTop: "18px", display: "flex", flexDirection: "column", gap: "0px" }}>
                {STAGE_NAMES.map((name, i) => {
                  const st = run.stages[name];
                  const status = st?.status ?? "pending";
                  const isRunning = status === "running";
                  const isActive = isRunning || TERMINAL.has(status);
                  return (
                    <div key={name} className="flex items-center justify-between" style={{ padding: "7px 0", borderBottom: i < STAGE_NAMES.length - 1 ? "1px solid var(--border-dim)" : "none" }}>
                      <div className="flex items-center gap-2.5">
                        <span className={`rounded-full shrink-0${isRunning ? " animate-pulse-dot" : ""}`} style={{ width: "6px", height: "6px", background: dotColor(status) }} />
                        <span style={{ fontFamily: "var(--font-jetbrains)", fontSize: "12px", color: isActive ? "var(--text-primary)" : "var(--text-tertiary)" }}>
                          {STAGE_LABELS[name] ?? name}
                        </span>
                      </div>
                      <div style={{ fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: status === "failed" ? "var(--error)" : status === "completed" ? "var(--success)" : "var(--text-tertiary)" }}>
                        {stageEtaLabel(name, status, st?.started_at ?? null, st?.duration_seconds ?? null, now)}
                      </div>
                    </div>
                  );
                })}
              </div>

              {hasFailed && (
                <div className="rounded-lg" style={{ background: "var(--error-dim)", border: "1px solid var(--error)", padding: "10px 14px", marginTop: "14px", color: "var(--error)", fontSize: "11px", fontFamily: "var(--font-jetbrains)", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
                  {STAGE_NAMES.map((n) => run.stages[n]).filter((s) => s?.status === "failed" && s.error).map((s) => s!.error).join("\n\n")}
                </div>
              )}
            </>
          ) : (
            <div style={{ color: "var(--text-tertiary)", fontFamily: "var(--font-jetbrains)", fontSize: "12px" }}>Starting pipeline…</div>
          )}
        </div>

        {/* Actions */}
        <div className="flex gap-3">
          {isDone && !hasFailed && (
            <Link href={`/runs/${runId}`} className="flex-1 rounded-lg flex items-center justify-center gap-2" style={{ background: "var(--gold)", color: "#08080B", padding: "11px 24px", fontSize: "13px", fontWeight: 600, fontFamily: "var(--font-syne)", letterSpacing: "0.05em", textDecoration: "none" }}>
              View Results
              <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
                <path d="M3 6.5H10M7.5 4L10 6.5L7.5 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </Link>
          )}
          <Link href={`/runs/${runId}`} className="rounded-lg flex items-center justify-center" style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", color: "var(--text-secondary)", padding: "11px 20px", fontSize: "13px", fontFamily: "var(--font-syne)", textDecoration: "none", flex: isDone && !hasFailed ? "0 0 auto" : "1" }}>
            {isDone ? "Open Run Detail" : "View Run Detail"}
          </Link>
        </div>

        {!isDone && (
          <div style={{ fontSize: "11px", color: "var(--text-tertiary)", fontFamily: "var(--font-jetbrains)", textAlign: "center" }}>
            Pipeline runs in the background — you can navigate away and return.
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function NewRunPage() {
  const [step, setStep] = useState(0);

  // Dataset selection
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);
  const [selectedDataset, setSelectedDataset] = useState<DatasetSummary | null>(null);
  const [branches, setBranches] = useState<DatasetBranch[]>([]);
  const [selectedBranch, setSelectedBranch] = useState<string | null>(null);
  const [versions, setVersions] = useState<DatasetVersion[]>([]);
  const [selectedVersion, setSelectedVersion] = useState<number | null>(null);

  // Config
  const [seed, setSeed] = useState(42);
  const [name, setName] = useState("");

  // Launch & run state
  const [phase, setPhase] = useState<"idle" | "launching" | "running" | "done">("idle");
  const [runId, setRunId] = useState<string | null>(null);
  const [run, setRun] = useState<RunDetail | null>(null);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);

  const activeRef  = useRef(true);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Cleanup on unmount
  useEffect(() => {
    activeRef.current = true;
    return () => {
      activeRef.current = false;
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

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

  // Load branches when dataset changes
  useEffect(() => {
    if (!selectedDataset) { setBranches([]); setSelectedBranch(null); return; }
    fetchDatasetBranches(selectedDataset.id)
      .then((bs) => { setBranches(bs); setSelectedBranch(null); })
      .catch(() => {});
  }, [selectedDataset]);

  // Load versions when dataset or branch changes
  useEffect(() => {
    if (!selectedDataset) { setVersions([]); setSelectedVersion(null); return; }
    const branchToFetch = selectedBranch ?? selectedDataset.default_branch_id ?? undefined;
    fetchDatasetVersions(selectedDataset.id, branchToFetch)
      .then((vs) => { setVersions(vs); setSelectedVersion(null); })
      .catch(() => {});
  }, [selectedDataset, selectedBranch]);

  const pollRun = useCallback((id: string) => {
    let failCount = 0;
    async function poll() {
      try {
        const detail = await fetchRunDetail(id);
        if (!activeRef.current) return;
        failCount = 0;
        setPollError(null);
        setRun(detail);
        if (allDone(detail)) {
          setPhase("done");
        } else {
          timeoutRef.current = setTimeout(poll, 2500);
        }
      } catch (e: unknown) {
        if (!activeRef.current) return;
        failCount++;
        if (failCount >= 3) {
          setPollError(`Cannot reach backend (${failCount} attempts): ${e instanceof Error ? e.message : String(e)}`);
        }
        timeoutRef.current = setTimeout(poll, 3000);
      }
    }
    poll();
  }, []);

  async function handleLaunch() {
    setPhase("launching");
    setLaunchError(null);
    setPollError(null);
    try {
      const created = await createRun({
        seed,
        dataset_id: selectedDataset?.id ?? null,
        branch_id: selectedBranch,
        dataset_version: selectedVersion,
        name: name || null,
      });
      await startStage(created.run_id, "run_full");
      setRunId(created.run_id);
      setPhase("running");
      pollRun(created.run_id);
    } catch (e: unknown) {
      setLaunchError(e instanceof Error ? e.message : String(e));
      setPhase("idle");
    }
  }

  // Show progress once launched
  if ((phase === "running" || phase === "done") && runId) {
    return <ProgressView runId={runId} run={run} phase={phase} pollError={pollError} />;
  }

  return (
    <div style={{ padding: "32px", maxWidth: "640px" }} className="animate-fade-up">
      {/* Page header */}
      <div style={{ marginBottom: "28px" }}>
        <div style={{ fontFamily: "var(--font-syne)", fontSize: "9.5px", fontWeight: 700, letterSpacing: "0.18em", textTransform: "uppercase", color: "var(--text-tertiary)", marginBottom: "6px" }}>
          Pipeline
        </div>
        <h1 style={{ fontFamily: "var(--font-syne)", fontWeight: 700, fontSize: "22px", color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
          Launch New Run
        </h1>
      </div>

      <StepIndicator step={step} />

      {step === 0 && (
        <SelectDatasetStep
          datasets={datasets}
          loading={datasetsLoading}
          error={datasetsError}
          selectedDataset={selectedDataset}
          onSelect={(ds) => { setSelectedDataset(ds); }}
          branches={branches}
          selectedBranch={selectedBranch}
          onSelectBranch={setSelectedBranch}
          versions={versions}
          selectedVersion={selectedVersion}
          onSelectVersion={setSelectedVersion}
          onNext={() => setStep(1)}
        />
      )}

      {step === 1 && (
        <ConfigureStep
          seed={seed}
          setSeed={setSeed}
          name={name}
          setName={setName}
          onBack={() => setStep(0)}
          onNext={() => setStep(2)}
        />
      )}

      {step === 2 && (
        <ReviewStep
          selectedDataset={selectedDataset}
          selectedBranch={selectedBranch}
          branches={branches}
          selectedVersion={selectedVersion}
          seed={seed}
          name={name}
          loading={phase === "launching"}
          error={launchError}
          onBack={() => setStep(1)}
          onLaunch={handleLaunch}
        />
      )}
    </div>
  );
}
