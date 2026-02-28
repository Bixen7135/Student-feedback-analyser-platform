"use client";
import { useEffect, useRef, useState } from "react";
import { use } from "react";
import Link from "next/link";
import { fetchRunDetail, RunDetail } from "@/app/lib/api";
import { StageProgress } from "@/app/components/StageProgress";
import { Disclaimer } from "@/app/components/Disclaimer";

// ---------------------------------------------------------------------------
// Stage metadata
// ---------------------------------------------------------------------------

const STAGE_NAMES = [
  "ingest_preprocess",
  "psychometrics",
  "splits",
  "text_tasks",
  "fusion",
  "contradiction",
  "evaluation",
  "reporting",
] as const;

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

/** Rough median wall-clock seconds per stage, used for ETA estimates. */
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

/** Returns the right-hand ETA label for a stage row. */
function stageEtaLabel(
  name: string,
  status: string,
  startedAt: string | null,
  durationSeconds: number | null,
  nowMs: number
): string {
  if (status === "completed") {
    return durationSeconds != null ? `${durationSeconds.toFixed(1)}s` : "";
  }
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
  // pending
  const est = STAGE_ESTIMATES[name];
  return est != null ? `~${est}s est.` : "";
}

function isPipelineComplete(run: RunDetail): boolean {
  // All stages reached a terminal state
  if (STAGE_NAMES.every((s) => { const st = run.stages[s]; return st && TERMINAL.has(st.status); })) return true;
  // Any stage failed — pipeline terminated early, remaining stages will never run
  if (STAGE_NAMES.some((s) => run.stages[s]?.status === "failed")) return true;
  return false;
}

type PipelineStatus = "idle" | "running" | "complete" | "failed";

function pipelineStatus(run: RunDetail): PipelineStatus {
  const values = Object.values(run.stages);
  if (values.length === 0) return "idle";
  if (values.some((s) => s.status === "failed")) return "failed";
  if (values.some((s) => s.status === "running")) return "running";
  if (STAGE_NAMES.every((s) => run.stages[s]?.status === "completed")) return "complete";
  return "running"; // stages exist but not all done → still in progress
}

function dotColor(status: string): string {
  if (status === "completed") return "var(--success)";
  if (status === "failed")    return "var(--error)";
  if (status === "running")   return "var(--running)";
  return "var(--text-tertiary)";
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

interface Props {
  params: Promise<{ runId: string }>;
}

const NAV_LINKS = [
  { key: "psychometrics",  label: "Psychometrics",  desc: "CFA fit stats, loadings, reliability" },
  { key: "classification", label: "Classification", desc: "F1, accuracy, confusion matrices" },
  { key: "fusion",         label: "Fusion",         desc: "MAE, R², survey vs text deltas" },
  { key: "contradiction",  label: "Contradiction",  desc: "Monitoring rates by language & level" },
  { key: "reports",        label: "Reports",        desc: "Evaluation reports & model cards" },
];

const MetaRow = ({ label, value }: { label: string; value: React.ReactNode }) => (
  <div>
    <div style={{ fontFamily: "var(--font-syne)", fontSize: "9px", fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase" as const, color: "var(--text-tertiary)", marginBottom: "3px" }}>
      {label}
    </div>
    <div style={{ fontFamily: "var(--font-jetbrains)", fontSize: "12px", color: "var(--text-secondary)", wordBreak: "break-all" as const }}>
      {value}
    </div>
  </div>
);

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function RunDetailPage({ params }: Props) {
  const { runId } = use(params);
  const [run, setRun] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const activeRef  = useRef(true);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Tick every second to keep elapsed/remaining ETA live while pipeline is running
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    activeRef.current = true;

    async function poll() {
      try {
        const detail = await fetchRunDetail(runId);
        if (!activeRef.current) return;
        setRun(detail);
        if (!isPipelineComplete(detail)) {
          timeoutRef.current = setTimeout(poll, 2500);
        }
      } catch (e: unknown) {
        if (!activeRef.current) return;
        setError((e as Error).message);
      }
    }

    poll();

    return () => {
      activeRef.current = false;
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, [runId]);

  if (error)
    return (
      <div style={{ padding: "32px" }}>
        <div className="rounded-lg" style={{ background: "var(--error-dim)", border: "1px solid var(--error)", padding: "12px 16px", color: "var(--error)", fontSize: "12px", fontFamily: "var(--font-jetbrains)" }}>
          Error: {error}
        </div>
      </div>
    );

  if (!run)
    return (
      <div style={{ padding: "32px", color: "var(--text-tertiary)", fontFamily: "var(--font-jetbrains)", fontSize: "12px" }}>
        Loading run {runId}…
      </div>
    );

  const status   = pipelineStatus(run);
  const isActive = status === "running";
  const isDone   = status === "complete";
  const isFailed = status === "failed";

  const completedCount = STAGE_NAMES.filter((s) => TERMINAL.has(run.stages[s]?.status ?? "")).length;

  return (
    <div style={{ padding: "32px", maxWidth: "900px" }} className="animate-fade-up space-y-6">

      {/* Breadcrumb */}
      <div className="flex items-center gap-2" style={{ fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--text-tertiary)" }}>
        <Link href="/runs" style={{ color: "var(--text-tertiary)", textDecoration: "none" }}>Runs</Link>
        <span>›</span>
        <span style={{ color: "var(--text-secondary)" }}>{runId}</span>
      </div>

      {/* Heading + live banner */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <div style={{ fontFamily: "var(--font-syne)", fontSize: "9.5px", fontWeight: 700, letterSpacing: "0.18em", textTransform: "uppercase", color: "var(--text-tertiary)", marginBottom: "6px" }}>
            Run Detail
          </div>
          <h1 style={{ fontFamily: "var(--font-syne)", fontWeight: 700, fontSize: "22px", color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
            {runId}
          </h1>
        </div>

        {/* Status pill */}
        <div
          className="inline-flex items-center gap-1.5 rounded-full"
          style={{
            marginTop: "20px",
            padding: "4px 12px",
            fontFamily: "var(--font-jetbrains)",
            fontSize: "11px",
            background: isActive ? "var(--running-dim)" : isDone ? "var(--success-dim)" : isFailed ? "var(--error-dim)" : "var(--bg-elevated)",
            border: `1px solid ${isActive ? "var(--running)" : isDone ? "var(--success)" : isFailed ? "var(--error)" : "var(--border-dim)"}`,
            color: isActive ? "var(--running)" : isDone ? "var(--success)" : isFailed ? "var(--error)" : "var(--text-tertiary)",
          }}
        >
          <span
            className={`rounded-full shrink-0${isActive ? " animate-pulse-dot" : ""}`}
            style={{ width: "5px", height: "5px", background: "currentColor" }}
          />
          {isActive ? "running" : isDone ? "complete" : isFailed ? "failed" : "idle"}
        </div>
      </div>

      <Disclaimer />

      {/* Metadata */}
      <div className="rounded-xl" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-dim)", padding: "20px 24px" }}>
        <div style={{ fontFamily: "var(--font-syne)", fontSize: "9.5px", fontWeight: 700, letterSpacing: "0.16em", textTransform: "uppercase", color: "var(--text-tertiary)", marginBottom: "16px" }}>
          Metadata
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-5">
          <MetaRow label="Created"       value={new Date(run.created_at).toLocaleString()} />
          <MetaRow label="Random Seed"   value={String(run.random_seed)} />
          <MetaRow label="Config Hash"   value={run.config_hash} />
          <MetaRow label="Data Snapshot" value={run.data_snapshot_id} />
          <MetaRow label="Git Commit"    value={run.git_commit ?? "N/A"} />
          <MetaRow
            label="Produced Models"
            value={run.produced_models_count > 0 ? (
              <Link
                href={`/models?run_id=${encodeURIComponent(runId)}&include_archived=true`}
                style={{ color: "var(--gold)", textDecoration: "none" }}
              >
                {run.produced_models_count}
              </Link>
            ) : "0"}
          />
        </div>
      </div>

      {run.produced_models_count > 0 && (
        <div className="rounded-xl" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-dim)", padding: "20px 24px" }}>
          <div className="flex items-center justify-between gap-3" style={{ marginBottom: "14px", flexWrap: "wrap" }}>
            <div style={{ fontFamily: "var(--font-syne)", fontSize: "9.5px", fontWeight: 700, letterSpacing: "0.16em", textTransform: "uppercase", color: "var(--text-tertiary)" }}>
              Produced Models
            </div>
            <Link
              href={`/models?run_id=${encodeURIComponent(runId)}&include_archived=true`}
              style={{ fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--gold)", textDecoration: "none" }}
            >
              View all {run.produced_models_count}
            </Link>
          </div>

          <div className="flex flex-col gap-2">
            {run.produced_models_preview.map((model) => {
              const isArchived = model.status !== "active";

              return (
                <Link
                  key={model.id}
                  href={`/models/${model.id}`}
                  className="rounded-lg flex items-center justify-between gap-3"
                  style={{
                    background: "var(--bg-elevated)",
                    border: "1px solid var(--border-dim)",
                    padding: "10px 12px",
                    textDecoration: "none",
                  }}
                >
                  <div className="flex items-center gap-3" style={{ minWidth: 0, flexWrap: "wrap" }}>
                    <span style={{ fontFamily: "var(--font-syne)", fontSize: "12px", fontWeight: 600, color: "var(--text-primary)" }}>
                      {model.name}
                    </span>
                    <span style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)" }}>
                      {model.task} / {model.model_type} / v{model.version}
                    </span>
                  </div>
                  <div className="flex items-center gap-2" style={{ flexShrink: 0, flexWrap: "wrap", justifyContent: "flex-end" }}>
                    <span
                      className="rounded"
                      style={{
                        background: isArchived ? "var(--warning-dim)" : "var(--success-dim)",
                        border: `1px solid ${isArchived ? "var(--warning)" : "var(--success)"}`,
                        color: isArchived ? "var(--warning)" : "var(--success)",
                        padding: "2px 6px",
                        fontFamily: "var(--font-jetbrains)",
                        fontSize: "9px",
                      }}
                    >
                      {model.status}
                    </span>
                    <span style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)" }}>
                      {new Date(model.created_at).toLocaleDateString()}
                    </span>
                  </div>
                </Link>
              );
            })}
          </div>

          {run.produced_models_count > run.produced_models_preview.length && (
            <div style={{ marginTop: "12px", fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--text-tertiary)" }}>
              Showing {run.produced_models_preview.length} most recent models.
            </div>
          )}
        </div>
      )}

      {/* Stage progress — chips + detailed stepper */}
      <div className="rounded-xl" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-dim)", padding: "20px 24px" }}>

        {/* Header row */}
        <div className="flex items-center justify-between" style={{ marginBottom: "14px" }}>
          <div style={{ fontFamily: "var(--font-syne)", fontSize: "9.5px", fontWeight: 700, letterSpacing: "0.16em", textTransform: "uppercase", color: "var(--text-tertiary)" }}>
            Stage Progress
          </div>
          <div style={{ fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--text-tertiary)" }}>
            {completedCount} / {STAGE_NAMES.length}
          </div>
        </div>

        {/* Chip badges */}
        <StageProgress stages={run.stages} />

        {/* Detailed stepper */}
        <div style={{ marginTop: "18px", display: "flex", flexDirection: "column" }}>
          {STAGE_NAMES.map((name, i) => {
            const st     = run.stages[name];
            const status = st?.status ?? "pending";
            const isRunning = status === "running";
            const isActive  = isRunning || TERMINAL.has(status);

            return (
              <div
                key={name}
                className="flex items-center justify-between"
                style={{
                  padding: "7px 0",
                  borderBottom: i < STAGE_NAMES.length - 1 ? "1px solid var(--border-dim)" : "none",
                }}
              >
                <div className="flex items-center gap-2.5">
                  <span
                    className={`rounded-full shrink-0${isRunning ? " animate-pulse-dot" : ""}`}
                    style={{ width: "6px", height: "6px", background: dotColor(status) }}
                  />
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

        {/* Error details if any stage failed */}
        {isFailed && (
          <div className="rounded-lg" style={{ background: "var(--error-dim)", border: "1px solid var(--error)", padding: "10px 14px", marginTop: "14px", color: "var(--error)", fontSize: "11px", fontFamily: "var(--font-jetbrains)", whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
            {STAGE_NAMES.map((n) => run.stages[n]).filter((s) => s?.status === "failed" && s.error).map((s) => s!.error).join("\n\n")}
          </div>
        )}

        {/* Live indicator */}
        {isActive && (
          <div style={{ marginTop: "14px", fontSize: "11px", color: "var(--text-tertiary)", fontFamily: "var(--font-jetbrains)" }}>
            Updating every 2.5 s — pipeline running in background.
          </div>
        )}
      </div>

      {/* Navigation to results */}
      <div>
        <div style={{ fontFamily: "var(--font-syne)", fontSize: "9.5px", fontWeight: 700, letterSpacing: "0.16em", textTransform: "uppercase", color: "var(--text-tertiary)", marginBottom: "14px" }}>
          Results
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {NAV_LINKS.map(({ key, label, desc }) => (
            <Link
              key={key}
              href={`/runs/${runId}/${key}`}
              className="flex items-start gap-4 rounded-xl transition-all duration-150"
              style={{ background: "var(--bg-surface)", border: "1px solid var(--border-dim)", padding: "16px 20px", textDecoration: "none" }}
            >
              <div
                className="flex items-center justify-center rounded-lg shrink-0"
                style={{ width: "34px", height: "34px", background: "var(--gold-faint)", border: "1px solid var(--gold-muted)", color: "var(--gold)", marginTop: "1px", fontFamily: "var(--font-jetbrains)", fontSize: "10px" }}
              >
                {key.slice(0, 2).toUpperCase()}
              </div>
              <div>
                <div style={{ fontFamily: "var(--font-syne)", fontWeight: 600, fontSize: "13px", color: "var(--text-primary)", marginBottom: "3px" }}>
                  {label}
                </div>
                <div style={{ fontSize: "11px", color: "var(--text-tertiary)" }}>{desc}</div>
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
