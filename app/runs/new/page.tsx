"use client";
import { useState, useEffect, useRef, useCallback } from "react";
import Link from "next/link";
import { createRun, startStage, fetchRunDetail, RunDetail } from "@/app/lib/api";
import { Disclaimer } from "@/app/components/Disclaimer";
import { StageProgress } from "@/app/components/StageProgress";

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

function allDone(run: RunDetail): boolean {
  // All 8 stages reached a terminal state
  if (STAGE_NAMES.every((s) => { const st = run.stages[s]; return st && TERMINAL.has(st.status); })) return true;
  // Any stage failed — pipeline terminated early, remaining stages will never run
  if (STAGE_NAMES.some((s) => run.stages[s]?.status === "failed")) return true;
  return false;
}

// ─── Config form ─────────────────────────────────────────────────────────────

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

function ConfigForm({
  seed,
  setSeed,
  loading,
  error,
  onLaunch,
}: {
  seed: number;
  setSeed: (v: number) => void;
  loading: boolean;
  error: string | null;
  onLaunch: () => void;
}) {
  return (
    <div style={{ padding: "32px", maxWidth: "640px" }} className="animate-fade-up">
      {/* Header */}
      <div style={{ marginBottom: "28px" }}>
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
          Pipeline
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
          Launch New Run
        </h1>
      </div>

      <div className="space-y-5">
        <Disclaimer />

        {/* Config card */}
        <div
          className="rounded-xl"
          style={{
            background: "var(--bg-surface)",
            border: "1px solid var(--border-dim)",
            padding: "24px",
          }}
        >
          {/* Seed input */}
          <div style={{ marginBottom: "20px" }}>
            <label
              style={{
                display: "block",
                fontFamily: "var(--font-syne)",
                fontSize: "9.5px",
                fontWeight: 700,
                letterSpacing: "0.14em",
                textTransform: "uppercase",
                color: "var(--text-tertiary)",
                marginBottom: "8px",
              }}
            >
              Random Seed
            </label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              style={{
                width: "120px",
                background: "var(--bg-elevated)",
                border: "1px solid var(--border)",
                borderRadius: "6px",
                padding: "7px 12px",
                fontFamily: "var(--font-jetbrains)",
                fontSize: "13px",
                color: "var(--text-primary)",
                outline: "none",
              }}
            />
            <div
              style={{
                fontSize: "11px",
                color: "var(--text-tertiary)",
                marginTop: "6px",
                fontFamily: "var(--font-jetbrains)",
              }}
            >
              Fixed seed ensures reproducibility across runs.
            </div>
          </div>

          {/* Pipeline steps */}
          <div>
            <div
              style={{
                fontFamily: "var(--font-syne)",
                fontSize: "9.5px",
                fontWeight: 700,
                letterSpacing: "0.14em",
                textTransform: "uppercase",
                color: "var(--text-tertiary)",
                marginBottom: "10px",
              }}
            >
              Pipeline Stages
            </div>
            <ol
              style={{
                listStyle: "none",
                padding: 0,
                margin: 0,
                display: "flex",
                flexDirection: "column",
                gap: "6px",
              }}
            >
              {PIPELINE_STEPS.map((step, i) => (
                <li
                  key={step}
                  className="flex items-start gap-3"
                  style={{ fontSize: "12px", color: "var(--text-secondary)" }}
                >
                  <span
                    style={{
                      fontFamily: "var(--font-jetbrains)",
                      fontSize: "10px",
                      color: "var(--gold)",
                      flexShrink: 0,
                      marginTop: "1px",
                      minWidth: "16px",
                    }}
                  >
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  {step}
                </li>
              ))}
            </ol>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div
            className="rounded-lg"
            style={{
              background: "var(--error-dim)",
              border: "1px solid var(--error)",
              padding: "12px 16px",
              color: "var(--error)",
              fontSize: "12px",
              fontFamily: "var(--font-jetbrains)",
            }}
          >
            {error}
          </div>
        )}

        {/* Launch button */}
        <button
          onClick={onLaunch}
          disabled={loading}
          className="w-full rounded-lg flex items-center justify-center gap-2 transition-all duration-150"
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
              <path
                d="M3 6.5H10M7.5 4L10 6.5L7.5 9"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          )}
        </button>
      </div>
    </div>
  );
}

// ─── Progress view ────────────────────────────────────────────────────────────

function dotColor(status: string): string {
  if (status === "completed") return "var(--success)";
  if (status === "failed")    return "var(--error)";
  if (status === "running")   return "var(--running)";
  return "var(--text-tertiary)";
}

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
  // Tick every second so elapsed/remaining updates in real-time while running
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (phase !== "running") return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [phase]);

  const completedCount = run
    ? STAGE_NAMES.filter((s) => TERMINAL.has(run.stages[s]?.status ?? "")).length
    : 0;
  const hasFailed = run
    ? STAGE_NAMES.some((s) => run.stages[s]?.status === "failed")
    : false;
  const isDone = phase === "done";

  const heading = isDone
    ? hasFailed
      ? "Run Encountered Errors"
      : "Pipeline Complete"
    : "Pipeline Running…";

  return (
    <div style={{ padding: "32px", maxWidth: "640px" }} className="animate-fade-up">
      {/* Header */}
      <div style={{ marginBottom: "28px" }}>
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
          Pipeline
        </div>
        <h1
          style={{
            fontFamily: "var(--font-syne)",
            fontWeight: 700,
            fontSize: "22px",
            color: hasFailed ? "var(--error)" : isDone ? "var(--success)" : "var(--text-primary)",
            letterSpacing: "-0.01em",
          }}
        >
          {heading}
        </h1>
      </div>

      <div className="space-y-4">
        {/* Run ID chip */}
        <div
          className="rounded-xl"
          style={{
            background: "var(--bg-surface)",
            border: "1px solid var(--border-dim)",
            padding: "14px 20px",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <div
            style={{
              fontFamily: "var(--font-syne)",
              fontSize: "9px",
              fontWeight: 700,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              color: "var(--text-tertiary)",
            }}
          >
            Run ID
          </div>
          <div
            style={{
              fontFamily: "var(--font-jetbrains)",
              fontSize: "11px",
              color: "var(--text-secondary)",
            }}
          >
            {runId}
          </div>
        </div>

        {/* Stage progress card */}
        <div
          className="rounded-xl"
          style={{
            background: "var(--bg-surface)",
            border: "1px solid var(--border-dim)",
            padding: "20px 24px",
          }}
        >
          {/* Header row */}
          <div
            className="flex items-center justify-between"
            style={{ marginBottom: "14px" }}
          >
            <div
              style={{
                fontFamily: "var(--font-syne)",
                fontSize: "9.5px",
                fontWeight: 700,
                letterSpacing: "0.16em",
                textTransform: "uppercase",
                color: "var(--text-tertiary)",
              }}
            >
              Stage Progress
            </div>
            <div
              style={{
                fontFamily: "var(--font-jetbrains)",
                fontSize: "11px",
                color: "var(--text-tertiary)",
              }}
            >
              {completedCount} / {STAGE_NAMES.length}
            </div>
          </div>

          {/* Poll error banner */}
          {pollError && !run && (
            <div
              className="rounded-lg"
              style={{
                background: "var(--error-dim)",
                border: "1px solid var(--error)",
                padding: "10px 14px",
                marginBottom: "12px",
                color: "var(--error)",
                fontSize: "11px",
                fontFamily: "var(--font-jetbrains)",
              }}
            >
              {pollError}
            </div>
          )}

          {run ? (
            <>
              <StageProgress stages={run.stages} />

              {/* Detailed stepper */}
              <div
                style={{
                  marginTop: "18px",
                  display: "flex",
                  flexDirection: "column",
                  gap: "0px",
                }}
              >
                {STAGE_NAMES.map((name, i) => {
                  const st = run.stages[name];
                  const status = st?.status ?? "pending";
                  const isRunning = status === "running";
                  const isActive = isRunning || TERMINAL.has(status);

                  return (
                    <div
                      key={name}
                      className="flex items-center justify-between"
                      style={{
                        padding: "7px 0",
                        borderBottom:
                          i < STAGE_NAMES.length - 1
                            ? "1px solid var(--border-dim)"
                            : "none",
                      }}
                    >
                      <div className="flex items-center gap-2.5">
                        <span
                          className={`rounded-full shrink-0${isRunning ? " animate-pulse-dot" : ""}`}
                          style={{
                            width: "6px",
                            height: "6px",
                            background: dotColor(status),
                          }}
                        />
                        <span
                          style={{
                            fontFamily: "var(--font-jetbrains)",
                            fontSize: "12px",
                            color: isActive
                              ? "var(--text-primary)"
                              : "var(--text-tertiary)",
                          }}
                        >
                          {STAGE_LABELS[name] ?? name}
                        </span>
                      </div>

                      <div
                        style={{
                          fontFamily: "var(--font-jetbrains)",
                          fontSize: "11px",
                          color:
                            status === "failed"
                              ? "var(--error)"
                              : status === "completed"
                              ? "var(--success)"
                              : "var(--text-tertiary)",
                        }}
                      >
                        {stageEtaLabel(name, status, st?.started_at ?? null, st?.duration_seconds ?? null, now)}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Error summary */}
              {hasFailed && (
                <div
                  className="rounded-lg"
                  style={{
                    background: "var(--error-dim)",
                    border: "1px solid var(--error)",
                    padding: "10px 14px",
                    marginTop: "14px",
                    color: "var(--error)",
                    fontSize: "11px",
                    fontFamily: "var(--font-jetbrains)",
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                  }}
                >
                  {STAGE_NAMES.map((n) => run.stages[n])
                    .filter((s) => s?.status === "failed" && s.error)
                    .map((s) => s!.error)
                    .join("\n\n")}
                </div>
              )}
            </>
          ) : (
            <div
              style={{
                color: "var(--text-tertiary)",
                fontFamily: "var(--font-jetbrains)",
                fontSize: "12px",
              }}
            >
              Starting pipeline…
            </div>
          )}
        </div>

        {/* Actions */}
        <div className="flex gap-3">
          {isDone && !hasFailed && (
            <Link
              href={`/runs/${runId}`}
              className="flex-1 rounded-lg flex items-center justify-center gap-2"
              style={{
                background: "var(--gold)",
                color: "#08080B",
                padding: "11px 24px",
                fontSize: "13px",
                fontWeight: 600,
                fontFamily: "var(--font-syne)",
                letterSpacing: "0.05em",
                textDecoration: "none",
              }}
            >
              View Results
              <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
                <path
                  d="M3 6.5H10M7.5 4L10 6.5L7.5 9"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </Link>
          )}
          <Link
            href={`/runs/${runId}`}
            className="rounded-lg flex items-center justify-center"
            style={{
              background: "var(--bg-surface)",
              border: "1px solid var(--border)",
              color: "var(--text-secondary)",
              padding: "11px 20px",
              fontSize: "13px",
              fontFamily: "var(--font-syne)",
              textDecoration: "none",
              flex: isDone && !hasFailed ? "0 0 auto" : "1",
            }}
          >
            {isDone ? "Open Run Detail" : "View Run Detail"}
          </Link>
        </div>

        {!isDone && (
          <div
            style={{
              fontSize: "11px",
              color: "var(--text-tertiary)",
              fontFamily: "var(--font-jetbrains)",
              textAlign: "center",
            }}
          >
            Pipeline runs in the background — you can navigate away and return.
          </div>
        )}
      </div>
    </div>
  );
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export default function NewRunPage() {
  const [seed, setSeed] = useState(42);
  const [phase, setPhase] = useState<"idle" | "launching" | "running" | "done">("idle");
  const [runId, setRunId] = useState<string | null>(null);
  const [run, setRun] = useState<RunDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pollError, setPollError] = useState<string | null>(null);

  const activeRef  = useRef(true);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    // Reset on every mount/remount (guards against React StrictMode double-invoke)
    activeRef.current = true;
    return () => {
      activeRef.current = false;
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

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
          setPollError(
            `Cannot reach backend (${failCount} attempts): ${e instanceof Error ? e.message : String(e)}`
          );
        }
        timeoutRef.current = setTimeout(poll, 3000);
      }
    }
    poll();
  }, []);

  async function handleLaunch() {
    setPhase("launching");
    setError(null);
    setPollError(null);
    try {
      const created = await createRun({ seed });
      await startStage(created.run_id, "run_full");
      setRunId(created.run_id);
      setPhase("running");
      pollRun(created.run_id);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e));
      setPhase("idle");
    }
  }

  if ((phase === "running" || phase === "done") && runId) {
    return <ProgressView runId={runId} run={run} phase={phase} pollError={pollError} />;
  }

  return (
    <ConfigForm
      seed={seed}
      setSeed={setSeed}
      loading={phase === "launching"}
      error={error}
      onLaunch={handleLaunch}
    />
  );
}
