"use client";
import type { StageStatus } from "@/app/lib/api";

const STAGE_ORDER = [
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
  ingest_preprocess: "Ingest",
  psychometrics: "Psychometrics",
  splits: "Splits",
  text_tasks: "Text Tasks",
  fusion: "Fusion",
  contradiction: "Contradiction",
  evaluation: "Evaluation",
  reporting: "Reporting",
};

type StatusKey = "pending" | "running" | "completed" | "failed" | "skipped";

const STATUS_STYLES: Record<StatusKey, { bg: string; border: string; color: string; dot: string }> = {
  pending:   { bg: "var(--pending-dim)",  border: "var(--border-dim)",  color: "var(--text-tertiary)", dot: "var(--text-tertiary)" },
  running:   { bg: "var(--running-dim)",  border: "var(--running)",     color: "var(--running)",       dot: "var(--running)" },
  completed: { bg: "var(--success-dim)",  border: "var(--success)",     color: "var(--success)",       dot: "var(--success)" },
  failed:    { bg: "var(--error-dim)",    border: "var(--error)",       color: "var(--error)",         dot: "var(--error)" },
  skipped:   { bg: "var(--bg-elevated)", border: "var(--border-dim)",  color: "var(--text-tertiary)", dot: "var(--border-strong)" },
};

interface Props {
  stages: Record<string, StageStatus>;
}

export function StageProgress({ stages }: Props) {
  return (
    <div className="flex flex-wrap gap-2">
      {STAGE_ORDER.map((stage) => {
        const s = stages[stage];
        const status = (s?.status ?? "pending") as StatusKey;
        const style = STATUS_STYLES[status] ?? STATUS_STYLES.pending;
        const isRunning = status === "running";

        return (
          <div
            key={stage}
            className="flex items-center gap-1.5 rounded"
            title={s?.error ?? undefined}
            style={{
              background: style.bg,
              border: `1px solid ${style.border}`,
              color: style.color,
              padding: "4px 10px",
              fontFamily: "var(--font-jetbrains)",
              fontSize: "11px",
            }}
          >
            <span
              className={`rounded-full shrink-0 ${isRunning ? "animate-pulse-dot" : ""}`}
              style={{
                width: "5px",
                height: "5px",
                background: style.dot,
              }}
            />
            <span>{STAGE_LABELS[stage] ?? stage}</span>
            {s?.duration_seconds != null && (
              <span style={{ color: "var(--text-tertiary)", marginLeft: "2px" }}>
                {s.duration_seconds.toFixed(1)}s
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
