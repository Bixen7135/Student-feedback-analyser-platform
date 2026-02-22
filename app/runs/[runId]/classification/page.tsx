"use client";
import { useEffect, useState } from "react";
import { use } from "react";
import Link from "next/link";
import { fetchClassificationMetrics, ClassificationMetrics } from "@/app/lib/api";
import { ConfusionMatrix } from "@/app/components/ConfusionMatrix";

const TASKS = ["language", "sentiment", "detail_level"];

interface Props { params: Promise<{ runId: string }> }

const SectionLabel = ({ children }: { children: React.ReactNode }) => (
  <div
    style={{
      fontFamily: "var(--font-syne)",
      fontSize: "9.5px",
      fontWeight: 700,
      letterSpacing: "0.16em",
      textTransform: "uppercase" as const,
      color: "var(--text-tertiary)",
      marginBottom: "14px",
    }}
  >
    {children}
  </div>
);

export default function ClassificationPage({ params }: Props) {
  const { runId } = use(params);
  const [byTask, setByTask] = useState<Record<string, ClassificationMetrics[]>>({});
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [activeTask, setActiveTask] = useState("sentiment");

  useEffect(() => {
    TASKS.forEach((task) => {
      fetchClassificationMetrics(runId, task)
        .then((data) => setByTask((prev) => ({ ...prev, [task]: data })))
        .catch((e: Error) => setErrors((prev) => ({ ...prev, [task]: e.message })));
    });
  }, [runId]);

  const taskMetrics = byTask[activeTask] ?? [];

  return (
    <div style={{ padding: "32px", maxWidth: "900px" }} className="animate-fade-up space-y-6">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2" style={{ fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--text-tertiary)" }}>
        <Link href="/runs" style={{ color: "var(--text-tertiary)", textDecoration: "none" }}>Runs</Link>
        <span>›</span>
        <Link href={`/runs/${runId}`} style={{ color: "var(--text-tertiary)", textDecoration: "none" }}>{runId}</Link>
        <span>›</span>
        <span style={{ color: "var(--text-secondary)" }}>Classification</span>
      </div>

      {/* Heading */}
      <div>
        <SectionLabel>Text Models</SectionLabel>
        <h1 style={{ fontFamily: "var(--font-syne)", fontWeight: 700, fontSize: "22px", color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
          Text Classification
        </h1>
        <p style={{ fontSize: "11px", color: "var(--text-tertiary)", marginTop: "4px", fontFamily: "var(--font-jetbrains)" }}>
          All tasks trained independently — no shared gradients
        </p>
      </div>

      {/* Task tabs */}
      <div className="flex gap-2">
        {TASKS.map((t) => {
          const active = activeTask === t;
          return (
            <button
              key={t}
              onClick={() => setActiveTask(t)}
              style={{
                padding: "6px 14px",
                borderRadius: "6px",
                fontSize: "12px",
                fontWeight: active ? 600 : 400,
                fontFamily: "var(--font-syne)",
                letterSpacing: active ? "0.04em" : "0",
                background: active ? "var(--gold-faint)" : "var(--bg-elevated)",
                border: `1px solid ${active ? "var(--gold-muted)" : "var(--border)"}`,
                color: active ? "var(--gold)" : "var(--text-secondary)",
                cursor: "pointer",
                transition: "all 0.1s",
              }}
            >
              {t.replace("_", " ")}
            </button>
          );
        })}
      </div>

      {/* Task error */}
      {errors[activeTask] && (
        <div className="rounded-lg" style={{ background: "var(--error-dim)", border: "1px solid var(--error)", padding: "12px 16px", color: "var(--error)", fontSize: "12px", fontFamily: "var(--font-jetbrains)" }}>
          {errors[activeTask]}
        </div>
      )}

      {/* Loading */}
      {taskMetrics.length === 0 && !errors[activeTask] && (
        <div style={{ color: "var(--text-tertiary)", fontFamily: "var(--font-jetbrains)", fontSize: "12px" }}>Loading…</div>
      )}

      {/* Model cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {taskMetrics.map((m) => (
          <div
            key={m.model_type}
            className="rounded-xl space-y-5"
            style={{ background: "var(--bg-surface)", border: "1px solid var(--border-dim)", padding: "20px 22px" }}
          >
            {/* Model type header */}
            <div
              style={{
                fontFamily: "var(--font-syne)",
                fontWeight: 600,
                fontSize: "13px",
                color: "var(--text-primary)",
                paddingBottom: "12px",
                borderBottom: "1px solid var(--border-dim)",
              }}
            >
              {m.model_type}
            </div>

            {/* Primary metrics */}
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: "Macro F1", value: `${(m.macro_f1 * 100).toFixed(1)}%` },
                { label: "Accuracy",  value: `${(m.accuracy  * 100).toFixed(1)}%` },
              ].map(({ label, value }) => (
                <div
                  key={label}
                  className="rounded-lg text-center"
                  style={{ background: "var(--bg-elevated)", border: "1px solid var(--border-dim)", padding: "14px 8px" }}
                >
                  <div style={{ fontFamily: "var(--font-syne)", fontWeight: 700, fontSize: "22px", color: "var(--gold)", lineHeight: 1 }}>
                    {value}
                  </div>
                  <div style={{ fontSize: "10px", color: "var(--text-tertiary)", marginTop: "5px", textTransform: "uppercase", letterSpacing: "0.08em", fontFamily: "var(--font-syne)" }}>
                    {label}
                  </div>
                </div>
              ))}
            </div>

            {/* Per-class F1 bars */}
            <div>
              <div style={{ fontFamily: "var(--font-syne)", fontSize: "9px", fontWeight: 700, letterSpacing: "0.14em", textTransform: "uppercase", color: "var(--text-tertiary)", marginBottom: "10px" }}>
                Per-class F1
              </div>
              <div className="flex flex-col gap-2">
                {Object.entries(m.per_class_f1).map(([cls, f1]) => (
                  <div key={cls} className="flex items-center gap-2">
                    <span style={{ width: "72px", fontSize: "11px", color: "var(--text-secondary)", fontFamily: "var(--font-jetbrains)", flexShrink: 0 }}>
                      {cls}
                    </span>
                    <div style={{ flex: 1, background: "var(--bg-overlay)", borderRadius: "3px", height: "4px" }}>
                      <div
                        style={{
                          width: `${f1 * 100}%`,
                          height: "4px",
                          borderRadius: "3px",
                          background: `linear-gradient(90deg, var(--gold) 0%, var(--teal) 100%)`,
                        }}
                      />
                    </div>
                    <span style={{ width: "36px", textAlign: "right", fontSize: "11px", fontFamily: "var(--font-jetbrains)", color: "var(--text-secondary)" }}>
                      {(f1 * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Confusion matrix */}
            {m.confusion_matrix.length > 0 && (
              <div style={{ paddingTop: "4px" }}>
                <ConfusionMatrix matrix={m.confusion_matrix} classes={m.classes} title="Confusion Matrix" />
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
