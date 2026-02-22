import Link from "next/link";
import { Disclaimer } from "@/app/components/Disclaimer";

const PIPELINE_STAGES = [
  { num: "01", label: "Ingest" },
  { num: "02", label: "Psychometrics" },
  { num: "03", label: "Splits" },
  { num: "04", label: "Text Tasks" },
  { num: "05", label: "Fusion" },
  { num: "06", label: "Contradiction" },
  { num: "07", label: "Evaluate" },
  { num: "08", label: "Report" },
];

const STATS = [
  { value: "18,476", label: "Responses" },
  { value: "3", label: "Latent Factors" },
  { value: "9", label: "Survey Items" },
  { value: "2", label: "Languages" },
];

const SectionLabel = ({ children }: { children: React.ReactNode }) => (
  <div
    style={{
      fontFamily: "var(--font-syne)",
      fontSize: "9.5px",
      fontWeight: 700,
      letterSpacing: "0.18em",
      textTransform: "uppercase",
      color: "var(--text-tertiary)",
      marginBottom: "16px",
    }}
  >
    {children}
  </div>
);

export default function DashboardPage() {
  return (
    <div style={{ padding: "32px", maxWidth: "900px" }} className="space-y-6 animate-fade-up">

      {/* ── Hero card ─────────────────────────────────── */}
      <div
        className="relative overflow-hidden rounded-xl"
        style={{
          background: "var(--bg-surface)",
          border: "1px solid var(--border-dim)",
        }}
      >
        {/* Dot grid */}
        <div className="dot-grid absolute inset-0" style={{ opacity: 0.5 }} aria-hidden="true" />
        {/* Gold top-edge accent */}
        <div
          className="absolute top-0 left-0 right-0"
          style={{
            height: "1px",
            background: "linear-gradient(90deg, var(--gold) 0%, transparent 55%)",
          }}
          aria-hidden="true"
        />

        <div className="relative" style={{ padding: "36px 36px 32px" }}>
          {/* Label */}
          <div
            className="inline-flex items-center gap-2"
            style={{
              fontFamily: "var(--font-jetbrains)",
              fontSize: "10px",
              color: "var(--gold)",
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              marginBottom: "16px",
            }}
          >
            <span
              className="rounded-full"
              style={{ width: "5px", height: "5px", background: "var(--gold)" }}
            />
            Multilingual Batch Analysis
          </div>

          {/* Heading */}
          <h1
            style={{
              fontFamily: "var(--font-syne)",
              fontWeight: 800,
              fontSize: "clamp(26px, 3.5vw, 42px)",
              lineHeight: 1.08,
              letterSpacing: "-0.02em",
              color: "var(--text-primary)",
              marginBottom: "14px",
            }}
          >
            Student Feedback
            <br />
            <span style={{ color: "var(--gold)" }}>Analysis Platform</span>
          </h1>

          <p
            style={{
              fontSize: "13px",
              color: "var(--text-secondary)",
              lineHeight: 1.65,
              maxWidth: "440px",
              marginBottom: "28px",
            }}
          >
            Ordinal CFA psychometrics · Supervised text classification ·
            Late-fusion experiments · Fully reproducible, batch-only pipeline
          </p>

          {/* Stats */}
          <div className="flex flex-wrap gap-8" style={{ marginBottom: "30px" }}>
            {STATS.map(({ value, label }) => (
              <div key={label}>
                <div
                  style={{
                    fontFamily: "var(--font-syne)",
                    fontWeight: 700,
                    fontSize: "26px",
                    color: "var(--gold)",
                    lineHeight: 1,
                  }}
                >
                  {value}
                </div>
                <div
                  style={{
                    fontSize: "10px",
                    color: "var(--text-tertiary)",
                    marginTop: "4px",
                    textTransform: "uppercase",
                    letterSpacing: "0.09em",
                    fontFamily: "var(--font-syne)",
                  }}
                >
                  {label}
                </div>
              </div>
            ))}
          </div>

          {/* CTAs */}
          <div className="flex flex-wrap gap-3">
            <Link
              href="/runs/new"
              className="inline-flex items-center gap-2 rounded-lg transition-all duration-150"
              style={{
                background: "var(--gold)",
                color: "#08080B",
                padding: "9px 20px",
                fontSize: "13px",
                fontWeight: 600,
                fontFamily: "var(--font-syne)",
                letterSpacing: "0.04em",
                textDecoration: "none",
              }}
            >
              Launch New Run
              <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
                <path d="M3 6.5H10M7.5 4L10 6.5L7.5 9" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </Link>
            <Link
              href="/runs"
              className="inline-flex items-center gap-2 rounded-lg transition-all duration-150"
              style={{
                background: "transparent",
                color: "var(--text-secondary)",
                padding: "9px 20px",
                fontSize: "13px",
                fontWeight: 400,
                border: "1px solid var(--border)",
                textDecoration: "none",
              }}
            >
              View Run History
            </Link>
          </div>
        </div>
      </div>

      {/* ── Pipeline stages ───────────────────────────── */}
      <div
        className="rounded-xl"
        style={{
          background: "var(--bg-surface)",
          border: "1px solid var(--border-dim)",
          padding: "24px 28px",
        }}
      >
        <SectionLabel>Pipeline Stages</SectionLabel>
        <div className="flex flex-wrap items-center" style={{ gap: "4px" }}>
          {PIPELINE_STAGES.map((stage, i) => (
            <div key={stage.num} className="flex items-center" style={{ gap: "4px" }}>
              <div
                className="flex flex-col items-center rounded-lg"
                style={{
                  background: "var(--bg-elevated)",
                  border: "1px solid var(--border-dim)",
                  padding: "8px 12px",
                  minWidth: "76px",
                }}
              >
                <span
                  style={{
                    fontFamily: "var(--font-jetbrains)",
                    fontSize: "9px",
                    color: "var(--gold)",
                    letterSpacing: "0.06em",
                    marginBottom: "3px",
                  }}
                >
                  {stage.num}
                </span>
                <span
                  style={{
                    fontSize: "11px",
                    color: "var(--text-secondary)",
                    textAlign: "center",
                    whiteSpace: "nowrap",
                  }}
                >
                  {stage.label}
                </span>
              </div>
              {i < PIPELINE_STAGES.length - 1 && (
                <span style={{ color: "var(--border-strong)", fontSize: "11px" }}>→</span>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* ── Quick start ───────────────────────────────── */}
      <div
        className="rounded-xl"
        style={{
          background: "var(--bg-surface)",
          border: "1px solid var(--border-dim)",
          padding: "24px 28px",
        }}
      >
        <SectionLabel>Quick Start</SectionLabel>
        <pre
          className="rounded-lg overflow-x-auto"
          style={{
            background: "var(--bg-elevated)",
            border: "1px solid var(--border)",
            padding: "16px 18px",
            fontFamily: "var(--font-jetbrains)",
            fontSize: "12px",
            color: "var(--teal)",
            lineHeight: 1.75,
          }}
        >{`# Run the full pipeline
cd backend
uv run python -m src.cli run-full --data ../mnt/data/dataset.csv

# Start the API server (required for this UI)
uv run uvicorn src.api.main:app --port 8000`}</pre>
      </div>

      {/* ── Disclaimer ────────────────────────────────── */}
      <Disclaimer />
    </div>
  );
}
