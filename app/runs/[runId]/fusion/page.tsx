"use client";
import { useEffect, useState } from "react";
import { use } from "react";
import Link from "next/link";
import { fetchFusionMetrics, FusionMetrics } from "@/app/lib/api";

interface Props { params: Promise<{ runId: string }> }

const SectionLabel = ({ children }: { children: React.ReactNode }) => (
  <div style={{ fontFamily: "var(--font-syne)", fontSize: "9.5px", fontWeight: 700, letterSpacing: "0.16em", textTransform: "uppercase" as const, color: "var(--text-tertiary)", marginBottom: "14px" }}>
    {children}
  </div>
);

const Card = ({ children }: { children: React.ReactNode }) => (
  <div className="rounded-xl" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-dim)", padding: "20px 24px" }}>
    {children}
  </div>
);

const Th = ({ children, right }: { children: React.ReactNode; right?: boolean }) => (
  <th style={{ textAlign: right ? "right" : "left", padding: "6px 0", borderBottom: "1px solid var(--border)", fontFamily: "var(--font-syne)", fontSize: "9px", fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase" as const, color: "var(--text-tertiary)" }}>
    {children}
  </th>
);

const Td = ({ children, right, highlight }: { children: React.ReactNode; right?: boolean; highlight?: "good" | "bad" | null }) => (
  <td style={{ padding: "7px 0", borderBottom: "1px solid var(--border-dim)", fontSize: "12px", textAlign: right ? "right" : "left", fontFamily: "var(--font-jetbrains)", color: highlight === "good" ? "var(--success)" : highlight === "bad" ? "var(--error)" : "var(--text-secondary)" }}>
    {children}
  </td>
);

const MetricBlock = ({ label, mae, r2 }: { label: string; mae: number; r2: number }) => (
  <div className="rounded-lg" style={{ background: "var(--bg-elevated)", border: "1px solid var(--border-dim)", padding: "16px 18px" }}>
    <div style={{ fontFamily: "var(--font-syne)", fontSize: "10px", fontWeight: 700, letterSpacing: "0.1em", textTransform: "uppercase" as const, color: "var(--text-tertiary)", marginBottom: "12px" }}>
      {label}
    </div>
    <div className="grid grid-cols-2 gap-4">
      {[{ key: "MAE", value: mae.toFixed(4) }, { key: "R²", value: r2.toFixed(4) }].map(({ key, value }) => (
        <div key={key}>
          <div style={{ fontFamily: "var(--font-syne)", fontWeight: 700, fontSize: "20px", color: "var(--gold)", lineHeight: 1 }}>
            {value}
          </div>
          <div style={{ fontSize: "10px", color: "var(--text-tertiary)", marginTop: "4px", textTransform: "uppercase" as const, letterSpacing: "0.08em", fontFamily: "var(--font-syne)" }}>
            {key}
          </div>
        </div>
      ))}
    </div>
  </div>
);

export default function FusionPage({ params }: Props) {
  const { runId } = use(params);
  const [data, setData] = useState<FusionMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchFusionMetrics(runId).then(setData).catch((e: Error) => setError(e.message));
  }, [runId]);

  return (
    <div className="page-shell page-standard page-shell--md animate-fade-up space-y-6">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2" style={{ fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--text-tertiary)" }}>
        <Link href="/runs" style={{ color: "var(--text-tertiary)", textDecoration: "none" }}>Runs</Link>
        <span>›</span>
        <Link href={`/runs/${runId}`} style={{ color: "var(--text-tertiary)", textDecoration: "none" }}>{runId}</Link>
        <span>›</span>
        <span style={{ color: "var(--text-secondary)" }}>Fusion</span>
      </div>

      {/* Heading */}
      <div>
        <SectionLabel>Late Fusion</SectionLabel>
        <h1 style={{ fontFamily: "var(--font-syne)", fontWeight: 700, fontSize: "22px", color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
          Fusion Experiments
        </h1>
        <p style={{ fontSize: "11px", color: "var(--text-tertiary)", marginTop: "4px", fontFamily: "var(--font-jetbrains)" }}>
          Regression targets: psychometric factor scores · lower MAE is better
        </p>
      </div>

      {/* Null-result note */}
      <div style={{ background: "var(--gold-faint)", borderLeft: "3px solid var(--gold-muted)", borderRadius: "0 6px 6px 0", padding: "9px 14px", fontSize: "11px", color: "var(--text-secondary)", fontFamily: "var(--font-jetbrains)" }}>
        Null results reported without spin — text may not improve on survey-only baseline.
      </div>

      {error && (
        <div className="rounded-lg" style={{ background: "var(--error-dim)", border: "1px solid var(--error)", padding: "12px 16px", color: "var(--error)", fontSize: "12px", fontFamily: "var(--font-jetbrains)" }}>
          {error}
        </div>
      )}

      {!data && !error && (
        <div style={{ color: "var(--text-tertiary)", fontFamily: "var(--font-jetbrains)", fontSize: "12px" }}>Loading fusion metrics…</div>
      )}

      {data && (
        <>
          {/* Three baseline blocks */}
          <div>
            <SectionLabel>Overall Results</SectionLabel>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <MetricBlock label="Survey only"  mae={data.survey_only.mae}  r2={data.survey_only.r_squared} />
              <MetricBlock label="Text only"    mae={data.text_only.mae}    r2={data.text_only.r_squared} />
              <MetricBlock label="Late fusion"  mae={data.late_fusion.mae}  r2={data.late_fusion.r_squared} />
            </div>
          </div>

          {/* Delta table */}
          <Card>
            <SectionLabel>Δ vs Survey-only (Fusion − Survey-only)</SectionLabel>
            <p style={{ fontSize: "11px", color: "var(--text-tertiary)", marginBottom: "14px", fontFamily: "var(--font-jetbrains)" }}>
              Negative ΔMAE = improvement · Positive = worse than survey-only
            </p>
            <div className="table-scroll">
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <Th>Factor</Th>
                  <Th right>Δ MAE</Th>
                  <Th right>Δ R²</Th>
                </tr>
              </thead>
              <tbody>
                {data.factor_names.map((f) => {
                  const dmae = data.delta_mae[f] ?? 0;
                  const dr2  = data.delta_r2[f]  ?? 0;
                  return (
                    <tr key={f}>
                      <Td>{f}</Td>
                      <Td right highlight={dmae < 0 ? "good" : dmae > 0 ? "bad" : null}>
                        {dmae >= 0 ? "+" : ""}{dmae.toFixed(4)}
                      </Td>
                      <Td right highlight={dr2 > 0 ? "good" : dr2 < 0 ? "bad" : null}>
                        {dr2 >= 0 ? "+" : ""}{dr2.toFixed(4)}
                      </Td>
                    </tr>
                  );
                })}
              </tbody>
              </table>
            </div>
          </Card>
        </>
      )}
    </div>
  );
}
