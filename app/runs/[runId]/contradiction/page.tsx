"use client";
import { useEffect, useState } from "react";
import { use } from "react";
import Link from "next/link";
import { fetchContradictionMetrics, ContradictionMetrics } from "@/app/lib/api";

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

const BigStat = ({ value, label }: { value: string; label: string }) => (
  <div className="rounded-xl text-center" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-dim)", padding: "24px 16px" }}>
    <div style={{ fontFamily: "var(--font-syne)", fontWeight: 700, fontSize: "28px", color: "var(--gold)", lineHeight: 1 }}>
      {value}
    </div>
    <div style={{ fontSize: "10px", color: "var(--text-tertiary)", marginTop: "7px", textTransform: "uppercase" as const, letterSpacing: "0.1em", fontFamily: "var(--font-syne)" }}>
      {label}
    </div>
  </div>
);

const RateBar = ({ label, rate, scale = 5 }: { label: string; rate: number; scale?: number }) => (
  <div className="flex items-center gap-3">
    <span style={{ width: "64px", fontSize: "11px", color: "var(--text-secondary)", fontFamily: "var(--font-jetbrains)", flexShrink: 0 }}>
      {label}
    </span>
    <div style={{ flex: 1, background: "var(--bg-overlay)", borderRadius: "3px", height: "4px" }}>
      <div
        style={{
          width: `${Math.min(rate * 100 * scale, 100)}%`,
          height: "4px",
          borderRadius: "3px",
          background: "var(--warning)",
        }}
      />
    </div>
    <span style={{ width: "40px", textAlign: "right", fontSize: "11px", fontFamily: "var(--font-jetbrains)", color: "var(--text-secondary)" }}>
      {(rate * 100).toFixed(2)}%
    </span>
  </div>
);

export default function ContradictionPage({ params }: Props) {
  const { runId } = use(params);
  const [data, setData] = useState<ContradictionMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchContradictionMetrics(runId).then(setData).catch((e: Error) => setError(e.message));
  }, [runId]);

  return (
    <div className="page-shell page-standard page-shell--md animate-fade-up space-y-6">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2" style={{ fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--text-tertiary)" }}>
        <Link href="/runs" style={{ color: "var(--text-tertiary)", textDecoration: "none" }}>Runs</Link>
        <span>›</span>
        <Link href={`/runs/${runId}`} style={{ color: "var(--text-tertiary)", textDecoration: "none" }}>{runId}</Link>
        <span>›</span>
        <span style={{ color: "var(--text-secondary)" }}>Contradiction</span>
      </div>

      {/* Heading */}
      <div>
        <SectionLabel>Monitoring</SectionLabel>
        <h1 style={{ fontFamily: "var(--font-syne)", fontWeight: 700, fontSize: "22px", color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
          Contradiction Monitoring
        </h1>
      </div>

      {error && (
        <div className="rounded-lg" style={{ background: "var(--error-dim)", border: "1px solid var(--error)", padding: "12px 16px", color: "var(--error)", fontSize: "12px", fontFamily: "var(--font-jetbrains)" }}>
          {error}
        </div>
      )}

      {!data && !error && (
        <div style={{ color: "var(--text-tertiary)", fontFamily: "var(--font-jetbrains)", fontSize: "12px" }}>Loading…</div>
      )}

      {data && (
        <>
          {/* Disclaimer from backend */}
          <div style={{ background: "var(--gold-faint)", borderLeft: "3px solid var(--gold-muted)", borderRadius: "0 var(--radius-unified) var(--radius-unified) 0", padding: "9px 14px", fontSize: "11px", color: "var(--text-secondary)", fontFamily: "var(--font-jetbrains)" }}>
            <strong style={{ color: "var(--gold)" }}>Monitoring only.</strong> {data.disclaimer}
          </div>

          {/* Big stats */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <BigStat value={`${(data.overall_rate * 100).toFixed(2)}%`} label="Overall Rate" />
            <BigStat value={data.n_contradictions.toLocaleString()} label="Total Flagged" />
            <BigStat value={data.n_total.toLocaleString()} label="Total Records" />
          </div>

          {/* By type */}
          <Card>
            <SectionLabel>By Contradiction Type</SectionLabel>
            <div className="table-scroll">
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <th style={{ textAlign: "left", padding: "6px 0", borderBottom: "1px solid var(--border)", fontFamily: "var(--font-syne)", fontSize: "9px", fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase" as const, color: "var(--text-tertiary)" }}>Type</th>
                  <th style={{ textAlign: "right", padding: "6px 0", borderBottom: "1px solid var(--border)", fontFamily: "var(--font-syne)", fontSize: "9px", fontWeight: 700, letterSpacing: "0.12em", textTransform: "uppercase" as const, color: "var(--text-tertiary)" }}>Rate</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.by_type).map(([type, rate]) => (
                  <tr key={type}>
                    <td style={{ padding: "7px 0", borderBottom: "1px solid var(--border-dim)", fontSize: "12px", color: "var(--text-secondary)" }}>
                      {type.replace(/_/g, " ")}
                    </td>
                    <td style={{ padding: "7px 0", borderBottom: "1px solid var(--border-dim)", textAlign: "right", fontFamily: "var(--font-jetbrains)", fontSize: "12px", color: "var(--text-secondary)" }}>
                      {(rate * 100).toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
              </table>
            </div>
          </Card>

          {/* Stratified */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Card>
              <SectionLabel>By Language</SectionLabel>
              <div className="flex flex-col gap-2.5">
                {Object.entries(data.stratified_by_language).map(([lang, rate]) => (
                  <RateBar key={lang} label={lang} rate={rate} />
                ))}
              </div>
            </Card>
            <Card>
              <SectionLabel>By Detail Level</SectionLabel>
              <div className="flex flex-col gap-2.5">
                {Object.entries(data.stratified_by_detail_level).map(([level, rate]) => (
                  <RateBar key={level} label={level} rate={rate} />
                ))}
              </div>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
