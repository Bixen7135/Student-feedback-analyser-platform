"use client";
import { useEffect, useState } from "react";
import { use } from "react";
import Link from "next/link";
import { fetchPsychometricsMetrics, PsychometricsMetrics } from "@/app/lib/api";

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

const Card = ({ children }: { children: React.ReactNode }) => (
  <div
    className="rounded-xl"
    style={{ background: "var(--bg-surface)", border: "1px solid var(--border-dim)", padding: "20px 24px" }}
  >
    {children}
  </div>
);

const Th = ({ children, right }: { children: React.ReactNode; right?: boolean }) => (
  <th
    style={{
      textAlign: right ? "right" : "left",
      padding: "6px 0",
      borderBottom: "1px solid var(--border)",
      fontFamily: "var(--font-syne)",
      fontSize: "9px",
      fontWeight: 700,
      letterSpacing: "0.12em",
      textTransform: "uppercase" as const,
      color: "var(--text-tertiary)",
    }}
  >
    {children}
  </th>
);

const Td = ({ children, mono, right }: { children: React.ReactNode; mono?: boolean; right?: boolean }) => (
  <td
    style={{
      padding: "7px 0",
      borderBottom: "1px solid var(--border-dim)",
      fontSize: "12px",
      color: "var(--text-secondary)",
      textAlign: right ? "right" : "left",
      fontFamily: mono ? "var(--font-jetbrains)" : undefined,
    }}
  >
    {children}
  </td>
);

export default function PsychometricsPage({ params }: Props) {
  const { runId } = use(params);
  const [data, setData] = useState<PsychometricsMetrics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchPsychometricsMetrics(runId).then(setData).catch((e: Error) => setError(e.message));
  }, [runId]);

  return (
    <div className="page-shell page-standard page-shell--md animate-fade-up space-y-6">
      {/* Breadcrumb */}
      <div
        className="flex items-center gap-2"
        style={{ fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--text-tertiary)" }}
      >
        <Link href="/runs" style={{ color: "var(--text-tertiary)", textDecoration: "none" }}>Runs</Link>
        <span>›</span>
        <Link href={`/runs/${runId}`} style={{ color: "var(--text-tertiary)", textDecoration: "none" }}>{runId}</Link>
        <span>›</span>
        <span style={{ color: "var(--text-secondary)" }}>Psychometrics</span>
      </div>

      {/* Heading */}
      <div>
        <SectionLabel>Ordinal CFA</SectionLabel>
        <h1 style={{ fontFamily: "var(--font-syne)", fontWeight: 700, fontSize: "22px", color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
          Psychometrics
        </h1>
      </div>

      {/* Note */}
      <div
        style={{
          background: "var(--gold-faint)",
          borderLeft: "3px solid var(--gold-muted)",
          borderRadius: "0 6px 6px 0",
          padding: "9px 14px",
          fontSize: "11px",
          color: "var(--text-secondary)",
          fontFamily: "var(--font-jetbrains)",
        }}
      >
        Latent factors are derived exclusively from survey items. Text is never used.
      </div>

      {error && (
        <div className="rounded-lg" style={{ background: "var(--error-dim)", border: "1px solid var(--error)", padding: "12px 16px", color: "var(--error)", fontSize: "12px", fontFamily: "var(--font-jetbrains)" }}>
          {error}
        </div>
      )}

      {!data && !error && (
        <div style={{ color: "var(--text-tertiary)", fontFamily: "var(--font-jetbrains)", fontSize: "12px" }}>Loading psychometrics…</div>
      )}

      {data && (
        <>
          {/* Method badge */}
          <div
            className="inline-flex items-center gap-2"
            style={{
              background: "var(--bg-elevated)",
              border: "1px solid var(--border)",
              borderRadius: "6px",
              padding: "6px 12px",
              fontFamily: "var(--font-jetbrains)",
              fontSize: "11px",
              color: "var(--text-secondary)",
            }}
          >
            <span style={{ color: "var(--gold)" }}>method</span>
            <span>{data.method}</span>
            <span style={{ color: "var(--border-strong)" }}>·</span>
            <span style={{ color: "var(--gold)" }}>N</span>
            <span>{data.n_obs.toLocaleString()}</span>
          </div>

          {/* Fit statistics */}
          <Card>
            <SectionLabel>Fit Statistics</SectionLabel>
            <div className="table-scroll">
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <Th>Statistic</Th>
                  <Th right>Value</Th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.fit_statistics).map(([k, v]) => (
                  <tr key={k}>
                    <Td>{k}</Td>
                    <Td mono right>{typeof v === "number" ? v.toFixed(4) : String(v)}</Td>
                  </tr>
                ))}
              </tbody>
              </table>
            </div>
          </Card>

          {/* Reliability */}
          <Card>
            <SectionLabel>Reliability</SectionLabel>
            <div className="table-scroll">
              <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  <Th>Factor</Th>
                  <Th right>Cronbach α</Th>
                  <Th right>McDonald ω</Th>
                </tr>
              </thead>
              <tbody>
                {data.factor_names.map((f) => (
                  <tr key={f}>
                    <Td>{f}</Td>
                    <Td mono right>{data.reliability.cronbach_alpha?.[f]?.toFixed(4) ?? "N/A"}</Td>
                    <Td mono right>{data.reliability.mcdonald_omega?.[f]?.toFixed(4) ?? "N/A"}</Td>
                  </tr>
                ))}
              </tbody>
              </table>
            </div>
          </Card>

          {/* Item loadings */}
          {data.loadings && (
            <Card>
              <SectionLabel>Item Loadings</SectionLabel>
              <div className="overflow-x-auto">
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr>
                      <Th>Item</Th>
                      {data.factor_names.map((f) => <Th key={f} right>{f}</Th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(data.loadings).map(([item, factorLoadings]) => (
                      <tr key={item}>
                        <Td mono>{item}</Td>
                        {data.factor_names.map((f) => (
                          <Td key={f} mono right>
                            {factorLoadings[f] !== undefined ? factorLoadings[f].toFixed(3) : "—"}
                          </Td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </>
      )}
    </div>
  );
}
