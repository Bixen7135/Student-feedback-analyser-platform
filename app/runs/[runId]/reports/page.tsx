"use client";
import { useEffect, useState } from "react";
import { use } from "react";
import Link from "next/link";
import { fetchArtifacts, downloadReport, ArtifactItem } from "@/app/lib/api";

interface Props { params: Promise<{ runId: string }> }

const SectionLabel = ({ children }: { children: React.ReactNode }) => (
  <div style={{ fontFamily: "var(--font-syne)", fontSize: "9.5px", fontWeight: 700, letterSpacing: "0.16em", textTransform: "uppercase" as const, color: "var(--text-tertiary)", marginBottom: "14px" }}>
    {children}
  </div>
);

const TYPE_LABEL: Record<string, string> = {
  report: "Report",
  model_card: "Model Card",
  data: "Data",
  model: "Model",
  metrics: "Metrics",
  config: "Config",
  snapshot: "Snapshot",
};

export default function ReportsPage({ params }: Props) {
  const { runId } = use(params);
  const [artifacts, setArtifacts] = useState<ArtifactItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [downloading, setDownloading] = useState<string | null>(null);

  useEffect(() => {
    fetchArtifacts(runId).then(setArtifacts).catch((e: Error) => setError(e.message));
  }, [runId]);

  const reports = artifacts.filter((a) => a.type === "report" || a.type === "model_card");

  async function handleDownload(artifact: ArtifactItem) {
    setDownloading(artifact.name);
    try {
      const blob = await downloadReport(runId, artifact.path.split("/").pop() ?? artifact.name);
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = artifact.name + ".md";
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      alert(`Download failed: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setDownloading(null);
    }
  }

  return (
    <div className="page-shell page-standard page-shell--md animate-fade-up space-y-6">
      {/* Breadcrumb */}
      <div className="flex items-center gap-2" style={{ fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--text-tertiary)" }}>
        <Link href="/runs" style={{ color: "var(--text-tertiary)", textDecoration: "none" }}>Runs</Link>
        <span>›</span>
        <Link href={`/runs/${runId}`} style={{ color: "var(--text-tertiary)", textDecoration: "none" }}>{runId}</Link>
        <span>›</span>
        <span style={{ color: "var(--text-secondary)" }}>Reports</span>
      </div>

      {/* Heading */}
      <div>
        <SectionLabel>Artifacts</SectionLabel>
        <h1 style={{ fontFamily: "var(--font-syne)", fontWeight: 700, fontSize: "22px", color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
          Reports &amp; Model Cards
        </h1>
        <p style={{ fontSize: "11px", color: "var(--text-tertiary)", marginTop: "4px", fontFamily: "var(--font-jetbrains)" }}>
          Download evaluation report, model cards, and data dictionary
        </p>
      </div>

      {error && (
        <div className="rounded-lg" style={{ background: "var(--error-dim)", border: "1px solid var(--error)", padding: "12px 16px", color: "var(--error)", fontSize: "12px", fontFamily: "var(--font-jetbrains)" }}>
          {error}
        </div>
      )}

      {/* Downloadable reports */}
      {reports.length === 0 && !error && (
        <div className="rounded-xl text-center" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-dim)", padding: "48px 24px" }}>
          <div style={{ fontFamily: "var(--font-syne)", fontWeight: 700, fontSize: "15px", color: "var(--text-secondary)", marginBottom: "6px" }}>
            No reports found
          </div>
          <div style={{ fontSize: "12px", color: "var(--text-tertiary)" }}>
            Run the reporting stage to generate evaluation reports and model cards.
          </div>
        </div>
      )}

      {reports.length > 0 && (
        <div>
          <SectionLabel>Downloadable Reports</SectionLabel>
          <div className="flex flex-col gap-3">
            {reports.map((artifact) => (
              <div
                key={artifact.name}
                className="flex items-center justify-between rounded-xl"
                style={{ background: "var(--bg-surface)", border: "1px solid var(--border-dim)", padding: "16px 20px" }}
              >
                <div>
                  <div style={{ fontFamily: "var(--font-syne)", fontWeight: 600, fontSize: "13px", color: "var(--text-primary)", marginBottom: "3px" }}>
                    {artifact.name}
                  </div>
                  <div style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)" }}>
                    {TYPE_LABEL[artifact.type] ?? artifact.type}
                    &nbsp;·&nbsp;{artifact.stage}
                    {artifact.size_bytes ? `&nbsp;·&nbsp;${(artifact.size_bytes / 1024).toFixed(1)} KB` : ""}
                  </div>
                </div>
                <button
                  onClick={() => handleDownload(artifact)}
                  disabled={downloading === artifact.name}
                  style={{
                    background: downloading === artifact.name ? "var(--bg-overlay)" : "var(--gold-faint)",
                    border: `1px solid ${downloading === artifact.name ? "var(--border)" : "var(--gold-muted)"}`,
                    color: downloading === artifact.name ? "var(--text-tertiary)" : "var(--gold)",
                    borderRadius: "6px",
                    padding: "6px 14px",
                    fontSize: "12px",
                    fontFamily: "var(--font-syne)",
                    fontWeight: 600,
                    letterSpacing: "0.04em",
                    cursor: downloading === artifact.name ? "not-allowed" : "pointer",
                    whiteSpace: "nowrap",
                  }}
                >
                  {downloading === artifact.name ? "Downloading…" : "Download"}
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* All artifacts table */}
      {artifacts.length > 0 && (
        <div className="rounded-xl" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-dim)", padding: "20px 24px" }}>
          <SectionLabel>All Artifacts</SectionLabel>
          <div className="overflow-x-auto">
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr>
                  {["Name", "Type", "Stage", "Size"].map((h, i) => (
                    <th
                      key={h}
                      style={{
                        textAlign: i === 3 ? "right" : "left",
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
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {artifacts.map((a) => (
                  <tr key={a.path}>
                    <td style={{ padding: "7px 0", borderBottom: "1px solid var(--border-dim)", fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--text-secondary)" }}>
                      {a.name}
                    </td>
                    <td style={{ padding: "7px 0", borderBottom: "1px solid var(--border-dim)", fontSize: "11px", color: "var(--text-tertiary)" }}>
                      {TYPE_LABEL[a.type] ?? a.type}
                    </td>
                    <td style={{ padding: "7px 0", borderBottom: "1px solid var(--border-dim)", fontSize: "11px", color: "var(--text-tertiary)" }}>
                      {a.stage}
                    </td>
                    <td style={{ padding: "7px 0", borderBottom: "1px solid var(--border-dim)", textAlign: "right", fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--text-tertiary)" }}>
                      {a.size_bytes ? `${(a.size_bytes / 1024).toFixed(1)} KB` : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
