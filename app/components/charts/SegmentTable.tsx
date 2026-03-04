"use client";

import { SegmentGroup } from "@/app/lib/api";

interface SegmentTableProps {
  groups: SegmentGroup[];
  groupBy: string;
  metricCol: string;
}

export function SegmentTable({ groups, groupBy, metricCol }: SegmentTableProps) {
  if (groups.length === 0) {
    return (
      <div style={{ fontSize: "12px", color: "var(--text-tertiary)" }}>
        No segment data available. Make sure <strong>{groupBy}</strong> is a categorical column and{" "}
        <strong>{metricCol}</strong> is numeric.
      </div>
    );
  }

  const thStyle: React.CSSProperties = {
    padding: "7px 12px",
    textAlign: "left",
    fontSize: "11px",
    fontFamily: "var(--font-jetbrains)",
    color: "var(--text-tertiary)",
    fontWeight: 500,
    borderBottom: "1px solid var(--border-dim)",
    whiteSpace: "nowrap",
  };

  const tdStyle: React.CSSProperties = {
    padding: "7px 12px",
    fontSize: "11px",
    fontFamily: "var(--font-jetbrains)",
    color: "var(--text-secondary)",
    borderBottom: "1px solid var(--border-dim)",
    whiteSpace: "nowrap",
  };

  return (
    <div>
      <div
        style={{
          fontSize: "12px",
          fontWeight: 600,
          color: "var(--text-secondary)",
          marginBottom: "10px",
        }}
      >
        <span style={{ fontFamily: "var(--font-jetbrains)" }}>{metricCol}</span>
        <span style={{ fontWeight: 400, color: "var(--text-tertiary)", marginLeft: "6px" }}>
          grouped by
        </span>
        <span style={{ fontFamily: "var(--font-jetbrains)", marginLeft: "6px" }}>{groupBy}</span>
      </div>
      <div
        style={{
          overflowX: "auto",
          borderRadius: "var(--radius-unified)",
          border: "1px solid var(--border-dim)",
        }}
      >
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ background: "var(--bg-surface)" }}>
              <th style={thStyle}>{groupBy}</th>
              <th style={{ ...thStyle, textAlign: "right" }}>count</th>
              <th style={{ ...thStyle, textAlign: "right" }}>mean</th>
              <th style={{ ...thStyle, textAlign: "right" }}>median</th>
              <th style={{ ...thStyle, textAlign: "right" }}>std</th>
            </tr>
          </thead>
          <tbody>
            {groups.map((g) => (
              <tr key={g.group}>
                <td
                  style={{ ...tdStyle, color: "var(--text-primary)", fontWeight: 500 }}
                >
                  {g.group === "" ? (
                    <em style={{ color: "var(--text-tertiary)" }}>(empty)</em>
                  ) : (
                    g.group
                  )}
                </td>
                <td style={{ ...tdStyle, textAlign: "right" }}>
                  {g.count.toLocaleString()}
                </td>
                <td style={{ ...tdStyle, textAlign: "right", color: "var(--gold)" }}>
                  {g.mean.toFixed(4)}
                </td>
                <td style={{ ...tdStyle, textAlign: "right" }}>{g.median.toFixed(4)}</td>
                <td style={{ ...tdStyle, textAlign: "right" }}>{g.std.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
