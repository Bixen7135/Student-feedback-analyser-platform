"use client";

interface DistributionChartProps {
  column: string;
  data: Record<string, number>; // { value: count }
  maxBars?: number;
}

export function DistributionChart({ column, data, maxBars = 20 }: DistributionChartProps) {
  const entries = Object.entries(data)
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxBars);

  const total = entries.reduce((sum, [, v]) => sum + v, 0);

  if (entries.length === 0) {
    return (
      <div style={{ fontSize: "12px", color: "var(--text-tertiary)" }}>
        No data for <strong>{column}</strong>.
      </div>
    );
  }

  return (
    <div>
      <div
        style={{
          fontSize: "12px",
          fontWeight: 600,
          color: "var(--text-secondary)",
          marginBottom: "10px",
          fontFamily: "var(--font-jetbrains)",
        }}
      >
        {column}
        <span style={{ fontWeight: 400, color: "var(--text-tertiary)", marginLeft: "8px" }}>
          ({total.toLocaleString()} rows)
        </span>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: "5px" }}>
        {entries.map(([value, count]) => {
          const pct = total > 0 ? (count / total) * 100 : 0;
          return (
            <div
              key={value}
              style={{ display: "flex", alignItems: "center", gap: "8px" }}
            >
              <div
                style={{
                  width: "130px",
                  fontSize: "11px",
                  color: "var(--text-secondary)",
                  fontFamily: "var(--font-jetbrains)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                  flexShrink: 0,
                }}
                title={value}
              >
                {value === "" ? <em style={{ color: "var(--text-tertiary)" }}>(empty)</em> : value}
              </div>
              <div
                style={{
                  flex: 1,
                  height: "14px",
                  background: "var(--bg-base)",
                  borderRadius: "3px",
                  overflow: "hidden",
                  border: "1px solid var(--border-dim)",
                }}
              >
                <div
                  style={{
                    width: `${pct}%`,
                    height: "100%",
                    background: "var(--gold)",
                    borderRadius: "3px",
                    transition: "width 0.4s ease",
                    minWidth: pct > 0 ? "2px" : "0",
                  }}
                />
              </div>
              <div
                style={{
                  width: "80px",
                  fontSize: "11px",
                  color: "var(--text-tertiary)",
                  textAlign: "right",
                  fontFamily: "var(--font-jetbrains)",
                  flexShrink: 0,
                }}
              >
                {count.toLocaleString()} ({pct.toFixed(1)}%)
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
