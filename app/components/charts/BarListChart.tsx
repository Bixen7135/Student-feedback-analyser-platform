"use client";

import type { CategoricalDatum } from "@/app/components/charts/types";

export function BarListChart({
  title,
  series,
  activeLabel,
  onSelect,
  maxItems = 12,
}: {
  title?: string;
  series: CategoricalDatum[];
  activeLabel?: string | null;
  onSelect?: (label: string) => void;
  maxItems?: number;
}) {
  const entries = [...series]
    .sort((a, b) => b.value - a.value)
    .slice(0, maxItems);
  const total = entries.reduce((sum, item) => sum + item.value, 0);

  if (entries.length === 0) {
    return (
      <div style={{ fontSize: "12px", color: "var(--text-tertiary)" }}>
        No data available.
      </div>
    );
  }

  return (
    <div>
      {title && (
        <div
          style={{
            fontSize: "11px",
            color: "var(--text-tertiary)",
            textTransform: "uppercase",
            letterSpacing: "0.08em",
            marginBottom: "10px",
          }}
        >
          {title}
        </div>
      )}
      <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
        {entries.map((entry) => {
          const pct = total > 0 ? (entry.value / total) * 100 : 0;
          const isActive = activeLabel === entry.label;
          return (
            <button
              key={entry.label}
              type="button"
              onClick={() => onSelect?.(entry.label)}
              style={{
                display: "grid",
                gridTemplateColumns: "minmax(0, 1.15fr) minmax(0, 1.85fr) auto",
                gap: "clamp(8px, 1.4vw, 10px)",
                alignItems: "center",
                border: "none",
                background: "transparent",
                padding: 0,
                textAlign: "left",
                cursor: onSelect ? "pointer" : "default",
              }}
            >
              <span
                title={entry.label}
                style={{
                  fontSize: "11px",
                  color: isActive ? "var(--gold)" : "var(--text-secondary)",
                  fontFamily: "var(--font-jetbrains)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {entry.label || "(empty)"}
              </span>
              <span
                style={{
                  height: "14px",
                  borderRadius: "999px",
                  background: "var(--bg-base)",
                  border: "1px solid var(--border-dim)",
                  overflow: "hidden",
                  display: "block",
                }}
              >
                <span
                  style={{
                    display: "block",
                    width: `${pct}%`,
                    minWidth: pct > 0 ? "2px" : 0,
                    height: "100%",
                    borderRadius: "999px",
                    background: isActive ? "var(--gold)" : entry.color ?? "var(--gold)",
                    opacity: isActive ? 1 : 0.8,
                  }}
                />
              </span>
              <span
                style={{
                  fontSize: "11px",
                  color: "var(--text-tertiary)",
                  fontFamily: "var(--font-jetbrains)",
                  justifySelf: "end",
                  whiteSpace: "nowrap",
                }}
              >
                {entry.value.toLocaleString()} ({pct.toFixed(1)}%)
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
