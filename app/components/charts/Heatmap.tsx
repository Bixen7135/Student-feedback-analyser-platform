"use client";

import type { HeatmapCell } from "@/app/components/charts/types";

export function Heatmap({
  cells,
  onSelect,
}: {
  cells: HeatmapCell[];
  onSelect?: (cell: HeatmapCell) => void;
}) {
  if (cells.length === 0) {
    return (
      <div style={{ fontSize: "12px", color: "var(--text-tertiary)" }}>
        No matrix data available.
      </div>
    );
  }

  const xLabels = Array.from(new Set(cells.map((cell) => cell.x)));
  const yLabels = Array.from(new Set(cells.map((cell) => cell.y)));
  const maxValue = Math.max(...cells.map((cell) => Math.abs(cell.value)), 1);

  return (
    <div className="overflow-x-auto">
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `minmax(88px, 132px) repeat(${xLabels.length}, minmax(44px, 1fr))`,
          gap: "6px",
          alignItems: "center",
          minWidth: "max-content",
        }}
      >
        <div />
        {xLabels.map((label) => (
          <div
            key={label}
            style={{
              fontSize: "10px",
              color: "var(--text-tertiary)",
              textAlign: "center",
              fontFamily: "var(--font-jetbrains)",
            }}
          >
            {label}
          </div>
        ))}
        {yLabels.map((yLabel) => (
          <div
            key={yLabel}
            style={{ display: "contents" }}
          >
            <div
              style={{
                fontSize: "10px",
                color: "var(--text-secondary)",
                fontFamily: "var(--font-jetbrains)",
              }}
            >
              {yLabel}
            </div>
            {xLabels.map((xLabel) => {
              const cell =
                cells.find((item) => item.x === xLabel && item.y === yLabel) ??
                { x: xLabel, y: yLabel, value: 0 };
              const intensity = Math.min(Math.abs(cell.value) / maxValue, 1);
              const background =
                cell.value >= 0
                  ? `rgba(52, 211, 153, ${0.08 + intensity * 0.5})`
                  : `rgba(239, 68, 68, ${0.08 + intensity * 0.5})`;
              return (
                <button
                  key={`${xLabel}-${yLabel}`}
                  type="button"
                  onClick={() => onSelect?.(cell)}
                  title={cell.label ?? `${yLabel} / ${xLabel}: ${cell.value.toFixed(3)}`}
                  style={{
                    minHeight: "44px",
                    borderRadius: "var(--radius-unified)",
                    border: "1px solid var(--border-dim)",
                    background,
                    color: "var(--text-primary)",
                    fontSize: "11px",
                    fontFamily: "var(--font-jetbrains)",
                    cursor: onSelect ? "pointer" : "default",
                  }}
                >
                  {cell.value.toFixed(2)}
                </button>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
