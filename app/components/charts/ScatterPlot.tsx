"use client";

import type { ScatterDatum } from "@/app/components/charts/types";

const WIDTH = 520;
const HEIGHT = 320;
const PADDING = 24;

function scale(value: number, min: number, max: number, size: number): number {
  if (max <= min) return size / 2;
  return ((value - min) / (max - min)) * size;
}

export function ScatterPlot({
  points,
  selectedGroup,
  onSelectGroup,
}: {
  points: ScatterDatum[];
  selectedGroup?: string | null;
  onSelectGroup?: (group: string) => void;
}) {
  if (points.length === 0) {
    return (
      <div style={{ fontSize: "12px", color: "var(--text-tertiary)" }}>
        No points available.
      </div>
    );
  }

  const xMin = Math.min(...points.map((point) => point.x));
  const xMax = Math.max(...points.map((point) => point.x));
  const yMin = Math.min(...points.map((point) => point.y));
  const yMax = Math.max(...points.map((point) => point.y));
  const groups = Array.from(
    new Set(
      points
        .map((point) => point.group)
        .filter((group): group is string => Boolean(group)),
    ),
  );

  return (
    <div>
      <svg
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        style={{
          width: "100%",
          maxWidth: `${WIDTH}px`,
          borderRadius: "var(--radius-unified)",
          background:
            "radial-gradient(circle at top right, rgba(201,155,64,0.08), transparent 42%), var(--bg-base)",
          border: "1px solid var(--border-dim)",
        }}
      >
        {points.slice(0, 800).map((point) => {
          const group = point.group ?? "";
          const cx =
            PADDING + scale(point.x, xMin, xMax, WIDTH - PADDING * 2);
          const cy =
            HEIGHT - PADDING - scale(point.y, yMin, yMax, HEIGHT - PADDING * 2);
          const faded = selectedGroup && group && selectedGroup !== group;
          return (
            <circle
              key={point.id}
              cx={cx}
              cy={cy}
              r={3.5}
              fill={point.color ?? "var(--gold)"}
              fillOpacity={faded ? 0.15 : 0.72}
              stroke={selectedGroup && group === selectedGroup ? "var(--text-primary)" : "none"}
              strokeWidth={selectedGroup && group === selectedGroup ? 1.2 : 0}
            >
              <title>
                {point.label ?? String(point.id)}
                {group ? ` (${group})` : ""}
              </title>
            </circle>
          );
        })}
      </svg>
      {groups.length > 0 && (
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "8px",
            marginTop: "12px",
          }}
        >
          {groups.map((group) => {
            const active = selectedGroup === group;
            return (
              <button
                key={group}
                type="button"
                onClick={() => onSelectGroup?.(group)}
                style={{
                  padding: "5px 10px",
                  borderRadius: "999px",
                  border: "1px solid",
                  borderColor: active ? "var(--gold)" : "var(--border-dim)",
                  background: active ? "rgba(201,155,64,0.12)" : "transparent",
                  color: active ? "var(--gold)" : "var(--text-tertiary)",
                  fontSize: "11px",
                  fontFamily: "var(--font-jetbrains)",
                  cursor: onSelectGroup ? "pointer" : "default",
                }}
              >
                {group}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
