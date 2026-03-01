"use client";

import { useEffect, useState } from "react";

import type { HistogramBin, NumericRange } from "@/app/components/charts/types";

export function Histogram({
  bins,
  activeRange,
  onRangeSelect,
  valueFormatter = (value) => value.toFixed(0),
}: {
  bins: HistogramBin[];
  activeRange?: NumericRange | null;
  onRangeSelect?: (range: NumericRange) => void;
  valueFormatter?: (value: number) => string;
}) {
  const [dragStart, setDragStart] = useState<number | null>(null);
  const [dragEnd, setDragEnd] = useState<number | null>(null);

  useEffect(() => {
    if (dragStart == null) return;
    function handleMouseUp() {
      if (dragStart == null || dragEnd == null || !onRangeSelect) {
        setDragStart(null);
        setDragEnd(null);
        return;
      }
      const startIndex = Math.min(dragStart, dragEnd);
      const endIndex = Math.max(dragStart, dragEnd);
      const startBin = bins[startIndex];
      const endBin = bins[endIndex];
      if (startBin && endBin) {
        onRangeSelect({ min: startBin.start, max: endBin.end });
      }
      setDragStart(null);
      setDragEnd(null);
    }

    window.addEventListener("mouseup", handleMouseUp);
    return () => window.removeEventListener("mouseup", handleMouseUp);
  }, [bins, dragEnd, dragStart, onRangeSelect]);

  if (bins.length === 0) {
    return (
      <div style={{ fontSize: "12px", color: "var(--text-tertiary)" }}>
        No numeric data available.
      </div>
    );
  }

  const maxCount = Math.max(...bins.map((bin) => bin.count), 1);
  const previewStart = dragStart == null || dragEnd == null ? null : Math.min(dragStart, dragEnd);
  const previewEnd = dragStart == null || dragEnd == null ? null : Math.max(dragStart, dragEnd);
  const chartMinWidth = `${Math.max(bins.length * 2.75, 16)}rem`;

  return (
    <div className="overflow-x-auto">
      <div style={{ minWidth: chartMinWidth }}>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: `repeat(${bins.length}, minmax(0, 1fr))`,
            gap: "6px",
            alignItems: "end",
            minHeight: "144px",
          }}
        >
          {bins.map((bin, index) => {
            const isPreview =
              previewStart != null &&
              previewEnd != null &&
              index >= previewStart &&
              index <= previewEnd;
            const isActive =
              activeRange != null &&
              bin.start <= activeRange.max &&
              bin.end >= activeRange.min;
            const height = `${Math.max((bin.count / maxCount) * 100, bin.count > 0 ? 10 : 4)}%`;
            return (
              <button
                key={`${bin.start}-${bin.end}-${index}`}
                type="button"
                onMouseDown={() => {
                  setDragStart(index);
                  setDragEnd(index);
                }}
                onMouseEnter={() => {
                  if (dragStart != null) {
                    setDragEnd(index);
                  }
                }}
                onClick={() => onRangeSelect?.({ min: bin.start, max: bin.end })}
                title={`${valueFormatter(bin.start)} - ${valueFormatter(bin.end)} (${bin.count})`}
                style={{
                  border: "none",
                  padding: 0,
                  background: "transparent",
                  cursor: onRangeSelect ? "pointer" : "default",
                }}
              >
                <div
                  style={{
                    height: "116px",
                    display: "flex",
                    alignItems: "end",
                  }}
                >
                  <span
                    style={{
                      display: "block",
                      width: "100%",
                      height,
                      borderRadius: "8px 8px 2px 2px",
                      background: isPreview || isActive ? "var(--gold)" : "rgba(201,155,64,0.55)",
                      border: "1px solid rgba(201,155,64,0.25)",
                    }}
                  />
                </div>
                <span
                  style={{
                    display: "block",
                    marginTop: "8px",
                    fontSize: "10px",
                    lineHeight: 1.3,
                    color: "var(--text-tertiary)",
                    fontFamily: "var(--font-jetbrains)",
                  }}
                >
                  {bin.label ?? valueFormatter(bin.start)}
                </span>
              </button>
            );
          })}
        </div>
      </div>
      {onRangeSelect && (
        <p
          style={{
            margin: "10px 0 0",
            fontSize: "10px",
            color: "var(--text-tertiary)",
          }}
        >
          Click a bin or drag across multiple bins to filter this range.
        </p>
      )}
    </div>
  );
}
