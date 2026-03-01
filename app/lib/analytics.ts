import type {
  AnalyticsSummary,
  AnalysisClusterResponse,
  AnalysisEmbeddingsResponse,
  ClassificationDiagnostics,
  CorrelationResult,
} from "@/app/lib/api";
import type {
  CategoricalDatum,
  HeatmapCell,
  HistogramBin,
  ScatterDatum,
} from "@/app/components/charts/types";

interface NumericSummaryLike {
  count?: number;
  min?: number;
  q25?: number;
  median?: number;
  q75?: number;
  max?: number;
}

interface TextSummaryLike {
  count?: number;
  char_length?: NumericSummaryLike;
  word_length?: NumericSummaryLike;
}

function num(value: unknown, fallback = 0): number {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

export function buildCategoricalSections(
  summary: AnalyticsSummary,
  maxColumns = 4,
  maxLevels = 8,
): Array<{ column: string; count: number; series: CategoricalDatum[] }> {
  return Object.entries(summary.categorical)
    .map(([column, stats]) => ({
      column,
      count: num(stats.count),
      series: Object.entries(stats.levels ?? {})
        .map(([label, level]) => ({
          label,
          value: num(level.count),
          secondaryLabel:
            typeof level.proportion === "number"
              ? `${(level.proportion * 100).toFixed(1)}%`
              : undefined,
        }))
        .slice(0, maxLevels),
    }))
    .filter((section) => section.series.length > 0)
    .sort((a, b) => b.count - a.count)
    .slice(0, maxColumns);
}

export function buildQuartileBins(
  stats: NumericSummaryLike | undefined,
): HistogramBin[] {
  if (!stats) return [];
  const count = Math.max(0, Math.round(num(stats.count)));
  if (count === 0) return [];

  const values = [
    num(stats.min),
    num(stats.q25),
    num(stats.median),
    num(stats.q75),
    num(stats.max),
  ];
  const unique = new Set(values);
  if (unique.size <= 1) {
    const only = values[0] ?? 0;
    return [{ start: only, end: only, count, label: `${only.toFixed(0)}` }];
  }

  const weights = [
    Math.round(count * 0.25),
    Math.round(count * 0.25),
    Math.round(count * 0.25),
    count - Math.round(count * 0.75),
  ];

  return [
    {
      start: values[0],
      end: values[1],
      count: weights[0],
      label: `${values[0].toFixed(0)}-${values[1].toFixed(0)}`,
    },
    {
      start: values[1],
      end: values[2],
      count: weights[1],
      label: `${values[1].toFixed(0)}-${values[2].toFixed(0)}`,
    },
    {
      start: values[2],
      end: values[3],
      count: weights[2],
      label: `${values[2].toFixed(0)}-${values[3].toFixed(0)}`,
    },
    {
      start: values[3],
      end: values[4],
      count: weights[3],
      label: `${values[3].toFixed(0)}-${values[4].toFixed(0)}`,
    },
  ];
}

export function buildTextLengthSections(
  summary: AnalyticsSummary,
  maxColumns = 4,
): Array<{ column: string; count: number; charBins: HistogramBin[]; wordBins: HistogramBin[] }> {
  return Object.entries(summary.text)
    .map(([column, stats]) => {
      const typed = stats as TextSummaryLike;
      return {
        column,
        count: num(typed.count),
        charBins: buildQuartileBins(typed.char_length),
        wordBins: buildQuartileBins(typed.word_length),
      };
    })
    .filter((section) => section.charBins.length > 0 || section.wordBins.length > 0)
    .sort((a, b) => b.count - a.count)
    .slice(0, maxColumns);
}

export function buildNumericSections(
  summary: AnalyticsSummary,
  maxColumns = 4,
): Array<{ column: string; count: number; bins: HistogramBin[] }> {
  return Object.entries(summary.numeric)
    .map(([column, stats]) => ({
      column,
      count: num((stats as NumericSummaryLike).count),
      bins: buildQuartileBins(stats as NumericSummaryLike),
    }))
    .filter((section) => section.bins.length > 0)
    .sort((a, b) => b.count - a.count)
    .slice(0, maxColumns);
}

export function buildCorrelationCells(correlations: CorrelationResult[]): HeatmapCell[] {
  const cells: HeatmapCell[] = [];
  for (const item of correlations) {
    cells.push({
      x: item.right,
      y: item.left,
      value: item.value,
      label: `${item.metric} (${item.n} rows)`,
    });
    cells.push({
      x: item.left,
      y: item.right,
      value: item.value,
      label: `${item.metric} (${item.n} rows)`,
    });
  }
  return cells;
}

export function buildConfusionCells(
  diagnostics: ClassificationDiagnostics,
): HeatmapCell[] {
  const cells: HeatmapCell[] = [];
  diagnostics.confusion_matrix.forEach((row, rowIndex) => {
    row.forEach((value, colIndex) => {
      const y = diagnostics.labels[rowIndex] ?? `row_${rowIndex}`;
      const x = diagnostics.labels[colIndex] ?? `col_${colIndex}`;
      cells.push({
        x,
        y,
        value,
        label: `True ${y} / Pred ${x}`,
      });
    });
  });
  return cells;
}

export function buildCalibrationBins(
  diagnostics: ClassificationDiagnostics,
): HistogramBin[] {
  return diagnostics.calibration.map((bin, index) => ({
    start: bin.range[0],
    end: bin.range[1],
    count: bin.count,
    label: `${index + 1}`,
  }));
}

export function buildEmbeddingPoints(
  embeddings: AnalysisEmbeddingsResponse,
): ScatterDatum[] {
  const labelKey =
    embeddings.columns.find((column) => column.endsWith("_pred")) ??
    embeddings.columns.find((column) => column === "sentiment_class") ??
    embeddings.columns.find((column) => column === "language") ??
    null;

  return embeddings.points.map((point) => ({
    id: point.row_idx,
    x: num(point.x),
    y: num(point.y),
    label: labelKey ? String(point[labelKey] ?? point.row_idx) : String(point.row_idx),
    group: labelKey ? String(point[labelKey] ?? "") : undefined,
  }));
}

export function buildClusterPoints(
  clusters: AnalysisClusterResponse,
): ScatterDatum[] {
  return clusters.clusters.map((point) => ({
    id: point.row_idx,
    x: num(point.x),
    y: num(point.y),
    label: String(point.row_idx),
    group: String(point.cluster),
  }));
}
