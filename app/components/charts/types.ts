export interface CategoricalDatum {
  label: string;
  value: number;
  secondaryLabel?: string;
  color?: string;
}

export interface HistogramBin {
  start: number;
  end: number;
  count: number;
  label?: string;
}

export interface ScatterDatum {
  id: string | number;
  x: number;
  y: number;
  label?: string;
  group?: string;
  color?: string;
}

export interface HeatmapCell {
  x: string;
  y: string;
  value: number;
  label?: string;
}

export interface HeatmapMatrix {
  xLabels: string[];
  yLabels: string[];
  values: number[][];
  cells?: HeatmapCell[];
}

export interface NumericRange {
  min: number;
  max: number;
}
