"use client";

import { BarListChart } from "@/app/components/charts/BarListChart";
import type { CategoricalDatum } from "@/app/components/charts/types";

interface DistributionChartProps {
  column: string;
  data: Record<string, number>; // { value: count }
  maxBars?: number;
}

export function DistributionChart({ column, data, maxBars = 20 }: DistributionChartProps) {
  const series: CategoricalDatum[] = Object.entries(data)
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxBars)
    .map(([label, value]) => ({ label, value }));

  return <BarListChart title={column} series={series} />;
}
