"use client";

import { use, useEffect, useState } from "react";
import Link from "next/link";

import {
  type DatasetCorrelationsResponse,
  type DatasetDescriptiveAnalytics,
  fetchDatasetCorrelations,
  fetchDatasetDescriptiveAnalytics,
} from "@/app/lib/api";
import {
  buildCategoricalSections,
  buildCorrelationCells,
  buildNumericSections,
  buildTextLengthSections,
} from "@/app/lib/analytics";
import { BarListChart } from "@/app/components/charts/BarListChart";
import { ChartCard } from "@/app/components/charts/ChartCard";
import { Heatmap } from "@/app/components/charts/Heatmap";
import { Histogram } from "@/app/components/charts/Histogram";

export default function DatasetAnalyticsPage({
  params,
}: {
  params: Promise<{ datasetId: string }>;
}) {
  const { datasetId } = use(params);
  const [descriptive, setDescriptive] = useState<DatasetDescriptiveAnalytics | null>(null);
  const [correlations, setCorrelations] = useState<DatasetCorrelationsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    Promise.all([
      fetchDatasetDescriptiveAnalytics(datasetId),
      fetchDatasetCorrelations(datasetId),
    ])
      .then(([a, b]) => {
        if (!active) return;
        setDescriptive(a);
        setCorrelations(b);
      })
      .catch((err: unknown) => {
        if (!active) return;
        setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      active = false;
    };
  }, [datasetId]);

  const categoricalSections = descriptive ? buildCategoricalSections(descriptive.summary, 4, 8) : [];
  const numericSections = descriptive ? buildNumericSections(descriptive.summary, 4) : [];
  const textSections = descriptive ? buildTextLengthSections(descriptive.summary, 4) : [];
  const correlationCells = correlations ? buildCorrelationCells(correlations.correlations) : [];

  return (
    <div className="page-shell page-standard page-shell--xl animate-fade-up">
      <div style={{ fontSize: "12px", color: "var(--text-tertiary)", marginBottom: "14px" }}>
        <Link href="/datasets" style={{ color: "inherit", textDecoration: "none" }}>
          Datasets
        </Link>
        {" / "}
        <Link href={`/datasets/${datasetId}`} style={{ color: "inherit", textDecoration: "none" }}>
          {datasetId}
        </Link>
        {" / "}
        <span style={{ color: "var(--text-secondary)" }}>Analytics</span>
      </div>

      <h1 style={{ margin: 0, fontFamily: "var(--font-syne)", fontSize: "24px", color: "var(--text-primary)" }}>
        Dataset Analytics
      </h1>
      <p style={{ margin: "6px 0 18px", fontSize: "12px", color: "var(--text-tertiary)" }}>
        Descriptive statistics and associations for the current dataset snapshot.
      </p>

      {error && (
        <div style={{ color: "var(--error, #ef4444)", fontSize: "12px", marginBottom: "12px" }}>
          {error}
        </div>
      )}

      {categoricalSections.length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "16px", marginBottom: "16px" }}>
          {categoricalSections.map((section) => (
            <ChartCard key={section.column} title={section.column} subtitle={`${section.count.toLocaleString()} non-empty rows`}>
              <BarListChart series={section.series} />
            </ChartCard>
          ))}
        </div>
      )}

      {(textSections.length > 0 || numericSections.length > 0) && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: "16px", marginBottom: "16px" }}>
          {textSections.map((section) => (
            <ChartCard key={`text-${section.column}`} title={`${section.column} text length`} subtitle="Quartile-derived distribution bins.">
              <Histogram bins={section.charBins} />
            </ChartCard>
          ))}
          {numericSections.map((section) => (
            <ChartCard key={`num-${section.column}`} title={`${section.column} distribution`} subtitle="Quartile-derived distribution bins.">
              <Histogram bins={section.bins} />
            </ChartCard>
          ))}
        </div>
      )}

      {correlationCells.length > 0 && (
        <ChartCard title="Correlation Heatmap" subtitle="Mixed-type pairwise associations within the dataset.">
          <Heatmap cells={correlationCells} />
        </ChartCard>
      )}
    </div>
  );
}
