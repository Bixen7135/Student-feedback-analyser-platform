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
import { useI18n } from "@/app/lib/i18n/provider";
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
  const { locale, t } = useI18n();
  const [descriptive, setDescriptive] = useState<DatasetDescriptiveAnalytics | null>(null);
  const [correlations, setCorrelations] = useState<DatasetCorrelationsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const numberFormatter = new Intl.NumberFormat(locale);

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
          {t("Datasets")}
        </Link>
        {" / "}
        <Link href={`/datasets/${datasetId}`} style={{ color: "inherit", textDecoration: "none" }}>
          {datasetId}
        </Link>
        {" / "}
        <span style={{ color: "var(--text-secondary)" }}>{t("Analytics")}</span>
      </div>

      <h1 style={{ margin: 0, fontFamily: "var(--font-syne)", fontSize: "24px", color: "var(--text-primary)" }}>
        {t("Dataset Analytics")}
      </h1>
      <p style={{ margin: "6px 0 18px", fontSize: "12px", color: "var(--text-tertiary)" }}>
        {t("Descriptive statistics and associations for the current dataset snapshot.")}
      </p>

      {error && (
        <div style={{ color: "var(--error, #ef4444)", fontSize: "12px", marginBottom: "12px" }}>
          {error}
        </div>
      )}

      {categoricalSections.length > 0 && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "16px", marginBottom: "16px" }}>
          {categoricalSections.map((section) => (
            <ChartCard
              key={section.column}
              title={section.column}
              subtitle={`${numberFormatter.format(section.count)} ${t("non-empty rows")}`}
              skipAutoI18n
            >
              <BarListChart series={section.series} />
            </ChartCard>
          ))}
        </div>
      )}

      {(textSections.length > 0 || numericSections.length > 0) && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: "16px", marginBottom: "16px" }}>
          {textSections.map((section) => (
            <ChartCard
              key={`text-${section.column}`}
              title={`${section.column} ${t("text length")}`}
              subtitle={t("Quartile-derived distribution bins.")}
              skipAutoI18n
            >
              <Histogram bins={section.charBins} />
            </ChartCard>
          ))}
          {numericSections.map((section) => (
            <ChartCard
              key={`num-${section.column}`}
              title={`${section.column} ${t("distribution")}`}
              subtitle={t("Quartile-derived distribution bins.")}
              skipAutoI18n
            >
              <Histogram bins={section.bins} />
            </ChartCard>
          ))}
        </div>
      )}

      {correlationCells.length > 0 && (
        <ChartCard
          title={t("Correlation Heatmap")}
          subtitle={t("Mixed-type pairwise associations within the dataset.")}
          skipAutoI18n
        >
          <Heatmap cells={correlationCells} />
        </ChartCard>
      )}
    </div>
  );
}
