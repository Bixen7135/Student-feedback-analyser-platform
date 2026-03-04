"use client";

import { Sidebar } from "@/app/components/Sidebar";
import { I18nRuntime } from "@/app/components/I18nRuntime";
import type { Locale } from "@/app/lib/i18n/catalog";
import { I18nProvider } from "@/app/lib/i18n/provider";

export function AppShell({
  children,
  initialLocale,
}: {
  children: React.ReactNode;
  initialLocale: Locale;
}) {
  return (
    <I18nProvider initialLocale={initialLocale}>
      <I18nRuntime />
      <Sidebar />
      <main className="app-main">{children}</main>
    </I18nProvider>
  );
}
