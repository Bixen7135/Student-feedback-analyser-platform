"use client";

import { useMemo } from "react";
import type { Locale } from "@/app/lib/i18n/catalog";
import { useI18n } from "@/app/lib/i18n/provider";

type DateTimeInput = string | number | Date | null | undefined;

const INTL_LOCALE_BY_APP_LOCALE: Record<Locale, string> = {
  en: "en-US",
  ru: "ru-RU",
  kk: "kk-KZ",
};

function toValidDate(value: DateTimeInput): Date | null {
  if (value == null) {
    return null;
  }

  const date = value instanceof Date ? value : new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

export function useDateTimeLocale(): string {
  const { locale } = useI18n();

  return useMemo(() => INTL_LOCALE_BY_APP_LOCALE[locale], [locale]);
}

export function formatLocalizedDate(
  value: DateTimeInput,
  locale: string,
  options: Intl.DateTimeFormatOptions = {},
): string {
  const date = toValidDate(value);

  if (!date) {
    return "";
  }

  return new Intl.DateTimeFormat(locale, options).format(date);
}

export function formatLocalizedDateTime(
  value: DateTimeInput,
  locale: string,
  options: Intl.DateTimeFormatOptions = {},
): string {
  const date = toValidDate(value);

  if (!date) {
    return "";
  }

  return new Intl.DateTimeFormat(locale, options).format(date);
}
