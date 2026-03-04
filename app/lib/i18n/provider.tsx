"use client";

import {
  createContext,
  startTransition,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import {
  DEFAULT_LOCALE,
  type Locale,
  isLocale,
  resolvePreferredLocale,
  translateText,
} from "@/app/lib/i18n/catalog";
import { LOCALE_COOKIE_KEY, LOCALE_STORAGE_KEY } from "@/app/lib/i18n/constants";

type I18nContextValue = {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (value: string) => string;
};

const I18nContext = createContext<I18nContextValue | null>(null);

export function I18nProvider({
  children,
  initialLocale = DEFAULT_LOCALE,
}: {
  children: React.ReactNode;
  initialLocale?: Locale;
}) {
  const [locale, setLocaleState] = useState<Locale>(initialLocale);

  useEffect(() => {
    const storedLocale = window.localStorage.getItem(LOCALE_STORAGE_KEY);
    const preferredLocale =
      storedLocale && isLocale(storedLocale)
        ? storedLocale
        : resolvePreferredLocale(window.navigator.language);

    if (preferredLocale !== initialLocale) {
      startTransition(() => {
        setLocaleState(preferredLocale);
      });
    }
  }, [initialLocale]);

  useEffect(() => {
    document.documentElement.lang = locale;
    document.documentElement.dataset.locale = locale;
    window.localStorage.setItem(LOCALE_STORAGE_KEY, locale);
    document.cookie = `${LOCALE_COOKIE_KEY}=${locale}; path=/; max-age=31536000; samesite=lax`;
  }, [locale]);

  const setLocale = useCallback((nextLocale: Locale) => {
    startTransition(() => {
      setLocaleState(nextLocale);
    });
  }, []);

  const t = useCallback(
    (value: string) => translateText(value, locale),
    [locale],
  );

  const contextValue = useMemo<I18nContextValue>(
    () => ({
      locale,
      setLocale,
      t,
    }),
    [locale, setLocale, t],
  );

  return <I18nContext.Provider value={contextValue}>{children}</I18nContext.Provider>;
}

export function useI18n() {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error("useI18n must be used within I18nProvider.");
  }
  return context;
}
