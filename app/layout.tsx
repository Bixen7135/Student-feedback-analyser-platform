import type { Metadata } from "next";
import { Syne, DM_Sans, JetBrains_Mono } from "next/font/google";
import { cookies, headers } from "next/headers";
import "./globals.css";
import { AppShell } from "@/app/components/AppShell";
import { isLocale, resolvePreferredLocale } from "@/app/lib/i18n/catalog";
import { LOCALE_COOKIE_KEY } from "@/app/lib/i18n/constants";

const syne = Syne({
  subsets: ["latin"],
  weight: ["400", "600", "700", "800"],
  variable: "--font-syne",
  display: "swap",
  preload: false,
});

const dmSans = DM_Sans({
  subsets: ["latin"],
  weight: ["300", "400", "500"],
  variable: "--font-dm-sans",
  display: "swap",
  preload: false,
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-jetbrains",
  display: "swap",
  preload: false,
});

export const metadata: Metadata = {
  title: "Feedback Lab",
  description: "Localized student feedback analysis platform",
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  const cookieStore = await cookies();
  const headerStore = await headers();
  const savedLocale = cookieStore.get(LOCALE_COOKIE_KEY)?.value;
  const initialLocale =
    savedLocale && isLocale(savedLocale)
      ? savedLocale
      : resolvePreferredLocale(headerStore.get("accept-language"));

  return (
    <html
      lang={initialLocale}
      data-locale={initialLocale}
      className={`${syne.variable} ${dmSans.variable} ${jetbrainsMono.variable}`}
    >
      <body className="app-frame">
        <AppShell initialLocale={initialLocale}>{children}</AppShell>
      </body>
    </html>
  );
}
