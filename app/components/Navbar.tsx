"use client";

import Link from "next/link";
import { useI18n } from "@/app/lib/i18n/provider";

export function Navbar() {
  const { t } = useI18n();

  return (
    <nav className="app-navbar">
      <Link href="/" className="app-navbar__brand">
        SFAP
      </Link>
      <Link href="/runs" className="app-navbar__link">
        {t("Run History")}
      </Link>
      <Link href="/runs/new" className="app-navbar__link">
        {t("Run Pipeline")}
      </Link>
      <span className="app-navbar__meta">{t("Student Feedback Analysis Platform v0.1")}</span>
    </nav>
  );
}
