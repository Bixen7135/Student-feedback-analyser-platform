"use client";

import { useI18n } from "@/app/lib/i18n/provider";

export function Disclaimer() {
  const { t } = useI18n();

  return (
    <div className="ui-disclaimer">
      <div className="ui-disclaimer__eyebrow">{t("Scope & Limitations")}</div>
      <ul className="ui-disclaimer__list">
        {[
          "Aggregate reporting and quality monitoring only.",
          "Not for individual-level decisions about students or staff.",
          "No causal claims are made from model outputs.",
          "Latent quality factors are derived from survey items only - text is never used.",
        ].map((text) => (
          <li key={text} className="ui-disclaimer__item">
            <span className="ui-disclaimer__bullet">&gt;</span>
            {t(text)}
          </li>
        ))}
      </ul>
    </div>
  );
}
