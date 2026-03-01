"use client";

export function ChartCard({
  title,
  subtitle,
  children,
  actions,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  actions?: React.ReactNode;
}) {
  return (
    <section
      style={{
        padding: "clamp(16px, 2vw, 20px)",
        borderRadius: "clamp(12px, 1.2vw, 14px)",
        border: "1px solid var(--border-dim)",
        background:
          "linear-gradient(180deg, rgba(201,155,64,0.05) 0%, rgba(255,255,255,0) 28%), var(--bg-surface)",
      }}
    >
      <div
        className="flex items-start justify-between gap-3"
        style={{ marginBottom: "14px", flexWrap: "wrap" }}
      >
        <div>
          <h3
            style={{
              margin: 0,
              fontFamily: "var(--font-syne)",
              fontSize: "15px",
              fontWeight: 700,
              color: "var(--text-primary)",
            }}
          >
            {title}
          </h3>
          {subtitle && (
            <p
              style={{
                margin: "4px 0 0",
                fontSize: "11px",
                color: "var(--text-tertiary)",
              }}
            >
              {subtitle}
            </p>
          )}
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}
