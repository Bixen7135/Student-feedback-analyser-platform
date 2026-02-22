export function Disclaimer() {
  return (
    <div
      style={{
        background: "var(--bg-elevated)",
        borderLeft: "3px solid var(--gold-muted)",
        borderRadius: "0 8px 8px 0",
        padding: "12px 16px",
      }}
    >
      <div
        style={{
          fontFamily: "var(--font-syne)",
          fontSize: "9.5px",
          fontWeight: 700,
          letterSpacing: "0.16em",
          textTransform: "uppercase",
          color: "var(--gold)",
          marginBottom: "8px",
        }}
      >
        Scope &amp; Limitations
      </div>
      <ul
        style={{
          listStyle: "none",
          padding: 0,
          margin: 0,
          display: "flex",
          flexDirection: "column",
          gap: "4px",
        }}
      >
        {[
          "Aggregate reporting and quality monitoring only.",
          "Not for individual-level decisions about students or staff.",
          "No causal claims are made from model outputs.",
          "Latent quality factors are derived from survey items only — text is never used.",
        ].map((text) => (
          <li
            key={text}
            className="flex items-start gap-2"
            style={{ fontSize: "12px", color: "var(--text-secondary)" }}
          >
            <span style={{ color: "var(--gold-muted)", flexShrink: 0, marginTop: "1px" }}>›</span>
            {text}
          </li>
        ))}
      </ul>
    </div>
  );
}
