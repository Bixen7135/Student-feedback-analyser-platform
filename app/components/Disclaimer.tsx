export function Disclaimer() {
  return (
    <div className="ui-disclaimer">
      <div className="ui-disclaimer__eyebrow">Scope &amp; Limitations</div>
      <ul className="ui-disclaimer__list">
        {[
          "Aggregate reporting and quality monitoring only.",
          "Not for individual-level decisions about students or staff.",
          "No causal claims are made from model outputs.",
          "Latent quality factors are derived from survey items only - text is never used.",
        ].map((text) => (
          <li key={text} className="ui-disclaimer__item">
            <span className="ui-disclaimer__bullet">&gt;</span>
            {text}
          </li>
        ))}
      </ul>
    </div>
  );
}
