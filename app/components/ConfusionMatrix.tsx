"use client";

interface Props {
  matrix: number[][];
  classes: string[];
  title?: string;
}

export function ConfusionMatrix({ matrix, classes, title }: Props) {
  const maxVal = Math.max(...matrix.flat(), 1);

  return (
    <div>
      {title && (
        <div
          style={{
            fontFamily: "var(--font-syne)",
            fontSize: "9.5px",
            fontWeight: 700,
            letterSpacing: "0.14em",
            textTransform: "uppercase",
            color: "var(--text-tertiary)",
            marginBottom: "10px",
          }}
        >
          {title}
        </div>
      )}
      <div className="overflow-x-auto">
        <table
          style={{
            borderCollapse: "collapse",
            fontFamily: "var(--font-jetbrains)",
            fontSize: "11px",
          }}
        >
          <thead>
            <tr>
              <th
                style={{
                  padding: "4px 8px",
                  color: "var(--text-tertiary)",
                  textAlign: "right",
                  fontWeight: 400,
                  fontSize: "10px",
                }}
              >
                True ↓ / Pred →
              </th>
              {classes.map((c) => (
                <th
                  key={c}
                  style={{
                    padding: "4px 10px",
                    color: "var(--text-secondary)",
                    fontWeight: 500,
                    textAlign: "center",
                  }}
                >
                  {c}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, i) => (
              <tr key={classes[i]}>
                <td
                  style={{
                    padding: "4px 8px",
                    color: "var(--text-secondary)",
                    textAlign: "right",
                    fontWeight: 500,
                    whiteSpace: "nowrap",
                  }}
                >
                  {classes[i]}
                </td>
                {row.map((val, j) => {
                  const intensity = val / maxVal;
                  const isCorrect = i === j;
                  const bg = isCorrect
                    ? `rgba(52, 211, 153, ${0.08 + intensity * 0.55})`
                    : val > 0
                    ? `rgba(240, 112, 112, ${0.04 + intensity * 0.38})`
                    : "transparent";
                  const color = isCorrect
                    ? intensity > 0.5 ? "var(--success)" : "var(--text-secondary)"
                    : val > 0
                    ? intensity > 0.5 ? "var(--error)" : "var(--text-secondary)"
                    : "var(--text-tertiary)";
                  return (
                    <td
                      key={j}
                      style={{
                        padding: "5px 10px",
                        textAlign: "center",
                        background: bg,
                        color,
                        border: "1px solid var(--border-dim)",
                      }}
                    >
                      {val}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
        <p
          style={{
            fontSize: "10px",
            color: "var(--text-tertiary)",
            marginTop: "6px",
            fontFamily: "var(--font-jetbrains)",
          }}
        >
          Rows: true label · Columns: predicted
        </p>
      </div>
    </div>
  );
}
