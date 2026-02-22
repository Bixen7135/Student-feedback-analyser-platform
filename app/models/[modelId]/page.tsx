"use client";

import { useEffect, useState, use } from "react";
import { useRouter } from "next/navigation";
import {
  fetchModelDetail,
  fetchModelVersions,
  updateModelMetadata,
  ModelSummary,
} from "@/app/lib/api";

const TASK_COLORS: Record<string, string> = {
  language: "var(--teal)",
  sentiment: "var(--gold)",
  detail_level: "var(--running)",
};

export default function ModelDetailPage({
  params,
}: {
  params: Promise<{ modelId: string }>;
}) {
  const { modelId } = use(params);
  const router = useRouter();
  const [model, setModel] = useState<ModelSummary | null>(null);
  const [versions, setVersions] = useState<ModelSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"metrics" | "config" | "versions">("metrics");

  // Edit metadata state
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState("");
  const [metaSaving, setMetaSaving] = useState(false);

  useEffect(() => {
    fetchModelDetail(modelId)
      .then((m) => {
        setModel(m);
        setEditName(m.name);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  }, [modelId]);

  useEffect(() => {
    if (tab === "versions") {
      fetchModelVersions(modelId)
        .then(setVersions)
        .catch((e) => setError(e.message));
    }
  }, [tab, modelId]);

  // ------------------------------------------------------------------
  // Handlers
  // ------------------------------------------------------------------

  async function handleMetaSave() {
    setMetaSaving(true);
    try {
      const updated = await updateModelMetadata(modelId, { name: editName });
      setModel(updated);
      setEditing(false);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setMetaSaving(false);
    }
  }

  const btnBase = {
    borderRadius: "6px",
    padding: "5px 12px",
    fontSize: "11px",
    fontFamily: "var(--font-jetbrains)",
    cursor: "pointer" as const,
    border: "1px solid var(--border)",
  } as const;

  const inputStyle = {
    background: "var(--bg-elevated)",
    border: "1px solid var(--border)",
    borderRadius: "6px",
    padding: "6px 10px",
    color: "var(--text-primary)",
    fontSize: "12px",
    fontFamily: "var(--font-jetbrains)",
    width: "100%",
    boxSizing: "border-box" as const,
  } as const;

  if (loading) {
    return (
      <div
        style={{
          padding: "48px 32px",
          color: "var(--text-tertiary)",
          fontFamily: "var(--font-jetbrains)",
          fontSize: "12px",
        }}
      >
        Loading…
      </div>
    );
  }

  if (!model) {
    return (
      <div
        style={{
          padding: "48px 32px",
          color: "var(--error)",
          fontFamily: "var(--font-jetbrains)",
          fontSize: "12px",
        }}
      >
        Model not found.
      </div>
    );
  }

  const taskColor = TASK_COLORS[model.task] || "var(--text-secondary)";

  const tabStyle = (t: string) => ({
    background: tab === t ? "var(--bg-elevated)" : "transparent",
    border: tab === t ? "1px solid var(--border)" : "1px solid transparent",
    borderBottom: tab === t ? "1px solid var(--bg-elevated)" : "1px solid transparent",
    borderRadius: "6px 6px 0 0",
    padding: "6px 14px",
    color: tab === t ? "var(--text-primary)" : "var(--text-tertiary)",
    fontSize: "11px",
    fontFamily: "var(--font-jetbrains)",
    cursor: "pointer" as const,
    fontWeight: tab === t ? 500 : 400,
  });

  return (
    <div style={{ padding: "32px", maxWidth: "800px" }} className="animate-fade-up">
      <button
        onClick={() => router.push("/models")}
        style={{
          background: "none",
          border: "none",
          color: "var(--text-tertiary)",
          fontFamily: "var(--font-jetbrains)",
          fontSize: "11px",
          cursor: "pointer",
          padding: 0,
          marginBottom: "12px",
        }}
      >
        ← Models
      </button>

      {error && (
        <div
          className="rounded-lg"
          style={{
            background: "var(--error-dim)",
            border: "1px solid var(--error)",
            padding: "10px 16px",
            color: "var(--error)",
            fontSize: "12px",
            fontFamily: "var(--font-jetbrains)",
            marginBottom: "12px",
          }}
        >
          {error}
        </div>
      )}

      {/* Header card */}
      <div
        className="rounded-xl"
        style={{
          background: "var(--bg-surface)",
          border: "1px solid var(--border-dim)",
          borderLeft: `3px solid ${taskColor}`,
          padding: "20px 24px",
          marginBottom: "20px",
        }}
      >
        {editing ? (
          <div className="flex flex-col gap-3">
            <input
              value={editName}
              onChange={(e) => setEditName(e.target.value)}
              style={inputStyle}
              placeholder="Model name"
            />
            <div className="flex items-center gap-2">
              <button
                onClick={handleMetaSave}
                disabled={metaSaving}
                style={{
                  background: "var(--gold)",
                  color: "#08080B",
                  border: "none",
                  borderRadius: "6px",
                  padding: "5px 14px",
                  fontSize: "11px",
                  fontWeight: 600,
                  fontFamily: "var(--font-syne)",
                  cursor: "pointer",
                }}
              >
                {metaSaving ? "Saving…" : "Save"}
              </button>
              <button
                onClick={() => {
                  setEditing(false);
                  setEditName(model.name);
                }}
                style={{
                  ...btnBase,
                  background: "transparent",
                  color: "var(--text-secondary)",
                  fontFamily: "var(--font-syne)",
                }}
              >
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <>
            <div
              className="flex items-center justify-between"
              style={{ marginBottom: "8px" }}
            >
              <div className="flex items-center gap-3">
                <h1
                  style={{
                    fontFamily: "var(--font-syne)",
                    fontWeight: 700,
                    fontSize: "20px",
                    color: "var(--text-primary)",
                  }}
                >
                  {model.name}
                </h1>
                <span
                  className="rounded"
                  style={{
                    background: `${taskColor}22`,
                    border: `1px solid ${taskColor}66`,
                    color: taskColor,
                    padding: "2px 8px",
                    fontSize: "10px",
                    fontFamily: "var(--font-jetbrains)",
                  }}
                >
                  {model.task}
                </span>
                <span
                  className="rounded"
                  style={{
                    background: "var(--bg-elevated)",
                    border: "1px solid var(--border)",
                    color: "var(--text-tertiary)",
                    padding: "2px 8px",
                    fontSize: "10px",
                    fontFamily: "var(--font-jetbrains)",
                  }}
                >
                  {model.model_type}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setEditing(true)}
                  style={{ ...btnBase, background: "var(--bg-elevated)", color: "var(--text-secondary)" }}
                >
                  Edit
                </button>
              </div>
            </div>
            <div
              className="flex items-center gap-5"
              style={{
                fontFamily: "var(--font-jetbrains)",
                fontSize: "10px",
                color: "var(--text-tertiary)",
              }}
            >
              <span>v{model.version}</span>
              <span>{new Date(model.created_at).toLocaleString()}</span>
              {model.run_id && <span>run: {model.run_id}</span>}
            </div>
          </>
        )}
      </div>

      {/* Tabs */}
      <div
        className="flex items-end"
        style={{
          borderBottom: "1px solid var(--border-dim)",
          marginBottom: "16px",
        }}
      >
        <button style={tabStyle("metrics")} onClick={() => setTab("metrics")}>
          Metrics
        </button>
        <button style={tabStyle("config")} onClick={() => setTab("config")}>
          Config
        </button>
        <button style={tabStyle("versions")} onClick={() => setTab("versions")}>
          Versions
        </button>
      </div>

      {/* Metrics tab */}
      {tab === "metrics" && (
        <div
          className="rounded-lg"
          style={{
            background: "var(--bg-surface)",
            border: "1px solid var(--border-dim)",
            padding: "16px 20px",
          }}
        >
          {Object.keys(model.metrics).length === 0 ? (
            <div
              style={{
                color: "var(--text-tertiary)",
                fontFamily: "var(--font-jetbrains)",
                fontSize: "12px",
              }}
            >
              No metrics available.
            </div>
          ) : (
            <div className="flex flex-col gap-2">
              {Object.entries(model.metrics).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between">
                  <span
                    style={{
                      fontFamily: "var(--font-jetbrains)",
                      fontSize: "11px",
                      color: "var(--text-secondary)",
                    }}
                  >
                    {key}
                  </span>
                  <span
                    style={{
                      fontFamily: "var(--font-jetbrains)",
                      fontSize: "11px",
                      color: "var(--text-primary)",
                    }}
                  >
                    {typeof value === "number"
                      ? value.toFixed(4)
                      : JSON.stringify(value)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Config tab */}
      {tab === "config" && (
        <div
          className="rounded-lg"
          style={{
            background: "var(--bg-surface)",
            border: "1px solid var(--border-dim)",
            padding: "16px 20px",
          }}
        >
          <pre
            style={{
              fontFamily: "var(--font-jetbrains)",
              fontSize: "11px",
              color: "var(--text-secondary)",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {JSON.stringify(model.config, null, 2)}
          </pre>
        </div>
      )}

      {/* Versions tab */}
      {tab === "versions" && (
        <div className="flex flex-col gap-2">
          {versions.map((v) => (
            <div
              key={v.id}
              className="rounded-lg"
              style={{
                background:
                  v.id === model.id ? "var(--bg-elevated)" : "var(--bg-surface)",
                border: `1px solid ${v.id === model.id ? "var(--border-strong)" : "var(--border-dim)"}`,
                padding: "12px 16px",
              }}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <span
                    style={{
                      fontFamily: "var(--font-syne)",
                      fontSize: "13px",
                      fontWeight: 600,
                      color: "var(--text-primary)",
                    }}
                  >
                    v{v.version}
                  </span>
                  {v.id === model.id && (
                    <span
                      style={{
                        fontSize: "9px",
                        fontFamily: "var(--font-jetbrains)",
                        color: "var(--gold)",
                      }}
                    >
                      current
                    </span>
                  )}
                </div>
                <span
                  style={{
                    fontFamily: "var(--font-jetbrains)",
                    fontSize: "10px",
                    color: "var(--text-tertiary)",
                  }}
                >
                  {new Date(v.created_at).toLocaleString()}
                </span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
