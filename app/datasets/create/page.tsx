"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { createEmptyDataset } from "@/app/lib/api";
import { TagInput } from "@/app/components/TagInput";

export default function CreateDatasetPage() {
  const router = useRouter();

  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [author, setAuthor] = useState("");
  const [columns, setColumns] = useState<string[]>(["", ""]);

  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const addColumn = () => setColumns((prev) => [...prev, ""]);

  const removeColumn = (i: number) => {
    if (columns.length <= 2) return;
    setColumns((prev) => prev.filter((_, idx) => idx !== i));
  };

  const updateColumn = (i: number, val: string) => {
    setColumns((prev) => prev.map((c, idx) => (idx === i ? val : c)));
  };

  const validate = (): string | null => {
    if (!name.trim()) return "Dataset name is required";
    const filled = columns.filter((c) => c.trim());
    if (filled.length < 2) return "At least 2 column names are required";
    const dupes = filled.filter((c, i) => filled.indexOf(c) !== i);
    if (dupes.length > 0) return `Duplicate column names: ${dupes.join(", ")}`;
    return null;
  };

  const handleCreate = async () => {
    const validError = validate();
    if (validError) { setError(validError); return; }

    setCreating(true);
    setError(null);
    try {
      const ds = await createEmptyDataset({
        name: name.trim(),
        columns: columns.filter((c) => c.trim()),
        description: description.trim(),
        tags,
        author: author.trim(),
      });
      router.push(`/datasets/${ds.id}`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to create dataset");
      setCreating(false);
    }
  };

  const inputStyle = {
    background: "var(--bg-elevated)",
    border: "1px solid var(--border)",
    borderRadius: "6px",
    padding: "8px 12px",
    color: "var(--text-primary)",
    fontSize: "13px",
    fontFamily: "var(--font-jetbrains)",
    width: "100%",
    boxSizing: "border-box" as const,
    outline: "none",
  } as const;

  const labelStyle = {
    fontFamily: "var(--font-syne)",
    fontSize: "11px",
    fontWeight: 700,
    letterSpacing: "0.1em",
    textTransform: "uppercase" as const,
    color: "var(--text-tertiary)",
    marginBottom: "6px",
    display: "block",
  };

  return (
    <div className="page-shell page-standard page-shell--xs animate-fade-up">
      {/* Back */}
      <button
        onClick={() => router.push("/datasets")}
        style={{ background: "none", border: "none", color: "var(--text-tertiary)", fontFamily: "var(--font-jetbrains)", fontSize: "11px", cursor: "pointer", padding: 0, marginBottom: "16px" }}
      >
        ← Datasets
      </button>

      <h1 style={{ fontFamily: "var(--font-syne)", fontSize: "20px", fontWeight: 700, color: "var(--text-primary)", marginBottom: "4px" }}>
        Create Dataset
      </h1>
      <p style={{ fontFamily: "var(--font-jetbrains)", fontSize: "12px", color: "var(--text-tertiary)", marginBottom: "28px" }}>
        Define column names to create an empty dataset. You can add rows after creation.
      </p>

      {error && (
        <div className="rounded-lg" style={{ background: "var(--error-dim)", border: "1px solid var(--error)", padding: "10px 16px", color: "var(--error)", fontSize: "12px", fontFamily: "var(--font-jetbrains)", marginBottom: "16px" }}>
          {error}
        </div>
      )}

      <div className="rounded-xl" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-dim)", padding: "24px" }}>
        <div className="flex flex-col gap-5">

          {/* Name */}
          <div>
            <label style={labelStyle}>Dataset Name *</label>
            <input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Survey Responses 2026"
              style={inputStyle}
            />
          </div>

          {/* Description */}
          <div>
            <label style={labelStyle}>Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Optional description"
              rows={2}
              style={{ ...inputStyle, resize: "vertical" }}
            />
          </div>

          {/* Tags */}
          <div>
            <label style={labelStyle}>Tags</label>
            <TagInput tags={tags} onChange={setTags} />
          </div>

          {/* Author */}
          <div>
            <label style={labelStyle}>Author</label>
            <input
              value={author}
              onChange={(e) => setAuthor(e.target.value)}
              placeholder="Your name (optional)"
              style={inputStyle}
            />
          </div>

          {/* Columns */}
          <div>
            <label style={labelStyle}>Columns *</label>
            <p style={{ fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--text-tertiary)", marginBottom: "10px" }}>
              Minimum 2 columns required. Each must have a unique name.
            </p>
            <div className="flex flex-col gap-2">
              {columns.map((col, i) => (
                <div key={i} className="flex items-center gap-2">
                  <span style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)", minWidth: "20px", textAlign: "right" }}>
                    {i + 1}
                  </span>
                  <input
                    value={col}
                    onChange={(e) => updateColumn(i, e.target.value)}
                    placeholder={`column_${i + 1}`}
                    style={{ ...inputStyle, width: "auto", flex: 1 }}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") { e.preventDefault(); addColumn(); }
                    }}
                  />
                  <button
                    onClick={() => removeColumn(i)}
                    disabled={columns.length <= 2}
                    style={{
                      background: "none",
                      border: "none",
                      color: columns.length <= 2 ? "var(--border)" : "var(--error)",
                      cursor: columns.length <= 2 ? "default" : "pointer",
                      fontSize: "14px",
                      padding: "4px",
                      flexShrink: 0,
                    }}
                    title="Remove column"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
            <button
              onClick={addColumn}
              style={{
                marginTop: "10px",
                background: "transparent",
                border: "1px dashed var(--border)",
                borderRadius: "6px",
                padding: "6px 14px",
                color: "var(--text-tertiary)",
                fontSize: "12px",
                fontFamily: "var(--font-jetbrains)",
                cursor: "pointer",
                width: "100%",
              }}
            >
              + Add column
            </button>
          </div>

          {/* Actions */}
          <div
            style={{
              paddingTop: "4px",
              display: "grid",
              gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
              gap: "0.75rem",
              width: "100%",
              maxWidth: "42rem",
              marginInline: "auto",
              alignItems: "stretch",
            }}
          >
            <button
              type="button"
              onClick={handleCreate}
              disabled={creating}
              style={{
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                width: "100%",
                minHeight: "2.75rem",
                background: "var(--gold)",
                color: "#08080B",
                border: "none",
                borderRadius: "6px",
                padding: "0.625rem 1rem",
                fontSize: "13px",
                fontWeight: 600,
                fontFamily: "var(--font-syne)",
                textAlign: "center",
                cursor: creating ? "default" : "pointer",
                opacity: creating ? 0.7 : 1,
              }}
            >
              {creating ? "Creating…" : "Create Dataset"}
            </button>
            <button
              type="button"
              onClick={() => router.push("/datasets")}
              style={{
                display: "inline-flex",
                alignItems: "center",
                justifyContent: "center",
                width: "100%",
                minHeight: "2.75rem",
                background: "transparent",
                border: "1px solid var(--border)",
                borderRadius: "6px",
                padding: "0.625rem 1rem",
                fontSize: "13px",
                fontWeight: 600,
                color: "var(--text-secondary)",
                fontFamily: "var(--font-syne)",
                textAlign: "center",
                cursor: "pointer",
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

