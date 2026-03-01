"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { FileUpload } from "@/app/components/FileUpload";
import { TagInput } from "@/app/components/TagInput";
import { uploadDataset } from "@/app/lib/api";

export default function UploadDatasetPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [tags, setTags] = useState<string[]>([]);
  const [author, setAuthor] = useState("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!file) return;
    if (!name.trim()) {
      setError("Name is required");
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const result = await uploadDataset(file, {
        name: name.trim(),
        description: description.trim(),
        tags,
        author: author.trim(),
      });
      router.push(`/datasets/${result.id}`);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : "Upload failed";
      // Try to extract validation errors
      try {
        const parsed = JSON.parse(msg.replace(/^Upload failed: /, ""));
        if (parsed.detail?.errors) {
          setError(parsed.detail.errors.join("; "));
        } else {
          setError(msg);
        }
      } catch {
        setError(msg);
      }
    } finally {
      setUploading(false);
    }
  }

  const inputStyle = {
    background: "var(--bg-elevated)",
    border: "1px solid var(--border)",
    borderRadius: "6px",
    padding: "8px 12px",
    color: "var(--text-primary)",
    fontSize: "12px",
    fontFamily: "var(--font-jetbrains)",
    width: "100%",
  } as const;

  const labelStyle = {
    fontFamily: "var(--font-syne)",
    fontSize: "11px",
    fontWeight: 600,
    letterSpacing: "0.06em",
    color: "var(--text-secondary)",
    marginBottom: "6px",
    display: "block",
  } as const;

  return (
    <div className="page-shell page-standard page-shell--xs animate-fade-up">
      <div style={{ marginBottom: "16px" }}>
        <button
          type="button"
          onClick={() => router.push("/datasets")}
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "0.35rem",
            background: "none",
            border: "none",
            padding: 0,
            color: "var(--text-tertiary)",
            fontFamily: "var(--font-jetbrains)",
            fontSize: "11px",
            cursor: "pointer",
          }}
        >
          <span aria-hidden="true">&larr;</span>
          <span>Datasets</span>
        </button>
      </div>

      {/* Header */}
      <div style={{ marginBottom: "28px" }}>
        <div
          style={{
            fontFamily: "var(--font-syne)",
            fontSize: "9.5px",
            fontWeight: 700,
            letterSpacing: "0.18em",
            textTransform: "uppercase",
            color: "var(--text-tertiary)",
            marginBottom: "6px",
          }}
        >
          Data Management
        </div>
        <h1
          style={{
            fontFamily: "var(--font-syne)",
            fontWeight: 700,
            fontSize: "22px",
            color: "var(--text-primary)",
            letterSpacing: "-0.01em",
          }}
        >
          Upload Dataset
        </h1>
      </div>

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
            marginBottom: "16px",
          }}
        >
          {error}
        </div>
      )}

      <form onSubmit={handleSubmit}>
        <div
          className="rounded-xl"
          style={{
            background: "var(--bg-surface)",
            border: "1px solid var(--border-dim)",
            padding: "24px",
          }}
        >
          {/* File */}
          <div style={{ marginBottom: "20px" }}>
            <label style={labelStyle}>CSV File</label>
            <FileUpload onFile={setFile} disabled={uploading} />
          </div>

          {/* Name */}
          <div style={{ marginBottom: "16px" }}>
            <label style={labelStyle}>
              Name <span style={{ color: "var(--error)" }}>*</span>
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g., Student Survey Spring 2026"
              style={inputStyle}
              disabled={uploading}
            />
          </div>

          {/* Description */}
          <div style={{ marginBottom: "16px" }}>
            <label style={labelStyle}>Description</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Optional description…"
              rows={3}
              style={{ ...inputStyle, resize: "vertical" }}
              disabled={uploading}
            />
          </div>

          {/* Tags */}
          <div style={{ marginBottom: "16px" }}>
            <label style={labelStyle}>Tags</label>
            <TagInput tags={tags} onChange={setTags} />
          </div>

          {/* Author */}
          <div style={{ marginBottom: "24px" }}>
            <label style={labelStyle}>Author</label>
            <input
              type="text"
              value={author}
              onChange={(e) => setAuthor(e.target.value)}
              placeholder="Your name"
              style={inputStyle}
              disabled={uploading}
            />
          </div>

          {/* Submit */}
          <div
            style={{
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
              type="submit"
              disabled={!file || !name.trim() || uploading}
              className="inline-flex items-center gap-2 rounded-lg"
              style={{
                justifyContent: "center",
                width: "100%",
                minHeight: "2.75rem",
                background:
                  !file || !name.trim() || uploading
                    ? "var(--bg-elevated)"
                    : "var(--gold)",
                color:
                  !file || !name.trim() || uploading
                    ? "var(--text-tertiary)"
                    : "#08080B",
                padding: "0.625rem 1rem",
                fontSize: "12px",
                fontWeight: 600,
                fontFamily: "var(--font-syne)",
                letterSpacing: "0.04em",
                border: "none",
                cursor:
                  !file || !name.trim() || uploading ? "not-allowed" : "pointer",
              }}
            >
              {uploading ? (
                <>
                  <span
                    className="animate-spin"
                    style={{
                      width: "12px",
                      height: "12px",
                      border: "2px solid transparent",
                      borderTopColor: "currentColor",
                      borderRadius: "50%",
                      display: "inline-block",
                    }}
                  />
                  Uploading…
                </>
              ) : (
                "Upload"
              )}
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
                borderRadius: "8px",
                padding: "0.625rem 1rem",
                color: "var(--text-secondary)",
                fontSize: "12px",
                fontWeight: 600,
                fontFamily: "var(--font-syne)",
                textAlign: "center",
                cursor: "pointer",
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}

