"use client";

import { useRef, useState, useCallback } from "react";

interface FileUploadProps {
  accept?: string;
  onFile: (file: File) => void;
  disabled?: boolean;
}

export function FileUpload({ accept = ".csv", onFile, disabled }: FileUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);

  const handleFile = useCallback(
    (file: File) => {
      setFileName(file.name);
      onFile(file);
    },
    [onFile]
  );

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDragging(false);
    if (disabled) return;
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => { if (!disabled) inputRef.current?.click(); }}
      className="flex flex-col items-center justify-center rounded-lg transition-all duration-150"
      style={{
        border: `2px dashed ${dragging ? "var(--gold)" : "var(--border)"}`,
        background: dragging ? "var(--gold-faint)" : "var(--bg-elevated)",
        padding: "32px 20px",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.5 : 1,
      }}
    >
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        onChange={handleChange}
        style={{ display: "none" }}
      />
      <svg
        width="28"
        height="28"
        viewBox="0 0 28 28"
        fill="none"
        style={{ marginBottom: "10px", opacity: 0.5 }}
      >
        <path
          d="M14 18V8M14 8L10 12M14 8L18 12"
          stroke="var(--text-tertiary)"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M6 20H22"
          stroke="var(--text-tertiary)"
          strokeWidth="1.5"
          strokeLinecap="round"
        />
      </svg>
      {fileName ? (
        <div
          style={{
            fontFamily: "var(--font-jetbrains)",
            fontSize: "12px",
            color: "var(--gold)",
          }}
        >
          {fileName}
        </div>
      ) : (
        <>
          <div
            style={{
              fontFamily: "var(--font-syne)",
              fontSize: "13px",
              fontWeight: 600,
              color: "var(--text-secondary)",
              marginBottom: "4px",
            }}
          >
            Drop CSV file here
          </div>
          <div
            style={{
              fontFamily: "var(--font-jetbrains)",
              fontSize: "10px",
              color: "var(--text-tertiary)",
            }}
          >
            or click to browse
          </div>
        </>
      )}
    </div>
  );
}
