"use client";

import { useRef, useState, useCallback } from "react";
import { useI18n } from "@/app/lib/i18n/provider";

interface FileUploadProps {
  accept?: string;
  onFile: (file: File) => void;
  disabled?: boolean;
}

export function FileUpload({ accept = ".csv", onFile, disabled }: FileUploadProps) {
  const { t } = useI18n();
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
      className={`ui-dropzone flex flex-col items-center justify-center transition-all duration-150${
        dragging ? " is-dragging" : ""
      }${disabled ? " is-disabled" : ""}`}
      style={{
        borderColor: dragging ? "var(--gold)" : undefined,
        background: dragging ? "var(--gold-faint)" : undefined,
      }}
    >
      <input
        ref={inputRef}
        type="file"
        aria-label={t("Choose file")}
        accept={accept}
        onChange={handleChange}
        style={{ display: "none" }}
      />
      <svg
        width="28"
        height="28"
        viewBox="0 0 28 28"
        fill="none"
        className="ui-dropzone__icon"
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
        <div className="ui-dropzone__name">{fileName}</div>
      ) : (
        <>
          <div className="ui-dropzone__title">{t("Drop CSV file here")}</div>
          <div className="ui-dropzone__hint">{t("or click to browse")}</div>
        </>
      )}
    </div>
  );
}
