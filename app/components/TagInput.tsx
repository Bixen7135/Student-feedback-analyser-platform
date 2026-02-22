"use client";

import { useState, useRef } from "react";

interface TagInputProps {
  tags: string[];
  onChange: (tags: string[]) => void;
  placeholder?: string;
}

export function TagInput({ tags, onChange, placeholder = "Add tag…" }: TagInputProps) {
  const [input, setInput] = useState("");
  const ref = useRef<HTMLInputElement>(null);

  function add() {
    const val = input.trim().toLowerCase();
    if (val && !tags.includes(val)) {
      onChange([...tags, val]);
    }
    setInput("");
  }

  function remove(tag: string) {
    onChange(tags.filter((t) => t !== tag));
  }

  return (
    <div
      className="flex flex-wrap items-center gap-1.5"
      style={{
        background: "var(--bg-elevated)",
        border: "1px solid var(--border)",
        borderRadius: "6px",
        padding: "4px 6px",
        minHeight: "34px",
        cursor: "text",
      }}
      onClick={() => ref.current?.focus()}
    >
      {tags.map((tag) => (
        <span
          key={tag}
          className="inline-flex items-center gap-1 rounded"
          style={{
            background: "var(--gold-faint)",
            border: "1px solid var(--gold-muted)",
            color: "var(--gold)",
            padding: "1px 6px",
            fontSize: "10px",
            fontFamily: "var(--font-jetbrains)",
          }}
        >
          {tag}
          <button
            type="button"
            onClick={(e) => { e.stopPropagation(); remove(tag); }}
            style={{
              background: "none",
              border: "none",
              color: "var(--gold-muted)",
              cursor: "pointer",
              padding: "0 1px",
              fontSize: "12px",
              lineHeight: 1,
            }}
          >
            ×
          </button>
        </span>
      ))}
      <input
        ref={ref}
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === ",") {
            e.preventDefault();
            add();
          }
          if (e.key === "Backspace" && !input && tags.length) {
            remove(tags[tags.length - 1]);
          }
        }}
        onBlur={add}
        placeholder={tags.length === 0 ? placeholder : ""}
        style={{
          background: "transparent",
          border: "none",
          outline: "none",
          color: "var(--text-primary)",
          fontSize: "11px",
          fontFamily: "var(--font-jetbrains)",
          flex: 1,
          minWidth: "60px",
          padding: "2px 4px",
        }}
      />
    </div>
  );
}
