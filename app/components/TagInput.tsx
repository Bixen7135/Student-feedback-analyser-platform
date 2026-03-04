"use client";

import { useRef, useState } from "react";
import { useI18n } from "@/app/lib/i18n/provider";

interface TagInputProps {
  tags: string[];
  onChange: (tags: string[]) => void;
  placeholder?: string;
}

export function TagInput({
  tags,
  onChange,
  placeholder = "Add tag...",
}: TagInputProps) {
  const { t } = useI18n();
  const [input, setInput] = useState("");
  const ref = useRef<HTMLInputElement>(null);

  function add() {
    const value = input.trim().toLowerCase();
    if (value && !tags.includes(value)) {
      onChange([...tags, value]);
    }
    setInput("");
  }

  function remove(tag: string) {
    onChange(tags.filter((item) => item !== tag));
  }

  return (
    <div className="ui-tag-input" onClick={() => ref.current?.focus()}>
      {tags.map((tag) => (
        <span key={tag} className="ui-tag-chip">
          {tag}
          <button
            type="button"
            aria-label={t("Remove tag")}
            onClick={(event) => {
              event.stopPropagation();
              remove(tag);
            }}
            className="ui-tag-chip__remove"
          >
            x
          </button>
        </span>
      ))}
      <input
        ref={ref}
        value={input}
        onChange={(event) => setInput(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === ",") {
            event.preventDefault();
            add();
          }
          if (event.key === "Backspace" && !input && tags.length) {
            remove(tags[tags.length - 1]);
          }
        }}
        onBlur={add}
        placeholder={tags.length === 0 ? t(placeholder) : ""}
        className="ui-tag-input__field"
      />
    </div>
  );
}
