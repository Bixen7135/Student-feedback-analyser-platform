"use client";

import { useState, useEffect, useRef, useCallback } from "react";

interface DataTableProps {
  storageKey?: string;
  columns: string[];
  rows: string[][];
  totalRows: number;
  offset: number;
  limit: number;
  onPageChange: (offset: number) => void;
  onLimitChange?: (limit: number) => void;
  loading?: boolean;
  // Phase 6: editing
  editable?: boolean;
  pendingEdits?: Map<string, string>;
  onCellEdit?: (absoluteRowIdx: number, colIdx: number, value: string) => void;
  onDeleteRows?: (absoluteRowIndices: number[]) => void;
  // Column rename
  pendingColumnRenames?: Map<number, string>;
  onColumnRename?: (colIdx: number, newName: string) => void;
}

interface CellRef {
  absRowIdx: number;
  colIdx: number;
}

const MIN_COLUMN_WIDTH = 80;
const MAX_AUTO_COLUMN_WIDTH = 176;
const AUTO_COLUMN_CHAR_WIDTH = 7;
const AUTO_COLUMN_BASE_PADDING = 34;
const MIN_ROW_HEIGHT = 26;
const MAX_ROW_HEIGHT = 240;

function getAutoColumnWidth(label: string): number {
  const normalized = label.trim() || "Column";
  const estimatedWidth =
    normalized.length * AUTO_COLUMN_CHAR_WIDTH + AUTO_COLUMN_BASE_PADDING;

  return Math.max(
    MIN_COLUMN_WIDTH,
    Math.min(MAX_AUTO_COLUMN_WIDTH, estimatedWidth)
  );
}

export function DataTable({
  storageKey,
  columns,
  rows,
  totalRows,
  offset,
  limit,
  onPageChange,
  onLimitChange,
  loading,
  editable,
  pendingEdits,
  onCellEdit,
  onDeleteRows,
  pendingColumnRenames,
  onColumnRename,
}: DataTableProps) {
  const totalPages = Math.ceil(totalRows / limit) || 1;
  const currentPage = Math.floor(offset / limit) + 1;

  // Page jump input (controlled locally, syncs when offset changes)
  const [pageInput, setPageInput] = useState(String(currentPage));
  useEffect(() => { setPageInput(String(currentPage)); }, [currentPage]);

  // Clamp pagination when page size or total rows change.
  useEffect(() => {
    if (totalRows <= 0 && offset !== 0) {
      onPageChange(0);
      return;
    }
    if (totalRows <= 0) return;
    const maxOffset = Math.max(0, (totalPages - 1) * limit);
    if (offset > maxOffset) {
      onPageChange(maxOffset);
    }
  }, [offset, totalRows, totalPages, limit, onPageChange]);

  // Cell editing
  const [editingCell, setEditingCell] = useState<CellRef | null>(null);
  const [editValue, setEditValue] = useState("");
  const editInputRef = useRef<HTMLInputElement>(null);
  const [activeCell, setActiveCell] = useState<CellRef | null>(null);
  const cellRefs = useRef<Map<string, HTMLTableCellElement>>(new Map());
  const [columnWidths, setColumnWidths] = useState<Map<number, number>>(new Map());
  const [rowHeights, setRowHeights] = useState<Map<number, number>>(new Map());
  const [dragCursor, setDragCursor] = useState<"col-resize" | "row-resize" | null>(null);
  const resizingColumnRef = useRef<{ colIdx: number; startX: number; startWidth: number } | null>(null);
  const resizingRowRef = useRef<{ absRowIdx: number; startY: number; startHeight: number } | null>(null);

  // Column header editing
  const [editingHeader, setEditingHeader] = useState<number | null>(null);
  const [headerEditValue, setHeaderEditValue] = useState("");
  const headerEditRef = useRef<HTMLInputElement>(null);

  // Row selection for deletion
  const [selectedRows, setSelectedRows] = useState<Set<number>>(new Set());
  // Whether all rows in the entire document are selected (not just page)
  const [allDocumentSelected, setAllDocumentSelected] = useState(false);

  // Find / Replace
  const [showFind, setShowFind] = useState(false);
  const [showReplace, setShowReplace] = useState(false);
  const [findText, setFindText] = useState("");
  const [findCol, setFindCol] = useState("");
  const [replaceText, setReplaceText] = useState("");
  const [matchIdx, setMatchIdx] = useState(0);
  const findInputRef = useRef<HTMLInputElement>(null);

  // Global Ctrl+F to open find panel
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "f") {
        e.preventDefault();
        setShowFind(true);
        setTimeout(() => findInputRef.current?.focus(), 50);
      }
      if (e.key === "Escape") {
        setShowFind(false);
        setShowReplace(false);
        setEditingCell(null);
        setEditingHeader(null);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  // Focus edit input when cell enters edit mode
  useEffect(() => {
    if (editingCell) setTimeout(() => editInputRef.current?.focus(), 20);
  }, [editingCell]);

  useEffect(() => {
    const handlePointerMove = (e: MouseEvent) => {
      if (resizingColumnRef.current) {
        const { colIdx, startX, startWidth } = resizingColumnRef.current;
        const nextWidth = Math.max(MIN_COLUMN_WIDTH, startWidth + (e.clientX - startX));
        setColumnWidths((prev) => {
          const next = new Map(prev);
          next.set(colIdx, nextWidth);
          return next;
        });
      }
      if (resizingRowRef.current) {
        const { absRowIdx, startY, startHeight } = resizingRowRef.current;
        const nextHeight = Math.max(
          MIN_ROW_HEIGHT,
          Math.min(MAX_ROW_HEIGHT, startHeight + (e.clientY - startY))
        );
        setRowHeights((prev) => {
          const next = new Map(prev);
          next.set(absRowIdx, nextHeight);
          return next;
        });
      }
    };

    const handlePointerUp = () => {
      if (!resizingColumnRef.current && !resizingRowRef.current) return;
      resizingColumnRef.current = null;
      resizingRowRef.current = null;
      setDragCursor(null);
    };

    window.addEventListener("mousemove", handlePointerMove);
    window.addEventListener("mouseup", handlePointerUp);
    return () => {
      window.removeEventListener("mousemove", handlePointerMove);
      window.removeEventListener("mouseup", handlePointerUp);
    };
  }, []);

  // Restore persisted column widths for this table instance.
  useEffect(() => {
    if (!storageKey) return;
    try {
      const raw = window.localStorage.getItem(`datatable:column-widths:${storageKey}`);
      if (!raw) return;
      const saved = JSON.parse(raw) as Record<string, number>;
      const restored = new Map<number, number>();
      columns.forEach((col, colIdx) => {
        const width = saved[col];
        if (typeof width === "number" && Number.isFinite(width) && width >= MIN_COLUMN_WIDTH) {
          restored.set(colIdx, width);
        }
      });
      setColumnWidths(restored);
    } catch {
      // ignore invalid local storage value
    }
  }, [storageKey, columns]);

  // Persist column widths whenever user resizes columns.
  useEffect(() => {
    if (!storageKey) return;
    try {
      const serialized: Record<string, number> = {};
      columnWidths.forEach((width, colIdx) => {
        const col = columns[colIdx];
        if (col !== undefined) {
          serialized[col] = Math.round(width);
        }
      });
      window.localStorage.setItem(
        `datatable:column-widths:${storageKey}`,
        JSON.stringify(serialized)
      );
    } catch {
      // ignore storage failures (private mode / quota)
    }
  }, [storageKey, columnWidths, columns]);

  // Restore persisted row heights for this table instance.
  useEffect(() => {
    if (!storageKey) return;
    try {
      const raw = window.localStorage.getItem(`datatable:row-heights:${storageKey}`);
      if (!raw) return;
      const saved = JSON.parse(raw) as Record<string, number>;
      const restored = new Map<number, number>();
      Object.entries(saved).forEach(([rowIdx, height]) => {
        const absRowIdx = Number(rowIdx);
        if (
          Number.isInteger(absRowIdx) &&
          typeof height === "number" &&
          Number.isFinite(height)
        ) {
          restored.set(
            absRowIdx,
            Math.max(MIN_ROW_HEIGHT, Math.min(MAX_ROW_HEIGHT, Math.round(height)))
          );
        }
      });
      setRowHeights(restored);
    } catch {
      // ignore invalid local storage value
    }
  }, [storageKey]);

  // Persist row heights whenever user resizes rows.
  useEffect(() => {
    if (!storageKey) return;
    try {
      const serialized: Record<string, number> = {};
      rowHeights.forEach((height, absRowIdx) => {
        serialized[String(absRowIdx)] = Math.round(height);
      });
      window.localStorage.setItem(
        `datatable:row-heights:${storageKey}`,
        JSON.stringify(serialized)
      );
    } catch {
      // ignore storage failures (private mode / quota)
    }
  }, [storageKey, rowHeights]);

  useEffect(() => {
    document.body.style.userSelect = dragCursor ? "none" : "";
    document.body.style.cursor = dragCursor ?? "";
    return () => {
      document.body.style.userSelect = "";
      document.body.style.cursor = "";
    };
  }, [dragCursor]);

  // Keep active cell valid for current viewport and focus it for arrow navigation.
  useEffect(() => {
    if (rows.length === 0) {
      setActiveCell(null);
      return;
    }
    if (!activeCell) return;
    const minAbsRow = offset;
    const maxAbsRow = offset + rows.length - 1;
    if (
      activeCell.absRowIdx < minAbsRow ||
      activeCell.absRowIdx > maxAbsRow ||
      activeCell.colIdx < 0 ||
      activeCell.colIdx >= columns.length
    ) {
      setActiveCell({ absRowIdx: minAbsRow, colIdx: 0 });
      return;
    }
    const key = `${activeCell.absRowIdx}:${activeCell.colIdx}`;
    const target = cellRefs.current.get(key);
    if (target && document.activeElement !== editInputRef.current) {
      target.focus();
    }
  }, [activeCell, rows, offset, columns.length]);

  // Focus header edit input
  useEffect(() => {
    if (editingHeader !== null) setTimeout(() => headerEditRef.current?.focus(), 20);
  }, [editingHeader]);

  // Reset document-all selection when page changes
  useEffect(() => {
    setAllDocumentSelected(false);
  }, [offset]);

  // Build find matches from currently visible rows
  const matches: CellRef[] = [];
  if (findText) {
    const lowerFind = findText.toLowerCase();
    rows.forEach((row, rowIdx) => {
      row.forEach((cell, colIdx) => {
        if (findCol && columns[colIdx] !== findCol) return;
        const cellVal = getCellDisplayValue(rowIdx, colIdx, row, pendingEdits, offset);
        if (cellVal.toLowerCase().includes(lowerFind)) {
          matches.push({ absRowIdx: offset + rowIdx, colIdx });
        }
      });
    });
  }

  const safeMatchIdx = matches.length > 0 ? matchIdx % matches.length : 0;
  const matchSet = new Set(matches.map((m) => `${m.absRowIdx}:${m.colIdx}`));
  const currentMatchKey = matches.length > 0
    ? `${matches[safeMatchIdx].absRowIdx}:${matches[safeMatchIdx].colIdx}`
    : null;

  // Commit pending page jump
  const commitPageJump = useCallback(() => {
    const p = Math.max(1, Math.min(totalPages, parseInt(pageInput) || 1));
    setPageInput(String(p));
    onPageChange((p - 1) * limit);
  }, [pageInput, totalPages, limit, onPageChange]);

  // Start editing a cell
  const startEdit = (absRowIdx: number, colIdx: number, currentVal: string) => {
    if (!editable) return;
    const editKey = `${absRowIdx}:${colIdx}`;
    const pendingVal = pendingEdits?.get(editKey);
    setEditingCell({ absRowIdx, colIdx });
    setEditValue(pendingVal !== undefined ? pendingVal : currentVal);
  };

  const moveActiveCell = useCallback((dRow: number, dCol: number) => {
    if (rows.length === 0 || columns.length === 0) return;
    const minAbsRow = offset;
    const maxAbsRow = offset + rows.length - 1;
    const start = activeCell ?? { absRowIdx: minAbsRow, colIdx: 0 };
    const nextAbsRow = Math.max(minAbsRow, Math.min(maxAbsRow, start.absRowIdx + dRow));
    const nextColIdx = Math.max(0, Math.min(columns.length - 1, start.colIdx + dCol));
    setActiveCell({ absRowIdx: nextAbsRow, colIdx: nextColIdx });
  }, [activeCell, columns.length, offset, rows.length]);

  // Commit cell edit
  const commitEdit = () => {
    if (!editingCell) return;
    onCellEdit?.(editingCell.absRowIdx, editingCell.colIdx, editValue);
    setEditingCell(null);
  };

  // Start editing a column header
  const startHeaderEdit = (colIdx: number) => {
    if (!editable || !onColumnRename) return;
    const pendingName = pendingColumnRenames?.get(colIdx);
    setEditingHeader(colIdx);
    setHeaderEditValue(pendingName !== undefined ? pendingName : columns[colIdx]);
  };

  // Commit column header rename
  const commitHeaderEdit = () => {
    if (editingHeader === null) return;
    const trimmed = headerEditValue.trim();
    if (trimmed && trimmed !== columns[editingHeader]) {
      onColumnRename?.(editingHeader, trimmed);
    }
    setEditingHeader(null);
  };

  // Toggle row selection
  const toggleRow = (absRowIdx: number) => {
    if (allDocumentSelected) {
      // Deselect from document selection: switch back to "all page except this one"
      setAllDocumentSelected(false);
      const pageSet = new Set(rows.map((_, i) => offset + i));
      pageSet.delete(absRowIdx);
      setSelectedRows(pageSet);
      return;
    }
    setSelectedRows((prev) => {
      const next = new Set(prev);
      if (next.has(absRowIdx)) next.delete(absRowIdx);
      else next.add(absRowIdx);
      return next;
    });
  };

  const toggleAll = () => {
    if (allDocumentSelected) {
      // Clear everything
      setAllDocumentSelected(false);
      setSelectedRows(new Set());
    } else if (selectedRows.size > 0) {
      // Deselect all
      setSelectedRows(new Set());
      setAllDocumentSelected(false);
    } else {
      // Select all on current page
      setSelectedRows(new Set(rows.map((_, i) => offset + i)));
      setAllDocumentSelected(false);
    }
  };

  const selectAllDocument = () => {
    setAllDocumentSelected(true);
    setSelectedRows(new Set(rows.map((_, i) => offset + i)));
  };

  const clearSelection = () => {
    setAllDocumentSelected(false);
    setSelectedRows(new Set());
  };

  const handleDeleteSelected = () => {
    if (allDocumentSelected) {
      // Delete all rows in entire document
      const allIndices = Array.from({ length: totalRows }, (_, i) => i);
      onDeleteRows?.(allIndices);
    } else {
      onDeleteRows?.([...selectedRows]);
    }
    setSelectedRows(new Set());
    setAllDocumentSelected(false);
  };

  const onTableKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (editingCell) return;
    if (e.key === "ArrowUp") {
      e.preventDefault();
      moveActiveCell(-1, 0);
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      moveActiveCell(1, 0);
      return;
    }
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      moveActiveCell(0, -1);
      return;
    }
    if (e.key === "ArrowRight") {
      e.preventDefault();
      moveActiveCell(0, 1);
      return;
    }
    if ((e.key === "Enter" || e.key === "F2") && editable && activeCell) {
      e.preventDefault();
      const rowIdx = activeCell.absRowIdx - offset;
      if (rowIdx < 0 || rowIdx >= rows.length) return;
      const row = rows[rowIdx];
      const currentVal = getCellDisplayValue(rowIdx, activeCell.colIdx, row, pendingEdits, offset);
      startEdit(activeCell.absRowIdx, activeCell.colIdx, currentVal);
    }
  };

  const startColumnResize = (e: React.MouseEvent<HTMLDivElement>, colIdx: number) => {
    e.preventDefault();
    e.stopPropagation();
    const headerCell = e.currentTarget.parentElement as HTMLTableCellElement | null;
    const measuredWidth = headerCell?.getBoundingClientRect().width ?? MIN_COLUMN_WIDTH;
    const startWidth = columnWidths.get(colIdx) ?? measuredWidth;
    resizingColumnRef.current = { colIdx, startX: e.clientX, startWidth };
    setDragCursor("col-resize");
  };

  const startRowResize = (e: React.MouseEvent<HTMLDivElement>, absRowIdx: number) => {
    e.preventDefault();
    e.stopPropagation();
    const rowElement = e.currentTarget.closest("tr");
    const measuredHeight = rowElement?.getBoundingClientRect().height ?? MIN_ROW_HEIGHT;
    const startHeight = rowHeights.get(absRowIdx) ?? measuredHeight;
    resizingRowRef.current = { absRowIdx, startY: e.clientY, startHeight };
    setDragCursor("row-resize");
  };

  // Replace current match
  const replaceCurrent = () => {
    if (matches.length === 0) return;
    const m = matches[safeMatchIdx];
    const rowIdx = m.absRowIdx - offset;
    const currentVal = getCellDisplayValue(rowIdx, m.colIdx, rows[rowIdx], pendingEdits, offset);
    const newVal = currentVal.replace(new RegExp(escapeRegex(findText), "gi"), replaceText);
    onCellEdit?.(m.absRowIdx, m.colIdx, newVal);
    setMatchIdx((prev) => (prev + 1) % Math.max(1, matches.length));
  };

  // Replace all matches on current page
  const replaceAll = () => {
    matches.forEach((m) => {
      const rowIdx = m.absRowIdx - offset;
      const currentVal = getCellDisplayValue(rowIdx, m.colIdx, rows[rowIdx], pendingEdits, offset);
      const newVal = currentVal.replace(new RegExp(escapeRegex(findText), "gi"), replaceText);
      onCellEdit?.(m.absRowIdx, m.colIdx, newVal);
    });
  };

  // Derived checkbox states
  const allPageSelected = rows.length > 0 && selectedRows.size >= rows.length &&
    rows.every((_, i) => selectedRows.has(offset + i));
  const someSelected = selectedRows.size > 0 || allDocumentSelected;
  const showSelectAllDocumentBanner =
    allPageSelected && !allDocumentSelected && totalRows > rows.length && editable;

  const selectedCount = allDocumentSelected ? totalRows : selectedRows.size;

  return (
    <div>
      {/* Toolbar: search column + find button */}
      <div className="data-table__toolbar" style={{ marginBottom: "10px" }}>
        <span
          className="data-table__toolbar-count"
          style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)", marginLeft: "auto" }}
        >
          {totalRows.toLocaleString()} rows
        </span>
        <button
          type="button"
          className="data-table__toolbar-find"
          onClick={() => { setShowFind((v) => !v); setTimeout(() => findInputRef.current?.focus(), 50); }}
          style={{
            background: showFind ? "var(--accent)" : "var(--bg-elevated)",
            border: "1px solid var(--border)",
            borderRadius: "6px",
            padding: "4px 10px",
            color: showFind ? "#fff" : "var(--text-secondary)",
            fontSize: "11px",
            fontFamily: "var(--font-jetbrains)",
            cursor: "pointer",
          }}
        >
          Find {findText ? `(${matches.length})` : ""}
        </button>
      </div>

      {/* Find / Replace panel */}
      {showFind && (
        <div style={{
          background: "var(--bg-elevated)",
          border: "1px solid var(--border)",
          borderRadius: "8px",
          padding: "10px 12px",
          marginBottom: "10px",
          display: "flex",
          flexDirection: "column",
          gap: "8px",
        }}>
          <div className="flex items-center gap-2" style={{ flexWrap: "wrap" }}>
            <input
              ref={findInputRef}
              type="text"
              placeholder="Find…"
              value={findText}
              onChange={(e) => { setFindText(e.target.value); setMatchIdx(0); }}
              onKeyDown={(e) => {
                if (e.key === "Enter") setMatchIdx((p) => (p + 1) % Math.max(1, matches.length));
              }}
              style={inputStyle}
            />
            <select
              value={findCol}
              onChange={(e) => setFindCol(e.target.value)}
              style={{ ...selectStyle, maxWidth: "200px" }}
            >
              <option value="">All columns</option>
              {columns.map((c) => <option key={c} value={c}>{c}</option>)}
            </select>
            <span style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)", minWidth: "64px" }}>
              {matches.length > 0 ? `${safeMatchIdx + 1} / ${matches.length}` : findText ? "0 matches" : ""}
            </span>
            <button onClick={() => setMatchIdx((p) => Math.max(0, p - 1))} disabled={matches.length === 0} style={smallBtnStyle}>▲</button>
            <button onClick={() => setMatchIdx((p) => (p + 1) % Math.max(1, matches.length))} disabled={matches.length === 0} style={smallBtnStyle}>▼</button>
            <button
              onClick={() => setShowReplace((v) => !v)}
              style={{ ...smallBtnStyle, marginLeft: "auto" }}
            >
              {showReplace ? "▲ Replace" : "▼ Replace"}
            </button>
            <button onClick={() => { setShowFind(false); setShowReplace(false); setFindText(""); }} style={smallBtnStyle}>✕</button>
          </div>

          {showReplace && editable && (
            <div className="flex items-center gap-2">
              <input
                type="text"
                placeholder="Replace with…"
                value={replaceText}
                onChange={(e) => setReplaceText(e.target.value)}
                style={inputStyle}
              />
              <button onClick={replaceCurrent} disabled={matches.length === 0} style={smallBtnStyle}>Replace</button>
              <button onClick={replaceAll} disabled={matches.length === 0} style={smallBtnStyle}>Replace All</button>
            </div>
          )}
        </div>
      )}

      {/* Select-all-document banner */}
      {showSelectAllDocumentBanner && (
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "10px",
          background: "rgba(99,102,241,0.08)",
          border: "1px solid rgba(99,102,241,0.3)",
          borderRadius: "6px",
          padding: "6px 12px",
          marginBottom: "10px",
          fontFamily: "var(--font-jetbrains)",
          fontSize: "11px",
          color: "var(--text-secondary)",
        }}>
          <span>All {rows.length} rows on this page are selected.</span>
          <button
            onClick={selectAllDocument}
            style={{ ...smallBtnStyle, color: "rgb(99,102,241)", borderColor: "rgba(99,102,241,0.4)" }}
          >
            Select all {totalRows.toLocaleString()} rows in document
          </button>
        </div>
      )}

      {/* Delete selected bar */}
      {editable && someSelected && (
        <div style={{
          display: "flex",
          alignItems: "center",
          gap: "10px",
          background: "rgba(239,68,68,0.08)",
          border: "1px solid rgba(239,68,68,0.3)",
          borderRadius: "6px",
          padding: "6px 12px",
          marginBottom: "10px",
          fontFamily: "var(--font-jetbrains)",
          fontSize: "11px",
          color: "var(--text-secondary)",
        }}>
          <span>
            {selectedCount.toLocaleString()} row{selectedCount !== 1 ? "s" : ""} selected
            {allDocumentSelected && " (entire document)"}
          </span>
          <button
            onClick={handleDeleteSelected}
            style={{ ...smallBtnStyle, background: "rgba(239,68,68,0.15)", color: "rgb(239,68,68)" }}
          >
            Delete selected
          </button>
          <button onClick={clearSelection} style={smallBtnStyle}>Clear</button>
        </div>
      )}

      {/* Table */}
      <div
        tabIndex={0}
        onKeyDown={onTableKeyDown}
        style={{ overflowX: "auto", borderRadius: "8px", border: "1px solid var(--border-dim)", outline: "none" }}
      >
        <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "var(--font-jetbrains)", fontSize: "11px" }}>
          <thead>
            <tr>
              {editable && (
                <th style={thStyle({ width: "32px" })}>
                  <input
                    type="checkbox"
                    checked={allDocumentSelected || allPageSelected}
                    ref={(el) => {
                      if (el) el.indeterminate = !allDocumentSelected && !allPageSelected && selectedRows.size > 0;
                    }}
                    onChange={toggleAll}
                    style={{ cursor: "pointer" }}
                  />
                </th>
              )}
              <th style={thStyle({ width: "48px" })}>#</th>
              {columns.map((col, colIdx) => {
                const pendingName = pendingColumnRenames?.get(colIdx);
                const displayName = pendingName !== undefined ? pendingName : col;
                const hasPendingRename = pendingName !== undefined && pendingName !== col;
                const isEditingThisHeader = editingHeader === colIdx;
                const colWidth =
                  columnWidths.get(colIdx) ?? getAutoColumnWidth(displayName);

                return (
                  <th
                    key={colIdx}
                    style={{
                      ...thStyle(),
                      cursor: editable && onColumnRename ? "text" : undefined,
                      background: hasPendingRename ? "rgba(251,146,60,0.08)" : "var(--bg-elevated)",
                      borderLeft: hasPendingRename ? "2px solid rgb(251,146,60)" : undefined,
                      position: "relative",
                      width: `${colWidth}px`,
                      minWidth: `${colWidth}px`,
                      maxWidth: `${colWidth}px`,
                    }}
                    onDoubleClick={() => startHeaderEdit(colIdx)}
                    title={
                      editable && onColumnRename
                        ? `${displayName} - Double-click to rename column`
                        : displayName
                    }
                  >
                    {isEditingThisHeader ? (
                      <input
                        ref={headerEditRef}
                        value={headerEditValue}
                        onChange={(e) => setHeaderEditValue(e.target.value)}
                        onKeyDown={(e) => {
                          if (e.key === "Enter") { e.preventDefault(); commitHeaderEdit(); }
                          if (e.key === "Escape") { e.preventDefault(); setEditingHeader(null); }
                          if (e.key === "Tab") { e.preventDefault(); commitHeaderEdit(); }
                        }}
                        onBlur={commitHeaderEdit}
                        onClick={(e) => e.stopPropagation()}
                        style={{
                          background: "var(--bg-base)",
                          border: "1px solid var(--accent)",
                          borderRadius: "4px",
                          padding: "2px 6px",
                          color: "var(--text-primary)",
                          fontFamily: "var(--font-jetbrains)",
                          fontSize: "10px",
                          fontWeight: 600,
                          letterSpacing: "0.06em",
                          outline: "none",
                          minWidth: "60px",
                          width: "100%",
                        }}
                      />
                    ) : (
                      <span style={{ display: "block", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                        {displayName}
                      </span>
                    )}
                    <div
                      role="separator"
                      aria-orientation="vertical"
                      title="Drag to resize column"
                      onMouseDown={(e) => startColumnResize(e, colIdx)}
                      style={{
                        position: "absolute",
                        top: 0,
                        right: 0,
                        width: "8px",
                        height: "100%",
                        cursor: "col-resize",
                        zIndex: 2,
                      }}
                    />
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr>
                <td colSpan={columns.length + 1 + (editable ? 1 : 0)} style={emptyCellStyle}>Loading…</td>
              </tr>
            ) : rows.length === 0 ? (
              <tr>
                <td colSpan={columns.length + 1 + (editable ? 1 : 0)} style={emptyCellStyle}>No data</td>
              </tr>
            ) : (
              rows.map((row, rowIdx) => {
                const absRowIdx = offset + rowIdx;
                const isSelected = allDocumentSelected || selectedRows.has(absRowIdx);
                const rowHeight = rowHeights.get(absRowIdx);
                return (
                  <tr
                    key={rowIdx}
                    style={{
                      borderBottom: rowIdx < rows.length - 1 ? "1px solid var(--border-dim)" : undefined,
                      background: isSelected ? "rgba(239,68,68,0.06)" : undefined,
                    }}
                  >
                    {editable && (
                      <td style={{ padding: "6px 8px", textAlign: "center", height: rowHeight ? `${rowHeight}px` : undefined }}>
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggleRow(absRowIdx)}
                          style={{ cursor: "pointer" }}
                        />
                      </td>
                    )}
                    <td style={{ padding: "6px 10px", color: "var(--text-tertiary)", fontSize: "10px", position: "relative", height: rowHeight ? `${rowHeight}px` : undefined, verticalAlign: "top" }}>
                      {absRowIdx + 1}
                      <div
                        role="separator"
                        aria-orientation="horizontal"
                        title="Drag to resize row"
                        onMouseDown={(e) => startRowResize(e, absRowIdx)}
                        style={{
                          position: "absolute",
                          left: "8px",
                          right: "8px",
                          bottom: "-2px",
                          height: "6px",
                          cursor: "row-resize",
                        }}
                      />
                    </td>
                    {row.map((cell, colIdx) => {
                      const cellKey = `${absRowIdx}:${colIdx}`;
                      const isEditing = editingCell?.absRowIdx === absRowIdx && editingCell?.colIdx === colIdx;
                      const isActive = activeCell?.absRowIdx === absRowIdx && activeCell?.colIdx === colIdx;
                      const pendingVal = pendingEdits?.get(cellKey);
                      const displayVal = pendingVal !== undefined ? pendingVal : cell;
                      const isMatch = matchSet.has(cellKey);
                      const isCurrentMatch = currentMatchKey === cellKey;
                      const hasPending = pendingVal !== undefined && pendingVal !== cell;
                      const colWidth =
                        columnWidths.get(colIdx) ??
                        getAutoColumnWidth(
                          pendingColumnRenames?.get(colIdx) ?? columns[colIdx] ?? ""
                        );

                      return (
                        <td
                          key={colIdx}
                          ref={(el) => {
                            if (el) cellRefs.current.set(cellKey, el);
                            else cellRefs.current.delete(cellKey);
                          }}
                          tabIndex={isActive ? 0 : -1}
                          onFocus={() => setActiveCell({ absRowIdx, colIdx })}
                          onClick={() => {
                            setActiveCell({ absRowIdx, colIdx });
                            if (!isEditing) startEdit(absRowIdx, colIdx, displayVal);
                          }}
                          style={{
                            padding: isEditing ? "2px 4px" : "6px 10px",
                            color: "var(--text-secondary)",
                            width: `${colWidth}px`,
                            minWidth: `${colWidth}px`,
                            maxWidth: `${colWidth}px`,
                            overflow: isEditing ? "visible" : "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: isEditing ? "normal" : "nowrap",
                            height: rowHeight ? `${rowHeight}px` : undefined,
                            verticalAlign: "top",
                            cursor: editable ? "text" : "default",
                            background: isCurrentMatch
                              ? "rgba(234,179,8,0.35)"
                              : isMatch
                              ? "rgba(234,179,8,0.15)"
                              : hasPending
                              ? "rgba(251,146,60,0.08)"
                              : undefined,
                            borderLeft: hasPending ? "2px solid rgb(251,146,60)" : undefined,
                            boxShadow: isActive ? "inset 0 0 0 1px var(--accent)" : undefined,
                            position: "relative",
                            outline: "none",
                          }}
                          title={isEditing ? undefined : displayVal}
                        >
                          {isEditing ? (
                            <input
                              ref={editInputRef}
                              value={editValue}
                              onChange={(e) => setEditValue(e.target.value)}
                              onKeyDown={(e) => {
                                if (e.key === "Enter") { e.preventDefault(); commitEdit(); }
                                if (e.key === "Escape") { e.preventDefault(); setEditingCell(null); }
                                if (e.key === "Tab") { e.preventDefault(); commitEdit(); }
                              }}
                              onBlur={commitEdit}
                              style={{
                                width: "100%",
                                minWidth: "80px",
                                background: "var(--bg-base)",
                                border: "1px solid var(--accent)",
                                borderRadius: "4px",
                                padding: "2px 6px",
                                color: "var(--text-primary)",
                                fontFamily: "var(--font-jetbrains)",
                                fontSize: "11px",
                                outline: "none",
                              }}
                            />
                          ) : (
                            displayVal
                          )}
                        </td>
                      );
                    })}
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      <div className="data-table__pagination" style={{ marginTop: "12px" }}>
        <div className="data-table__pagination-controls">
          {/* First page */}
          <button
            onClick={() => onPageChange(0)}
            disabled={offset === 0}
            style={pageBtnStyle(offset === 0)}
            title="First page"
          >
            «
          </button>
          <button
            onClick={() => onPageChange(Math.max(0, offset - limit))}
            disabled={offset === 0}
            style={pageBtnStyle(offset === 0)}
          >
            Prev
          </button>
          {/* Page jump input */}
          <input
            type="number"
            min={1}
            max={totalPages}
            value={pageInput}
            onChange={(e) => setPageInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") commitPageJump(); }}
            onBlur={commitPageJump}
            style={{
              width: "52px",
              textAlign: "center",
              background: "var(--bg-elevated)",
              border: "1px solid var(--border)",
              borderRadius: "6px",
              padding: "4px 6px",
              color: "var(--text-primary)",
              fontFamily: "var(--font-jetbrains)",
              fontSize: "11px",
              MozAppearance: "textfield" as never,
            }}
          />
          <span style={{ fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--text-tertiary)" }}>
            / {totalPages}
          </span>
          <button
            onClick={() => onPageChange(offset + limit)}
            disabled={offset + limit >= totalRows}
            style={pageBtnStyle(offset + limit >= totalRows)}
          >
            Next
          </button>
          {/* Last page */}
          <button
            onClick={() => onPageChange((totalPages - 1) * limit)}
            disabled={currentPage === totalPages}
            style={pageBtnStyle(currentPage === totalPages)}
            title="Last page"
          >
            »
          </button>
        </div>
        {onLimitChange && (
          <select
            className="data-table__pagination-limit"
            value={limit}
            onChange={(e) => onLimitChange(Number(e.target.value))}
            style={selectStyle}
          >
            {[25, 50, 100].map((n) => (
              <option key={n} value={n}>{n} rows</option>
            ))}
          </select>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function getCellDisplayValue(
  rowIdx: number,
  colIdx: number,
  row: string[],
  pendingEdits: Map<string, string> | undefined,
  offset: number
): string {
  const absRowIdx = offset + rowIdx;
  const key = `${absRowIdx}:${colIdx}`;
  const pending = pendingEdits?.get(key);
  return pending !== undefined ? pending : (row[colIdx] ?? "");
}

function escapeRegex(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// ---------------------------------------------------------------------------
// Style helpers
// ---------------------------------------------------------------------------

const inputStyle: React.CSSProperties = {
  background: "var(--bg-base)",
  border: "1px solid var(--border)",
  borderRadius: "6px",
  padding: "5px 10px",
  color: "var(--text-primary)",
  fontSize: "11px",
  fontFamily: "var(--font-jetbrains)",
  flex: 1,
  minWidth: "120px",
  maxWidth: "240px",
  outline: "none",
};

const selectStyle: React.CSSProperties = {
  background: "var(--bg-elevated)",
  border: "1px solid var(--border)",
  borderRadius: "6px",
  padding: "4px 8px",
  color: "var(--text-secondary)",
  fontSize: "11px",
  fontFamily: "var(--font-jetbrains)",
  maxWidth: "200px",
};

const smallBtnStyle: React.CSSProperties = {
  background: "var(--bg-elevated)",
  border: "1px solid var(--border)",
  borderRadius: "6px",
  padding: "3px 8px",
  color: "var(--text-secondary)",
  fontSize: "11px",
  fontFamily: "var(--font-jetbrains)",
  cursor: "pointer",
  whiteSpace: "nowrap",
};

function thStyle(extra?: React.CSSProperties): React.CSSProperties {
  return {
    background: "var(--bg-elevated)",
    padding: "8px 10px",
    textAlign: "left",
    color: "var(--text-tertiary)",
    fontSize: "10px",
    fontWeight: 600,
    letterSpacing: "0.06em",
    whiteSpace: "nowrap",
    borderBottom: "1px solid var(--border-dim)",
    ...extra,
  };
}

function pageBtnStyle(disabled: boolean): React.CSSProperties {
  return {
    background: "var(--bg-elevated)",
    border: "1px solid var(--border)",
    borderRadius: "6px",
    padding: "4px 10px",
    color: disabled ? "var(--text-tertiary)" : "var(--text-secondary)",
    fontSize: "11px",
    fontFamily: "var(--font-jetbrains)",
    cursor: disabled ? "default" : "pointer",
    opacity: disabled ? 0.5 : 1,
  };
}

const emptyCellStyle: React.CSSProperties = {
  padding: "32px",
  textAlign: "center",
  color: "var(--text-tertiary)",
};
