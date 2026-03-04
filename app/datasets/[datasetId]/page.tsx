"use client";

import { useEffect, useState, use, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import {
  fetchDatasetDetail,
  fetchDatasetPreview,
  fetchDatasetVersions,
  fetchDatasetBranches,
  updateDatasetMetadata,
  deleteDataset,
  updateDatasetCells,
  addDatasetRows,
  deleteDatasetRows,
  renameDatasetColumns,
  createDatasetBranch,
  updateDatasetBranch,
  deleteDatasetBranch,
  setDefaultBranch,
  updateVersionMetadata,
  deleteDatasetVersion,
  copyDatasetVersion,
  moveDatasetVersion,
  restoreDatasetVersion,
  setDatasetVersionAsDefault,
  CellChange,
  DatasetSummary,
  DatasetPreview,
  DatasetVersion,
  DatasetBranch,
} from "@/app/lib/api";
import { DataTable } from "@/app/components/DataTable";
import { TagInput } from "@/app/components/TagInput";
import {
  formatLocalizedDate,
  formatLocalizedDateTime,
  useDateTimeLocale,
} from "@/app/lib/i18n/date-time";

type Tab = "preview" | "schema" | "versions" | "branches";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// ---------------------------------------------------------------------------
// Shared inline styles
// ---------------------------------------------------------------------------
const btnBase = {
  borderRadius: "var(--radius-unified)",
  padding: "5px 12px",
  fontSize: "11px",
  fontFamily: "var(--font-jetbrains)",
  cursor: "pointer" as const,
  border: "1px solid var(--border)",
} as const;

const inputStyle = {
  background: "var(--bg-elevated)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius-unified)",
  padding: "6px 10px",
  color: "var(--text-primary)",
  fontSize: "12px",
  fontFamily: "var(--font-jetbrains)",
  width: "100%",
  boxSizing: "border-box" as const,
} as const;

const selectStyle = {
  background: "var(--bg-elevated)",
  border: "1px solid var(--border)",
  borderRadius: "var(--radius-unified)",
  padding: "4px 8px",
  color: "var(--text-secondary)",
  fontSize: "11px",
  fontFamily: "var(--font-jetbrains)",
} as const;
// Main page component
// ---------------------------------------------------------------------------
export default function DatasetDetailPage({
  params,
}: {
  params: Promise<{ datasetId: string }>;
}) {
  const { datasetId } = use(params);
  const router = useRouter();
  const dateTimeLocale = useDateTimeLocale();

  const [ds, setDs] = useState<DatasetSummary | null>(null);
  const [tab, setTab] = useState<Tab>("preview");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Branches
  const [branches, setBranches] = useState<DatasetBranch[]>([]);
  const [activeBranchId, setActiveBranchId] = useState<string | null>(null);

  // Versions (all or filtered by branch)
  const [versions, setVersions] = useState<DatasetVersion[]>([]);

  // Preview state
  const [preview, setPreview] = useState<DatasetPreview | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewOffset, setPreviewOffset] = useState(0);
  const [previewLimit, setPreviewLimit] = useState(50);
  const [previewVersion, setPreviewVersion] = useState<number | null>(null);

  // Metadata edit state
  const [editing, setEditing] = useState(false);
  const [editName, setEditName] = useState("");
  const [editDesc, setEditDesc] = useState("");
  const [editTags, setEditTags] = useState<string[]>([]);
  const [metaSaving, setMetaSaving] = useState(false);

  // Delete dataset
  const [confirmDelete, setConfirmDelete] = useState(false);

  // --- Data editing (Phase 6) ---
  const [pendingEdits, setPendingEdits] = useState(new Map<string, string>());
  const [pendingDeleteRows, setPendingDeleteRows] = useState<number[]>([]);
  const [pendingNewRows, setPendingNewRows] = useState<Record<string, string>[]>([]);
  const [pendingColumnRenames, setPendingColumnRenames] = useState(new Map<number, string>());
  const [showVersionDialog, setShowVersionDialog] = useState(false);
  const [versionReason, setVersionReason] = useState("");
  const [versionAuthor, setVersionAuthor] = useState("");
  const [dataSaving, setDataSaving] = useState(false);

  // --- Branch management ---
  const [showCreateBranch, setShowCreateBranch] = useState(false);
  const [newBranchName, setNewBranchName] = useState("");
  const [newBranchDesc, setNewBranchDesc] = useState("");
  const [newBranchFromVersionId, setNewBranchFromVersionId] = useState<string>("");
  const [newBranchAuthor, setNewBranchAuthor] = useState("");
  const [branchSaving, setBranchSaving] = useState(false);

  // Edit branch
  const [editingBranch, setEditingBranch] = useState<DatasetBranch | null>(null);
  const [editBranchName, setEditBranchName] = useState("");
  const [editBranchDesc, setEditBranchDesc] = useState("");

  // Delete branch confirm
  const [confirmDeleteBranch, setConfirmDeleteBranch] = useState<string | null>(null);

  // --- Version management ---
  const [editingVersionId, setEditingVersionId] = useState<string | null>(null);
  const [editVersionReason, setEditVersionReason] = useState("");
  const [confirmDeleteVersionId, setConfirmDeleteVersionId] = useState<string | null>(null);
  const [versionActionSaving, setVersionActionSaving] = useState(false);
  const [showCopyVersionDialog, setShowCopyVersionDialog] = useState(false);
  const [copySourceVersionId, setCopySourceVersionId] = useState<string | null>(null);
  const [copyVersionReason, setCopyVersionReason] = useState("");
  const [copyVersionAuthor, setCopyVersionAuthor] = useState("");
  const [copySaving, setCopySaving] = useState(false);
  const [showMoveVersionDialog, setShowMoveVersionDialog] = useState(false);
  const [moveSourceVersionId, setMoveSourceVersionId] = useState<string | null>(null);
  const [moveSourceVersionReason, setMoveSourceVersionReason] = useState("");
  const [moveTargetBranchId, setMoveTargetBranchId] = useState<string | null>(null);
  const [moveSaving, setMoveSaving] = useState(false);
  const pendingNewRowsRef = useRef<HTMLDivElement>(null);
  const shouldScrollToNewRowsRef = useRef(false);
  const saveVersionDialogTitleRef = useRef<HTMLHeadingElement>(null);

  const totalPendingChanges =
    pendingEdits.size + pendingDeleteRows.length + pendingNewRows.length + pendingColumnRenames.size;

  const activeBranch = branches.find((b) => b.id === activeBranchId) ?? null;
  const defaultBranchId = ds?.default_branch_id ?? null;

  // The "current version" for the active branch head
  const activeBranchHead = activeBranchId
    ? (versions.find((v) => v.branch_id === activeBranchId) ?? null) // versions ordered DESC so first = head
    : null;
  const currentVersionForBranch = activeBranchHead?.version ?? ds?.current_version ?? null;

  // Read-only if viewing older version
  const isReadOnlyVersion =
    previewVersion !== null &&
    currentVersionForBranch !== null &&
    previewVersion !== currentVersionForBranch;

  // ---------------------------------------------------------------------------
  // Load on mount
  // ---------------------------------------------------------------------------
  const loadBranches = useCallback(async () => {
    try {
      const brs = await fetchDatasetBranches(datasetId);
      setBranches(brs);
      return brs;
    } catch {
      return [];
    }
  }, [datasetId]);

  useEffect(() => {
    fetchDatasetDetail(datasetId)
      .then(async (d) => {
        setDs(d);
        setEditName(d.name);
        setEditDesc(d.description);
        setEditTags(d.tags);
        setLoading(false);

        // Load branches and set active to default
        const brs = await loadBranches();
        const defaultBr = brs.find((b) => b.is_default) ?? brs[0] ?? null;
        if (defaultBr) setActiveBranchId(defaultBr.id);
      })
      .catch((e) => {
        setError(e.message);
        setLoading(false);
      });
  }, [datasetId, loadBranches]);

  // Reload versions when branch selection changes (fetches branch-specific versions)
  useEffect(() => {
    fetchDatasetVersions(datasetId, activeBranchId ?? undefined)
      .then(setVersions)
      .catch(() => {});
    // Reset version selection when branch changes
    setPreviewVersion(null);
    setPendingEdits(new Map());
    setPendingDeleteRows([]);
    setPendingNewRows([]);
    setPendingColumnRenames(new Map());
  }, [datasetId, activeBranchId]);

  // When entering branches tab, fetch all versions to show correct counts for each branch
  useEffect(() => {
    if (tab === "branches") {
      fetchDatasetVersions(datasetId, undefined)
        .then(setVersions)
        .catch(() => {});
    }
  }, [datasetId, tab]);

  // ---------------------------------------------------------------------------
  // Load preview
  // ---------------------------------------------------------------------------
  const loadPreview = useCallback(() => {
    setPreviewLoading(true);
    fetchDatasetPreview(datasetId, {
      version: previewVersion ?? undefined,
      branch_id: previewVersion == null && activeBranchId ? activeBranchId : undefined,
      offset: previewOffset,
      limit: previewLimit,
    })
      .then(setPreview)
      .catch((e) => setError(e.message))
      .finally(() => setPreviewLoading(false));
  }, [datasetId, previewOffset, previewLimit, previewVersion, activeBranchId]);

  useEffect(() => {
    if (tab === "preview") loadPreview();
  }, [tab, loadPreview]);

  // Reset offset on version change
  useEffect(() => {
    setPreviewOffset(0);
    setPendingEdits(new Map());
    setPendingDeleteRows([]);
    setPendingNewRows([]);
    setPendingColumnRenames(new Map());
  }, [previewVersion]);

  // ---------------------------------------------------------------------------
  // Metadata save
  // ---------------------------------------------------------------------------
  async function handleMetaSave() {
    setMetaSaving(true);
    try {
      const updated = await updateDatasetMetadata(datasetId, {
        name: editName,
        description: editDesc,
        tags: editTags,
      });
      setDs(updated);
      setEditing(false);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setMetaSaving(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Data version save
  // ---------------------------------------------------------------------------
  async function handleDataSave() {
    if (totalPendingChanges === 0) return;
    setDataSaving(true);
    setError(null);
    try {
      const reason = versionReason.trim() || "manual edit";
      const author = versionAuthor.trim();
      const branchId = activeBranchId ?? undefined;

      if (pendingColumnRenames.size > 0 && preview) {
        const renames: Record<string, string> = {};
        for (const [colIdx, newName] of pendingColumnRenames.entries()) {
          const oldName = preview.columns[colIdx];
          if (oldName !== undefined && newName !== oldName) {
            renames[oldName] = newName;
          }
        }
        if (Object.keys(renames).length > 0) {
          await renameDatasetColumns(datasetId, renames, `${reason} — rename columns`, author, branchId);
        }
        setPendingColumnRenames(new Map());
      }

      if (pendingDeleteRows.length > 0) {
        await deleteDatasetRows(datasetId, pendingDeleteRows, `${reason} — deleted rows`, author, branchId);
        setPendingDeleteRows([]);
      }

      if (pendingEdits.size > 0) {
        const changes: CellChange[] = [];
        for (const [key, value] of pendingEdits.entries()) {
          const [rowStr, colStr] = key.split(":");
          const absRowIdx = parseInt(rowStr);
          const colIdx = parseInt(colStr);
          const col = preview?.columns[colIdx];
          if (col !== undefined) changes.push({ row_idx: absRowIdx, col, value });
        }
        if (changes.length > 0) {
          await updateDatasetCells(datasetId, changes, `${reason} — cell edits`, author, branchId);
        }
        setPendingEdits(new Map());
      }

      if (pendingNewRows.length > 0) {
        await addDatasetRows(datasetId, pendingNewRows, `${reason} — added rows`, author, branchId);
        setPendingNewRows([]);
      }

      const [updated, newVersions] = await Promise.all([
        fetchDatasetDetail(datasetId),
        fetchDatasetVersions(datasetId, activeBranchId ?? undefined),
      ]);
      setDs(updated);
      setVersions(newVersions);
      setPreviewVersion(null);
      setShowVersionDialog(false);
      setVersionReason("");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setDataSaving(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Cell edit handlers
  // ---------------------------------------------------------------------------
  const handleCellEdit = useCallback((absRowIdx: number, colIdx: number, value: string) => {
    setPendingEdits((prev) => {
      const next = new Map(prev);
      next.set(`${absRowIdx}:${colIdx}`, value);
      return next;
    });
  }, []);

  const handleDeleteRows = useCallback((indices: number[]) => {
    setPendingDeleteRows((prev) => Array.from(new Set([...prev, ...indices])));
  }, []);

  const addNewRow = () => {
    if (!preview) return;
    const blank: Record<string, string> = {};
    for (const col of preview.columns) blank[col] = "";
    shouldScrollToNewRowsRef.current = true;
    setPendingNewRows((prev) => [...prev, blank]);
  };

  const openSaveVersionDialog = useCallback(() => {
    setShowVersionDialog(true);
  }, []);

  const handlePreviewLimitChange = useCallback((nextLimit: number) => {
    setPreviewLimit(nextLimit);
    setPreviewOffset(0);
  }, []);

  useEffect(() => {
    if (!showVersionDialog) return;
    window.scrollTo({ top: 0, behavior: "smooth" });
    const timer = window.setTimeout(() => {
      saveVersionDialogTitleRef.current?.focus();
    }, 80);
    return () => window.clearTimeout(timer);
  }, [showVersionDialog]);

  useEffect(() => {
    if (!shouldScrollToNewRowsRef.current || pendingNewRows.length === 0) return;
    shouldScrollToNewRowsRef.current = false;
    const timer = window.setTimeout(() => {
      pendingNewRowsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      const firstInput = pendingNewRowsRef.current?.querySelector("tbody tr:last-child input");
      if (firstInput instanceof HTMLInputElement) {
        firstInput.focus();
      }
    }, 60);
    return () => window.clearTimeout(timer);
  }, [pendingNewRows.length]);

  const updateNewRowCell = (rowIdx: number, col: string, value: string) => {
    setPendingNewRows((prev) => prev.map((r, i) => (i === rowIdx ? { ...r, [col]: value } : r)));
  };

  const removeNewRow = (rowIdx: number) => {
    setPendingNewRows((prev) => prev.filter((_, i) => i !== rowIdx));
  };

  const discardAllChanges = () => {
    setPendingEdits(new Map());
    setPendingDeleteRows([]);
    setPendingNewRows([]);
    setPendingColumnRenames(new Map());
  };

  const handleColumnRename = useCallback((colIdx: number, newName: string) => {
    setPendingColumnRenames((prev) => {
      const next = new Map(prev);
      next.set(colIdx, newName);
      return next;
    });
  }, []);

  // ---------------------------------------------------------------------------
  // Delete dataset
  // ---------------------------------------------------------------------------
  async function handleDelete() {
    try {
      const result = await deleteDataset(datasetId, true);
      if (result.deleted) router.push("/datasets");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Delete failed");
    }
  }

  // ---------------------------------------------------------------------------
  // Branch operations
  // ---------------------------------------------------------------------------
  async function handleCreateBranch() {
    setBranchSaving(true);
    setError(null);
    try {
      await createDatasetBranch(datasetId, {
        name: newBranchName,
        description: newBranchDesc,
        from_version_id: newBranchFromVersionId || undefined,
        author: newBranchAuthor || undefined,
      });
      const brs = await loadBranches();
      const created = brs.find((b) => b.name === newBranchName);
      if (created) setActiveBranchId(created.id);
      setShowCreateBranch(false);
      setNewBranchName("");
      setNewBranchDesc("");
      setNewBranchFromVersionId("");
      setNewBranchAuthor("");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Create branch failed");
    } finally {
      setBranchSaving(false);
    }
  }

  async function handleUpdateBranch() {
    if (!editingBranch) return;
    setBranchSaving(true);
    setError(null);
    try {
      await updateDatasetBranch(datasetId, editingBranch.id, {
        name: editBranchName || undefined,
        description: editBranchDesc,
      });
      await loadBranches();
      setEditingBranch(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Update branch failed");
    } finally {
      setBranchSaving(false);
    }
  }

  async function handleDeleteBranch(branchId: string) {
    setError(null);
    try {
      await deleteDatasetBranch(datasetId, branchId);
      await loadBranches();
      if (activeBranchId === branchId) {
        const remaining = branches.filter((b) => b.id !== branchId && !b.is_deleted);
        setActiveBranchId(remaining[0]?.id ?? null);
      }
      setConfirmDeleteBranch(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Delete branch failed");
      setConfirmDeleteBranch(null);
    }
  }

  async function handleSetDefaultBranch(branchId: string) {
    setError(null);
    try {
      const result = await setDefaultBranch(datasetId, branchId);
      const updated = await fetchDatasetDetail(datasetId);
      setDs(updated);
      await loadBranches();
      return result;
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Set default failed");
    }
  }

  // ---------------------------------------------------------------------------
  // Version operations
  // ---------------------------------------------------------------------------
  async function handleUpdateVersionReason() {
    if (!editingVersionId) return;
    setVersionActionSaving(true);
    setError(null);
    try {
      await updateVersionMetadata(datasetId, editingVersionId, editVersionReason);
      await fetchDatasetVersions(datasetId, activeBranchId ?? undefined).then(setVersions);
      setEditingVersionId(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Update version failed");
    } finally {
      setVersionActionSaving(false);
    }
  }

  async function handleDeleteVersion(versionId: string) {
    setVersionActionSaving(true);
    setError(null);
    try {
      await deleteDatasetVersion(datasetId, versionId);
      const [updated, newVersions] = await Promise.all([
        fetchDatasetDetail(datasetId),
        fetchDatasetVersions(datasetId, activeBranchId ?? undefined),
      ]);
      setDs(updated);
      setVersions(newVersions);
      if (previewVersion !== null) {
        const stillExists = newVersions.some((v) => v.version === previewVersion);
        if (!stillExists) setPreviewVersion(null);
      }
      setConfirmDeleteVersionId(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Delete version failed");
      setConfirmDeleteVersionId(null);
    } finally {
      setVersionActionSaving(false);
    }
  }

  async function handleCopyVersion(sourceVersionId: string) {
    setCopySaving(true);
    setError(null);
    try {
      await copyDatasetVersion(
        datasetId,
        sourceVersionId,
        copyVersionReason || "Copied version",
        copyVersionAuthor || undefined,
        activeBranchId ?? undefined
      );
      const [updated, newVersions] = await Promise.all([
        fetchDatasetDetail(datasetId),
        fetchDatasetVersions(datasetId, activeBranchId ?? undefined),
      ]);
      setDs(updated);
      setVersions(newVersions);
      setShowCopyVersionDialog(false);
      setCopySourceVersionId(null);
      setCopyVersionReason("");
      setCopyVersionAuthor("");
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Copy version failed");
    } finally {
      setCopySaving(false);
    }
  }

  async function handleMoveVersion(sourceVersionId: string) {
    const sourceVersion = versions.find((v) => v.id === sourceVersionId);
    if (sourceVersion && moveTargetBranchId === sourceVersion.branch_id) {
      setError("Cannot move a version to the same branch");
      return;
    }
    if (!moveTargetBranchId) {
      setError("Please select a target branch");
      return;
    }
    setMoveSaving(true);
    setError(null);
    try {
      await moveDatasetVersion(
        datasetId,
        sourceVersionId,
        moveTargetBranchId,
      );
      const [updated, newVersions] = await Promise.all([
        fetchDatasetDetail(datasetId),
        fetchDatasetVersions(datasetId, activeBranchId ?? undefined),
      ]);
      setDs(updated);
      setVersions(newVersions);
      setShowMoveVersionDialog(false);
      setMoveSourceVersionId(null);
      setMoveSourceVersionReason("");
      setMoveTargetBranchId(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Move version failed");
    } finally {
      setMoveSaving(false);
    }
  }

  async function handleRestoreVersion(versionId: string, versionNumber: number) {
    setVersionActionSaving(true);
    setError(null);
    try {
      await restoreDatasetVersion(datasetId, versionId, `Restored from v${versionNumber}`);
      const [updated, newVersions] = await Promise.all([
        fetchDatasetDetail(datasetId),
        fetchDatasetVersions(datasetId, activeBranchId ?? undefined),
      ]);
      setDs(updated);
      setVersions(newVersions);
      setPreviewVersion(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Restore version failed");
    } finally {
      setVersionActionSaving(false);
    }
  }

  async function handleSetVersionAsDefault(versionId: string) {
    setVersionActionSaving(true);
    setError(null);
    try {
      const ver = await setDatasetVersionAsDefault(datasetId, versionId);
      const [updated, newVersions] = await Promise.all([
        fetchDatasetDetail(datasetId),
        fetchDatasetVersions(datasetId, activeBranchId ?? undefined),
      ]);
      setDs(updated);
      setVersions(newVersions);
      await loadBranches();
      if (ver.branch_id) {
        setActiveBranchId(ver.branch_id);
      }
      setPreviewVersion(null);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Set as default failed");
    } finally {
      setVersionActionSaving(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  if (loading) {
    return (
      <div style={{ padding: "48px 32px", color: "var(--text-tertiary)", fontFamily: "var(--font-jetbrains)", fontSize: "12px" }}>
        Loading…
      </div>
    );
  }

  if (!ds) {
    return (
      <div style={{ padding: "48px 32px", color: "var(--error)", fontFamily: "var(--font-jetbrains)", fontSize: "12px" }}>
        Dataset not found.
      </div>
    );
  }

  const tabStyle = (t: Tab) => ({
    background: tab === t ? "var(--bg-elevated)" : "transparent",
    border: tab === t ? "1px solid var(--border)" : "1px solid transparent",
    borderBottom: tab === t ? "1px solid var(--bg-elevated)" : "1px solid transparent",
    borderRadius: "var(--radius-unified) var(--radius-unified) 0 0",
    padding: "6px 14px",
    color: tab === t ? "var(--text-primary)" : "var(--text-tertiary)",
    fontSize: "11px",
    fontFamily: "var(--font-jetbrains)",
    cursor: "pointer" as const,
    fontWeight: tab === t ? 500 : 400,
  });

  // Versions that belong to the active branch
  const branchVersions = activeBranchId
    ? versions.filter((v) => v.branch_id === activeBranchId)
    : versions;
  const branchHead = branchVersions[0] ?? null; // versions are DESC
  const selectedPreviewVersion = previewVersion !== null
    ? (branchVersions.find((v) => v.version === previewVersion) ?? null)
    : null;
  const moveSourceVersion = moveSourceVersionId
    ? (versions.find((v) => v.id === moveSourceVersionId) ?? null)
    : null;
  const moveTargetBranches = branches.filter(
    (b) => !b.is_deleted && b.id !== moveSourceVersion?.branch_id
  );

  return (
    <div className="page-shell page-standard page-shell--md animate-fade-up">
      {/* Back link */}
      <div style={{ marginBottom: "6px" }}>
        <button
          onClick={() => router.push("/datasets")}
          style={{ background: "none", border: "none", color: "var(--text-tertiary)", fontFamily: "var(--font-jetbrains)", fontSize: "11px", cursor: "pointer", padding: 0, marginBottom: "8px" }}
        >
          ← Datasets
        </button>
      </div>

      {error && (
        <div className="rounded-lg" style={{ background: "var(--error-dim)", border: "1px solid var(--error)", padding: "10px 16px", color: "var(--error)", fontSize: "12px", fontFamily: "var(--font-jetbrains)", marginBottom: "12px" }}>
          {error}
          <button onClick={() => setError(null)} style={{ float: "right", background: "none", border: "none", color: "var(--error)", cursor: "pointer", fontSize: "12px" }}>✕</button>
        </div>
      )}

      {/* Dataset info card */}
      <div className="rounded-xl dataset-detail__hero" style={{ background: "var(--bg-surface)", border: "1px solid var(--border-dim)", padding: "20px 24px", marginBottom: "20px" }}>
        {editing ? (
          <div className="flex flex-col gap-3">
            <input value={editName} onChange={(e) => setEditName(e.target.value)} style={inputStyle} placeholder="Dataset name" />
            <textarea value={editDesc} onChange={(e) => setEditDesc(e.target.value)} style={{ ...inputStyle, resize: "vertical" as const }} rows={2} placeholder="Description" />
            <TagInput tags={editTags} onChange={setEditTags} />
            <div className="flex items-center gap-2">
              <button onClick={handleMetaSave} disabled={metaSaving} style={{ background: "var(--gold)", color: "#08080B", border: "none", borderRadius: "var(--radius-unified)", padding: "5px 14px", fontSize: "11px", fontWeight: 600, fontFamily: "var(--font-syne)", cursor: "pointer" }}>
                {metaSaving ? "Saving…" : "Save"}
              </button>
              <button onClick={() => { setEditing(false); setEditName(ds.name); setEditDesc(ds.description); setEditTags(ds.tags); }}
                style={{ ...btnBase, background: "transparent", color: "var(--text-secondary)", fontFamily: "var(--font-syne)" }}>
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="dataset-detail__hero-header" style={{ marginBottom: "8px" }}>
              <div className="dataset-detail__hero-main">
                <h1 className="dataset-detail__hero-title" style={{ fontFamily: "var(--font-syne)", fontWeight: 700, fontSize: "20px", color: "var(--text-primary)", marginBottom: "4px" }}>{ds.name}</h1>
                {ds.description && <div className="dataset-detail__hero-description" style={{ fontSize: "13px", color: "var(--text-tertiary)", marginBottom: "8px" }}>{ds.description}</div>}
                <div className="dataset-detail__hero-tags flex flex-wrap gap-1.5" style={{ marginBottom: "8px" }}>
                  {ds.tags.map((tag) => (
                    <span key={tag} className="rounded" style={{ background: "var(--gold-faint)", border: "1px solid var(--gold-muted)", color: "var(--gold)", padding: "1px 6px", fontSize: "9px", fontFamily: "var(--font-jetbrains)" }}>{tag}</span>
                  ))}
                </div>
              </div>
              <div className="dataset-detail__hero-actions">
                <button
                  type="button"
                  className="dataset-detail__hero-action-btn"
                  onClick={() => setEditing(true)}
                  style={{ ...btnBase, background: "var(--bg-elevated)", color: "var(--text-secondary)" }}
                >
                  Edit
                </button>
                {confirmDelete ? (
                  <div className="dataset-detail__hero-delete-actions">
                    <button
                      type="button"
                      className="dataset-detail__hero-action-btn dataset-detail__hero-action-btn--danger"
                      onClick={handleDelete}
                      style={{ ...btnBase, background: "var(--error-dim)", border: "1px solid var(--error)", color: "var(--error)" }}
                    >
                      Confirm Delete
                    </button>
                    <button
                      type="button"
                      className="dataset-detail__hero-action-btn dataset-detail__hero-action-btn--muted"
                      onClick={() => setConfirmDelete(false)}
                      style={{ ...btnBase, background: "transparent", color: "var(--text-tertiary)" }}
                    >
                      Cancel
                    </button>
                  </div>
                ) : (
                  <button
                    type="button"
                    className="dataset-detail__hero-action-btn dataset-detail__hero-action-btn--muted"
                    onClick={() => setConfirmDelete(true)}
                    style={{ ...btnBase, background: "transparent", color: "var(--text-tertiary)" }}
                  >
                    Delete
                  </button>
                )}
              </div>
            </div>
            <div className="dataset-detail__hero-stats" style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)" }}>
              <span className="dataset-detail__hero-stat">{ds.row_count.toLocaleString()} rows</span>
              <span className="dataset-detail__hero-stat">{ds.schema_info.length} columns</span>
              <span className="dataset-detail__hero-stat">{formatBytes(ds.file_size_bytes)}</span>
              <span className="dataset-detail__hero-stat">v{ds.current_version}</span>
              {ds.author && <span className="dataset-detail__hero-stat">by {ds.author}</span>}
              <span className="dataset-detail__hero-stat">{formatLocalizedDateTime(ds.created_at, dateTimeLocale)}</span>
            </div>

            <div className="dataset-detail__hero-cta-grid" style={{ marginTop: "12px" }}>
              <Link
                href={`/training?dataset_id=${datasetId}`}
                className="dataset-detail__hero-cta dataset-detail__hero-cta--primary"
                style={{
                  background: "var(--gold)",
                  color: "#08080B",
                  borderRadius: "var(--radius-unified)",
                  padding: "6px 12px",
                  fontSize: "11px",
                  fontWeight: 600,
                  fontFamily: "var(--font-syne)",
                  textDecoration: "none",
                }}
              >
                Start Training
              </Link>
              <Link
                href={`/analyses/new?dataset_id=${datasetId}`}
                className="dataset-detail__hero-cta dataset-detail__hero-cta--secondary"
                style={{
                  background: "var(--bg-elevated)",
                  color: "var(--text-secondary)",
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius-unified)",
                  padding: "6px 12px",
                  fontSize: "11px",
                  fontWeight: 600,
                  fontFamily: "var(--font-syne)",
                  textDecoration: "none",
                }}
              >
                Start Analysis
              </Link>
              <Link
                href={`/datasets/${datasetId}/analytics`}
                className="dataset-detail__hero-cta dataset-detail__hero-cta--ghost"
                style={{
                  background: "transparent",
                  color: "var(--text-secondary)",
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius-unified)",
                  padding: "6px 12px",
                  fontSize: "11px",
                  fontWeight: 600,
                  fontFamily: "var(--font-syne)",
                  textDecoration: "none",
                }}
              >
                View Analytics
              </Link>
            </div>

            {/* Branch selector */}
            {branches.length > 0 && (
              <div className="dataset-detail__branch-bar" style={{ marginTop: "12px", paddingTop: "12px", borderTop: "1px solid var(--border-dim)" }}>
                <span className="dataset-detail__branch-label" style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)" }}>Branch:</span>
                <select
                  className="dataset-detail__branch-select"
                  value={activeBranchId ?? ""}
                  onChange={(e) => setActiveBranchId(e.target.value || null)}
                  style={selectStyle}
                >
                  {branches.map((b) => (
                    <option key={b.id} value={b.id}>
                      {b.name}{b.is_default ? " ● default" : ""}
                    </option>
                  ))}
                </select>
                {activeBranch && !activeBranch.is_default && (
                  <button
                    type="button"
                    className="dataset-detail__branch-action"
                    onClick={() => handleSetDefaultBranch(activeBranch.id)}
                    style={{ ...btnBase, background: "var(--bg-elevated)", color: "var(--gold)", border: "1px solid var(--gold-muted)", fontSize: "10px" }}
                  >
                    Set as default
                  </button>
                )}
                {activeBranch?.description ? (
                  <span className="dataset-detail__branch-meta" style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)" }}>
                    {activeBranch.description ? `— ${activeBranch.description}` : ""}
                  </span>
                ) : null}
              </div>
            )}
          </>
        )}
      </div>

      {/* Tabs */}
      <div className="dataset-detail__tabs" style={{ borderBottom: "1px solid var(--border-dim)", marginBottom: "16px" }}>
        <button
          type="button"
          className={`dataset-detail__tab${tab === "preview" ? " is-active" : ""}`}
          style={tabStyle("preview")}
          onClick={() => setTab("preview")}
        >
          Preview
        </button>
        <button
          type="button"
          className={`dataset-detail__tab${tab === "schema" ? " is-active" : ""}`}
          style={tabStyle("schema")}
          onClick={() => setTab("schema")}
        >
          Schema
        </button>
        <button
          type="button"
          className={`dataset-detail__tab${tab === "versions" ? " is-active" : ""}`}
          style={tabStyle("versions")}
          onClick={() => setTab("versions")}
        >
          Versions
        </button>
        <button
          type="button"
          className={`dataset-detail__tab${tab === "branches" ? " is-active" : ""}`}
          style={tabStyle("branches")}
          onClick={() => setTab("branches")}
        >
          Branches
          {branches.length > 0 && (
            <span style={{ marginLeft: "5px", background: "var(--bg-base)", borderRadius: "var(--radius-unified)", padding: "1px 5px", fontSize: "9px", color: "var(--text-tertiary)" }}>
              {branches.length}
            </span>
          )}
        </button>
      </div>

      {/* ===== PREVIEW TAB ===== */}
      {tab === "preview" && (
        <>
          <div className="dataset-detail__preview-toolbar" style={{ marginBottom: "12px" }}>
            {branchVersions.length > 0 && (
              <div className="dataset-detail__preview-version">
                <span
                  className="dataset-detail__preview-label"
                  style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)" }}
                >
                  Version:
                </span>
                <select
                  className="dataset-detail__preview-select"
                  value={previewVersion ?? "latest"}
                  onChange={(e) => {
                    const v = e.target.value === "latest" ? null : parseInt(e.target.value);
                    setPreviewVersion(v);
                  }}
                  style={selectStyle}
                >
                  <option value="latest">
                    v{branchHead?.version ?? ds.current_version} — latest
                  </option>
                  {branchVersions.filter((v) => v.version !== branchHead?.version).map((v) => (
                    <option key={v.version} value={v.version}>
                      v{v.version} — {formatLocalizedDate(v.created_at, dateTimeLocale)} ({v.row_count.toLocaleString()} rows)
                    </option>
                  ))}
                </select>
              </div>
            )}
            {isReadOnlyVersion && (
              <div className="flex items-center gap-2">
                <span style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--warning)", background: "rgba(251,191,36,0.1)", border: "1px solid rgba(251,191,36,0.3)", borderRadius: "var(--radius-unified)", padding: "2px 8px" }}>
                  Read-only — viewing historical version
                </span>
                {selectedPreviewVersion && (
                  <button
                    type="button"
                    className="dataset-detail__preview-secondary-action"
                    onClick={() => handleRestoreVersion(selectedPreviewVersion.id, selectedPreviewVersion.version)}
                    disabled={versionActionSaving}
                    style={{ ...btnBase, background: "transparent", color: "var(--text-secondary)", fontSize: "10px", padding: "3px 10px", opacity: versionActionSaving ? 0.6 : 1 }}
                  >
                    Restore
                  </button>
                )}
              </div>
            )}
            {!isReadOnlyVersion && (
              <button
                type="button"
                className="dataset-detail__preview-action"
                onClick={addNewRow}
                style={{ marginLeft: "auto", ...btnBase, background: "var(--bg-elevated)", color: "var(--text-secondary)" }}
              >
                + Add Row
              </button>
            )}
          </div>

          {preview && (
            <DataTable
              storageKey={`dataset:${datasetId}:branch:${activeBranchId ?? "none"}`}
              columns={preview.columns}
              rows={preview.rows}
              totalRows={preview.total_rows}
              offset={previewOffset}
              limit={previewLimit}
              onPageChange={setPreviewOffset}
              onLimitChange={handlePreviewLimitChange}
              loading={previewLoading}
              editable={!isReadOnlyVersion}
              pendingEdits={pendingEdits}
              onCellEdit={isReadOnlyVersion ? undefined : handleCellEdit}
              onDeleteRows={isReadOnlyVersion ? undefined : handleDeleteRows}
              pendingColumnRenames={isReadOnlyVersion ? undefined : pendingColumnRenames}
              onColumnRename={isReadOnlyVersion ? undefined : handleColumnRename}
            />
          )}

          {/* Pending new rows */}
          {pendingNewRows.length > 0 && preview && (
            <div ref={pendingNewRowsRef} style={{ marginTop: "16px", border: "1px solid rgba(251,146,60,0.4)", borderRadius: "var(--radius-unified)", overflow: "hidden" }}>
              <div style={{ background: "rgba(251,146,60,0.08)", padding: "8px 12px", borderBottom: "1px solid rgba(251,146,60,0.2)", display: "flex", alignItems: "center", gap: "8px" }}>
                <span style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "rgb(251,146,60)" }}>
                  {pendingNewRows.length} new row{pendingNewRows.length !== 1 ? "s" : ""} to add
                </span>
                <button
                  onClick={addNewRow}
                  style={{ marginLeft: "auto", ...btnBase, background: "var(--bg-base)", color: "var(--text-secondary)", padding: "3px 10px", fontSize: "10px" }}
                >
                  + Add Row
                </button>
              </div>
              <div style={{ overflowX: "auto" }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "var(--font-jetbrains)", fontSize: "11px" }}>
                  <thead>
                    <tr>
                      <th style={{ background: "var(--bg-elevated)", padding: "6px 10px", textAlign: "left", color: "var(--text-tertiary)", fontSize: "10px", borderBottom: "1px solid var(--border-dim)", width: "32px" }}></th>
                      {preview.columns.map((col) => (
                        <th key={col} style={{ background: "var(--bg-elevated)", padding: "6px 10px", textAlign: "left", color: "var(--text-tertiary)", fontSize: "10px", borderBottom: "1px solid var(--border-dim)", whiteSpace: "nowrap" }}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {pendingNewRows.map((row, rowIdx) => (
                      <tr key={rowIdx} style={{ borderBottom: rowIdx < pendingNewRows.length - 1 ? "1px solid var(--border-dim)" : undefined }}>
                        <td style={{ padding: "4px 8px", textAlign: "center" }}>
                          <button onClick={() => removeNewRow(rowIdx)} style={{ background: "none", border: "none", color: "var(--error)", cursor: "pointer", fontSize: "12px", padding: "2px" }}>✕</button>
                        </td>
                        {preview.columns.map((col) => (
                          <td key={col} style={{ padding: "2px 4px" }}>
                            <input
                              value={row[col] ?? ""}
                              onChange={(e) => updateNewRowCell(rowIdx, col, e.target.value)}
                              style={{ width: "100%", minWidth: "60px", background: "var(--bg-base)", border: "1px solid var(--border)", borderRadius: "var(--radius-unified)", padding: "3px 6px", color: "var(--text-primary)", fontFamily: "var(--font-jetbrains)", fontSize: "11px", outline: "none" }}
                            />
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Pending changes bar */}
          {totalPendingChanges > 0 && (
            <div style={{ marginTop: "16px", display: "flex", alignItems: "center", gap: "12px", background: "rgba(251,146,60,0.08)", border: "1px solid rgba(251,146,60,0.35)", borderRadius: "var(--radius-unified)", padding: "10px 16px" }}>
              <span style={{ fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "rgb(251,146,60)" }}>
                ● {totalPendingChanges} unsaved change{totalPendingChanges !== 1 ? "s" : ""}
                {pendingColumnRenames.size > 0 && ` · ${pendingColumnRenames.size} column rename${pendingColumnRenames.size !== 1 ? "s" : ""}`}
                {pendingEdits.size > 0 && ` · ${pendingEdits.size} cell edit${pendingEdits.size !== 1 ? "s" : ""}`}
                {pendingDeleteRows.length > 0 && ` · ${pendingDeleteRows.length} row deletion${pendingDeleteRows.length !== 1 ? "s" : ""}`}
                {pendingNewRows.length > 0 && ` · ${pendingNewRows.length} new row${pendingNewRows.length !== 1 ? "s" : ""}`}
                {activeBranch && <span style={{ color: "rgba(251,146,60,0.7)" }}> → branch &quot;{activeBranch.name}&quot;</span>}
              </span>
              <button onClick={discardAllChanges} style={{ marginLeft: "auto", ...btnBase, background: "transparent", color: "var(--text-secondary)" }}>
                Discard
              </button>
              <button
                onClick={openSaveVersionDialog}
                style={{ background: "var(--gold)", color: "#08080B", border: "none", borderRadius: "var(--radius-unified)", padding: "5px 14px", fontSize: "11px", fontWeight: 600, fontFamily: "var(--font-syne)", cursor: "pointer" }}
              >
                Save as new version →
              </button>
            </div>
          )}
        </>
      )}

      {/* ===== SCHEMA TAB ===== */}
      {tab === "schema" && ds.schema_info.length > 0 && (
        <div style={{ borderRadius: "var(--radius-unified)", border: "1px solid var(--border-dim)", overflow: "hidden" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontFamily: "var(--font-jetbrains)", fontSize: "11px" }}>
            <thead>
              <tr>
                {["Column", "Type", "Unique", "Nulls", "Sample Values"].map((h) => (
                  <th key={h} style={{ background: "var(--bg-elevated)", padding: "8px 12px", textAlign: "left", color: "var(--text-tertiary)", fontSize: "10px", fontWeight: 600, letterSpacing: "0.06em", borderBottom: "1px solid var(--border-dim)" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ds.schema_info.map((col, i) => (
                <tr key={col.name} style={{ borderBottom: i < ds.schema_info.length - 1 ? "1px solid var(--border-dim)" : undefined }}>
                  <td style={{ padding: "6px 12px", color: "var(--text-primary)", fontWeight: 500 }}>{col.name}</td>
                  <td style={{ padding: "6px 12px", color: "var(--teal)" }}>{col.dtype}</td>
                  <td style={{ padding: "6px 12px", color: "var(--text-secondary)" }}>{col.n_unique}</td>
                  <td style={{ padding: "6px 12px", color: col.n_null > 0 ? "var(--warning)" : "var(--text-secondary)" }}>{col.n_null}</td>
                  <td style={{ padding: "6px 12px", color: "var(--text-tertiary)", maxWidth: "300px", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{col.sample_values.join(", ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ===== VERSIONS TAB ===== */}
      {tab === "versions" && (
        <div className="flex flex-col gap-2">
          {branchVersions.length === 0 ? (
            <div style={{ color: "var(--text-tertiary)", fontFamily: "var(--font-jetbrains)", fontSize: "12px", padding: "24px 0" }}>
              {activeBranch ? `No versions on branch "${activeBranch.name}".` : "Loading versions…"}
            </div>
          ) : (
            branchVersions.map((v, idx) => {
              const isHead = activeBranch?.head_version_id
                ? activeBranch.head_version_id === v.id
                : idx === 0;
              const isDefaultVersion =
                v.branch_id === ds.default_branch_id && v.version === ds.current_version;
              const isEditing = editingVersionId === v.id;
              const isConfirmDelete = confirmDeleteVersionId === v.id;
              return (
                <div key={v.id} className="rounded-lg" style={{ background: "var(--bg-elevated)", border: "1px solid var(--border-dim)", padding: "12px 16px" }}>
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <span style={{ fontFamily: "var(--font-syne)", fontSize: "13px", fontWeight: 600, color: "var(--text-primary)" }}>v{v.version}</span>
                      {isHead && (
                        <span style={{ background: "var(--gold-faint)", border: "1px solid var(--gold-muted)", color: "var(--gold)", padding: "1px 6px", fontSize: "9px", fontFamily: "var(--font-jetbrains)", borderRadius: "var(--radius-unified)" }}>head</span>
                      )}
                      {isDefaultVersion && (
                        <span style={{ background: "rgba(20,184,166,0.1)", border: "1px solid rgba(20,184,166,0.3)", color: "var(--teal)", padding: "1px 6px", fontSize: "9px", fontFamily: "var(--font-jetbrains)", borderRadius: "var(--radius-unified)" }}>default</span>
                      )}
                      {!isEditing && (
                        <span style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)" }}>{v.reason}</span>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => { setPreviewVersion(v.version); setTab("preview"); }}
                        style={{ ...btnBase, background: "var(--bg-base)", color: "var(--text-secondary)", fontSize: "10px", padding: "3px 10px" }}
                      >
                        View
                      </button>
                      <button
                        onClick={() => {
                          setEditingVersionId(v.id);
                          setEditVersionReason(v.reason);
                        }}
                        style={{ ...btnBase, background: "transparent", color: "var(--text-tertiary)", fontSize: "10px", padding: "3px 10px" }}
                      >
                        Edit
                      </button>
                      {!isHead && (
                        <button
                          onClick={() => setConfirmDeleteVersionId(v.id)}
                          style={{ ...btnBase, background: "transparent", color: "var(--error)", border: "1px solid rgba(239,68,68,0.3)", fontSize: "10px", padding: "3px 10px" }}
                        >
                          Delete
                        </button>
                      )}
                      <button
                        onClick={() => {
                          setCopySourceVersionId(v.id);
                          setCopyVersionReason(`Copy of ${v.reason}`);
                          setShowCopyVersionDialog(true);
                        }}
                        style={{ ...btnBase, background: "transparent", color: "var(--text-secondary)", fontSize: "10px", padding: "3px 10px" }}
                      >
                        Copy
                      </button>
                      <button
                        onClick={() => {
                          setMoveSourceVersionId(v.id);
                          setMoveSourceVersionReason(v.reason);
                          const firstTargetBranch = branches.find(
                            (b) => !b.is_deleted && b.id !== v.branch_id
                          );
                          setMoveTargetBranchId(firstTargetBranch?.id ?? null);
                          setShowMoveVersionDialog(true);
                        }}
                        style={{ ...btnBase, background: "transparent", color: "var(--text-secondary)", fontSize: "10px", padding: "3px 10px" }}
                      >
                        Move
                      </button>
                      {!isDefaultVersion && (
                        <button
                          onClick={() => handleSetVersionAsDefault(v.id)}
                          disabled={versionActionSaving}
                          style={{ ...btnBase, background: "transparent", color: "var(--teal)", border: "1px solid rgba(20,184,166,0.35)", fontSize: "10px", padding: "3px 10px", opacity: versionActionSaving ? 0.6 : 1 }}
                        >
                          Set Default
                        </button>
                      )}
                      <span style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)" }}>
                        {formatLocalizedDateTime(v.created_at, dateTimeLocale)}
                      </span>
                    </div>
                  </div>

                  {/* Edit reason inline */}
                  {isEditing && (
                    <div className="flex items-center gap-2" style={{ marginTop: "8px" }}>
                      <input
                        value={editVersionReason}
                        onChange={(e) => setEditVersionReason(e.target.value)}
                        style={{ ...inputStyle, width: "auto", flexGrow: 1 }}
                        placeholder="Reason / description"
                      />
                      <button
                        onClick={handleUpdateVersionReason}
                        disabled={versionActionSaving}
                        style={{ ...btnBase, background: "var(--gold)", color: "#08080B", border: "none", fontFamily: "var(--font-syne)" }}
                      >
                        {versionActionSaving ? "Saving…" : "Save"}
                      </button>
                      <button
                        onClick={() => setEditingVersionId(null)}
                        style={{ ...btnBase, background: "transparent", color: "var(--text-secondary)" }}
                      >
                        Cancel
                      </button>
                    </div>
                  )}

                  {/* Delete confirm inline */}
                  {isConfirmDelete && (
                    <div className="flex items-center gap-2" style={{ marginTop: "8px", background: "var(--error-dim)", border: "1px solid var(--error)", borderRadius: "var(--radius-unified)", padding: "8px 12px" }}>
                      <span style={{ fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--error)" }}>
                        Delete v{v.version}? This cannot be undone.
                      </span>
                      <button
                        onClick={() => handleDeleteVersion(v.id)}
                        disabled={versionActionSaving}
                        style={{ ...btnBase, background: "var(--error)", color: "#fff", border: "none" }}
                      >
                        {versionActionSaving ? "Deleting…" : "Confirm Delete"}
                      </button>
                      <button onClick={() => setConfirmDeleteVersionId(null)} style={{ ...btnBase, background: "transparent", color: "var(--text-secondary)" }}>
                        Cancel
                      </button>
                    </div>
                  )}

                  <div className="flex items-center gap-4" style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)", marginTop: "4px" }}>
                    <span>{v.row_count.toLocaleString()} rows</span>
                    <span>{formatBytes(v.file_size_bytes)}</span>
                    {v.author && <span>by {v.author}</span>}
                    <span style={{ opacity: 0.6 }}>{v.sha256.slice(0, 12)}…</span>
                    {Object.keys(v.column_roles).length > 0 && (
                      <span style={{ color: "var(--teal)" }}>
                        roles: {Object.entries(v.column_roles).map(([c, r]) => `${c}→${r}`).join(", ")}
                      </span>
                    )}
                  </div>
                </div>
              );
            })
          )}
        </div>
      )}

      {/* ===== BRANCHES TAB ===== */}
      {tab === "branches" && (
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span style={{ fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--text-tertiary)" }}>
              {branches.length} branch{branches.length !== 1 ? "es" : ""}
            </span>
            <button
              onClick={() => setShowCreateBranch(true)}
              style={{ ...btnBase, background: "var(--bg-elevated)", color: "var(--text-secondary)" }}
            >
              + New Branch
            </button>
          </div>

          {branches.map((branch) => {
            const isActive = branch.id === activeBranchId;
            const isConfirmDeleteThis = confirmDeleteBranch === branch.id;
            const branchVers = versions.filter((v) => v.branch_id === branch.id);
            const headVer = branch.head_version_id
              ? (branchVers.find((v) => v.id === branch.head_version_id) ?? branchVers[0])
              : branchVers[0];

            return (
              <div
                key={branch.id}
                className="rounded-lg"
                style={{
                  background: isActive ? "rgba(250,189,0,0.05)" : "var(--bg-elevated)",
                  border: isActive ? "1px solid var(--gold-muted)" : "1px solid var(--border-dim)",
                  padding: "14px 16px",
                }}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span style={{ fontFamily: "var(--font-syne)", fontSize: "14px", fontWeight: 600, color: "var(--text-primary)" }}>
                      {branch.name}
                    </span>
                    {branch.is_default && (
                      <span style={{ background: "var(--gold-faint)", border: "1px solid var(--gold-muted)", color: "var(--gold)", padding: "1px 6px", fontSize: "9px", fontFamily: "var(--font-jetbrains)", borderRadius: "var(--radius-unified)" }}>
                        default
                      </span>
                    )}
                    {isActive && !branch.is_default && (
                      <span style={{ background: "rgba(20,184,166,0.1)", border: "1px solid rgba(20,184,166,0.3)", color: "var(--teal)", padding: "1px 6px", fontSize: "9px", fontFamily: "var(--font-jetbrains)", borderRadius: "var(--radius-unified)" }}>
                        active
                      </span>
                    )}
                    {branch.description && (
                      <span style={{ fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--text-tertiary)" }}>
                        {branch.description}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {!isActive && (
                      <button
                        onClick={() => setActiveBranchId(branch.id)}
                        style={{ ...btnBase, background: "var(--bg-base)", color: "var(--text-secondary)", fontSize: "10px", padding: "3px 10px" }}
                      >
                        Switch
                      </button>
                    )}
                    {!branch.is_default && (
                      <button
                        onClick={() => handleSetDefaultBranch(branch.id)}
                        style={{ ...btnBase, background: "transparent", color: "var(--gold)", border: "1px solid var(--gold-muted)", fontSize: "10px", padding: "3px 10px" }}
                      >
                        Set default
                      </button>
                    )}
                    <button
                      onClick={() => {
                        setEditingBranch(branch);
                        setEditBranchName(branch.name);
                        setEditBranchDesc(branch.description);
                      }}
                      style={{ ...btnBase, background: "transparent", color: "var(--text-tertiary)", fontSize: "10px", padding: "3px 10px" }}
                    >
                      Edit
                    </button>
                    {!branch.is_default && (
                      <button
                        onClick={() => setConfirmDeleteBranch(branch.id)}
                        style={{ ...btnBase, background: "transparent", color: "var(--error)", border: "1px solid rgba(239,68,68,0.3)", fontSize: "10px", padding: "3px 10px" }}
                      >
                        Delete
                      </button>
                    )}
                  </div>
                </div>

                <div className="flex items-center gap-4" style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)", marginTop: "6px" }}>
                  <span>{branchVers.length} version{branchVers.length !== 1 ? "s" : ""}</span>
                  {headVer && <span>head: v{headVer.version} ({headVer.row_count.toLocaleString()} rows)</span>}
                  {branch.base_version_id && <span>forked from version {branch.base_version_id.slice(0, 8)}…</span>}
                  <span>{formatLocalizedDateTime(branch.created_at, dateTimeLocale)}</span>
                  {branch.author && <span>by {branch.author}</span>}
                </div>

                {/* Delete branch confirm */}
                {isConfirmDeleteThis && (
                  <div className="flex items-center gap-2" style={{ marginTop: "10px", background: "var(--error-dim)", border: "1px solid var(--error)", borderRadius: "var(--radius-unified)", padding: "8px 12px" }}>
                    <span style={{ fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--error)" }}>
                      Delete branch &quot;{branch.name}&quot; and all its unique versions?
                    </span>
                    <button
                      onClick={() => handleDeleteBranch(branch.id)}
                      style={{ ...btnBase, background: "var(--error)", color: "#fff", border: "none" }}
                    >
                      Confirm Delete
                    </button>
                    <button onClick={() => setConfirmDeleteBranch(null)} style={{ ...btnBase, background: "transparent", color: "var(--text-secondary)" }}>
                      Cancel
                    </button>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* ===== SAVE VERSION DIALOG ===== */}
      {showVersionDialog && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", zIndex: 50, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div className="rounded-xl" style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", padding: "24px", width: "400px", maxWidth: "90vw" }}>
            <h2 ref={saveVersionDialogTitleRef} tabIndex={-1} style={{ fontFamily: "var(--font-syne)", fontWeight: 600, fontSize: "16px", color: "var(--text-primary)", marginBottom: "16px" }}>Save as new version</h2>
            {activeBranch && (
              <div style={{ marginBottom: "12px", fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)" }}>
                Branch: <span style={{ color: "var(--gold)" }}>{activeBranch.name}</span>
              </div>
            )}
            <div className="flex flex-col gap-3">
              <div>
                <label style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)", marginBottom: "4px", display: "block" }}>Reason for changes</label>
                <textarea
                  value={versionReason}
                  onChange={(e) => setVersionReason(e.target.value)}
                  placeholder="e.g. corrected typos in column X"
                  rows={3}
                  style={{ ...inputStyle, resize: "vertical" as const, outline: "none" }}
                />
              </div>
              <div>
                <label style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)", marginBottom: "4px", display: "block" }}>Author (optional)</label>
                <input
                  value={versionAuthor}
                  onChange={(e) => setVersionAuthor(e.target.value)}
                  placeholder="Your name"
                  style={{ ...inputStyle, outline: "none" }}
                />
              </div>
              <div className="flex items-center gap-2" style={{ marginTop: "4px" }}>
                <button
                  onClick={handleDataSave}
                  disabled={dataSaving}
                  style={{ background: "var(--gold)", color: "#08080B", border: "none", borderRadius: "var(--radius-unified)", padding: "6px 16px", fontSize: "11px", fontWeight: 600, fontFamily: "var(--font-syne)", cursor: dataSaving ? "default" : "pointer", opacity: dataSaving ? 0.7 : 1 }}
                >
                  {dataSaving ? "Saving…" : "Save Version"}
                </button>
                <button
                  onClick={() => setShowVersionDialog(false)}
                  style={{ ...btnBase, background: "transparent", color: "var(--text-secondary)", fontFamily: "var(--font-syne)" }}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ===== CREATE BRANCH DIALOG ===== */}
      {showCreateBranch && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", zIndex: 50, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div className="rounded-xl" style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", padding: "24px", width: "420px", maxWidth: "90vw" }}>
            <h2 style={{ fontFamily: "var(--font-syne)", fontWeight: 600, fontSize: "16px", color: "var(--text-primary)", marginBottom: "16px" }}>Create Branch</h2>
            <div className="flex flex-col gap-3">
              <div>
                <label style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)", marginBottom: "4px", display: "block" }}>Branch name *</label>
                <input
                  value={newBranchName}
                  onChange={(e) => setNewBranchName(e.target.value)}
                  placeholder="e.g. experiment, feature/clean"
                  style={{ ...inputStyle, outline: "none" }}
                />
              </div>
              <div>
                <label style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)", marginBottom: "4px", display: "block" }}>Description (optional)</label>
                <input
                  value={newBranchDesc}
                  onChange={(e) => setNewBranchDesc(e.target.value)}
                  placeholder="What is this branch for?"
                  style={{ ...inputStyle, outline: "none" }}
                />
              </div>
              <div>
                <label style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)", marginBottom: "4px", display: "block" }}>Fork from version (optional — defaults to current head)</label>
                <select
                  value={newBranchFromVersionId}
                  onChange={(e) => setNewBranchFromVersionId(e.target.value)}
                  style={{ ...selectStyle, width: "100%" }}
                >
                  <option value="">Current head</option>
                  {versions.map((v) => (
                    <option key={v.id} value={v.id}>
                      v{v.version} — {formatLocalizedDate(v.created_at, dateTimeLocale)} ({v.row_count.toLocaleString()} rows) [{v.branch_id === defaultBranchId ? "main" : (branches.find((b) => b.id === v.branch_id)?.name ?? "?")}]
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)", marginBottom: "4px", display: "block" }}>Author (optional)</label>
                <input
                  value={newBranchAuthor}
                  onChange={(e) => setNewBranchAuthor(e.target.value)}
                  placeholder="Your name"
                  style={{ ...inputStyle, outline: "none" }}
                />
              </div>
              <div className="flex items-center gap-2" style={{ marginTop: "4px" }}>
                <button
                  onClick={handleCreateBranch}
                  disabled={branchSaving || !newBranchName.trim()}
                  style={{ background: "var(--gold)", color: "#08080B", border: "none", borderRadius: "var(--radius-unified)", padding: "6px 16px", fontSize: "11px", fontWeight: 600, fontFamily: "var(--font-syne)", cursor: branchSaving || !newBranchName.trim() ? "default" : "pointer", opacity: branchSaving || !newBranchName.trim() ? 0.7 : 1 }}
                >
                  {branchSaving ? "Creating…" : "Create Branch"}
                </button>
                <button
                  onClick={() => { setShowCreateBranch(false); setNewBranchName(""); setNewBranchDesc(""); setNewBranchFromVersionId(""); }}
                  style={{ ...btnBase, background: "transparent", color: "var(--text-secondary)", fontFamily: "var(--font-syne)" }}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ===== EDIT BRANCH DIALOG ===== */}
      {editingBranch && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", zIndex: 50, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div className="rounded-xl" style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", padding: "24px", width: "400px", maxWidth: "90vw" }}>
            <h2 style={{ fontFamily: "var(--font-syne)", fontWeight: 600, fontSize: "16px", color: "var(--text-primary)", marginBottom: "16px" }}>Edit Branch</h2>
            <div className="flex flex-col gap-3">
              <div>
                <label style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)", marginBottom: "4px", display: "block" }}>Branch name</label>
                <input
                  value={editBranchName}
                  onChange={(e) => setEditBranchName(e.target.value)}
                  style={{ ...inputStyle, outline: "none" }}
                />
              </div>
              <div>
                <label style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)", marginBottom: "4px", display: "block" }}>Description</label>
                <input
                  value={editBranchDesc}
                  onChange={(e) => setEditBranchDesc(e.target.value)}
                  style={{ ...inputStyle, outline: "none" }}
                />
              </div>
              <div className="flex items-center gap-2" style={{ marginTop: "4px" }}>
                <button
                  onClick={handleUpdateBranch}
                  disabled={branchSaving}
                  style={{ background: "var(--gold)", color: "#08080B", border: "none", borderRadius: "var(--radius-unified)", padding: "6px 16px", fontSize: "11px", fontWeight: 600, fontFamily: "var(--font-syne)", cursor: branchSaving ? "default" : "pointer", opacity: branchSaving ? 0.7 : 1 }}
                >
                  {branchSaving ? "Saving…" : "Save"}
                </button>
                <button
                  onClick={() => setEditingBranch(null)}
                  style={{ ...btnBase, background: "transparent", color: "var(--text-secondary)", fontFamily: "var(--font-syne)" }}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ===== COPY VERSION DIALOG ===== */}
      {showCopyVersionDialog && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", zIndex: 50, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div className="rounded-xl" style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", padding: "24px", width: "400px", maxWidth: "90vw" }}>
            <h2 style={{ fontFamily: "var(--font-syne)", fontWeight: 600, fontSize: "16px", color: "var(--text-primary)", marginBottom: "16px" }}>Copy Version</h2>
            <div className="flex flex-col gap-3">
              <div>
                <label style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)", marginBottom: "4px", display: "block" }}>Reason / Description</label>
                <input
                  value={copyVersionReason}
                  onChange={(e) => setCopyVersionReason(e.target.value)}
                  style={{ ...inputStyle, outline: "none" }}
                  placeholder="Why are you copying this version?"
                />
              </div>
              <div>
                <label style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)", marginBottom: "4px", display: "block" }}>Author (optional)</label>
                <input
                  value={copyVersionAuthor}
                  onChange={(e) => setCopyVersionAuthor(e.target.value)}
                  style={{ ...inputStyle, outline: "none" }}
                  placeholder="Your name"
                />
              </div>
              <div className="flex items-center gap-2" style={{ marginTop: "4px" }}>
                <button
                  onClick={() => copySourceVersionId && handleCopyVersion(copySourceVersionId)}
                  disabled={copySaving || !copyVersionReason.trim()}
                  style={{ background: "var(--gold)", color: "#08080B", border: "none", borderRadius: "var(--radius-unified)", padding: "6px 16px", fontSize: "11px", fontWeight: 600, fontFamily: "var(--font-syne)", cursor: copySaving || !copyVersionReason.trim() ? "default" : "pointer", opacity: copySaving || !copyVersionReason.trim() ? 0.5 : 1 }}
                >
                  {copySaving ? "Copying…" : "Copy Version"}
                </button>
                <button
                  onClick={() => { setShowCopyVersionDialog(false); setCopySourceVersionId(null); setCopyVersionReason(""); setCopyVersionAuthor(""); }}
                  style={{ ...btnBase, background: "transparent", color: "var(--text-secondary)", fontFamily: "var(--font-syne)" }}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ===== MOVE VERSION DIALOG ===== */}
      {showMoveVersionDialog && (
        <div style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", zIndex: 50, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div className="rounded-xl" style={{ background: "var(--bg-surface)", border: "1px solid var(--border)", padding: "24px", width: "400px", maxWidth: "90vw" }}>
            <h2 style={{ fontFamily: "var(--font-syne)", fontWeight: 600, fontSize: "16px", color: "var(--text-primary)", marginBottom: "16px" }}>Move Version</h2>
            <div className="flex flex-col gap-3">
              <div>
                <label style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)", marginBottom: "4px", display: "block" }}>Version to move</label>
                <div style={{ fontFamily: "var(--font-jetbrains)", fontSize: "11px", color: "var(--text-secondary)", padding: "8px 12px", background: "var(--bg-base)", borderRadius: "var(--radius-unified)", border: "1px solid var(--border)" }}>
                  {moveSourceVersionReason || "Unknown version"}
                </div>
              </div>
              <div>
                <label style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--text-tertiary)", marginBottom: "4px", display: "block" }}>Target branch</label>
                <select
                  value={moveTargetBranchId ?? ""}
                  onChange={(e) => setMoveTargetBranchId(e.target.value || null)}
                  style={{ ...inputStyle, outline: "none" }}
                  disabled={moveTargetBranches.length === 0}
                >
                  <option value="" disabled>
                    {moveTargetBranches.length === 0 ? "No other branches available" : "Select a branch"}
                  </option>
                  {moveTargetBranches.map((b) => (
                    <option key={b.id} value={b.id}>
                      {b.name} {b.is_default ? "(default)" : ""}
                    </option>
                  ))}
                </select>
              </div>
              {moveTargetBranches.length === 0 && (
                <div style={{ fontFamily: "var(--font-jetbrains)", fontSize: "10px", color: "var(--warning)" }}>
                  Create another branch first to move this version.
                </div>
              )}
              <div className="flex items-center gap-2" style={{ marginTop: "4px" }}>
                <button
                  onClick={() => moveSourceVersionId && handleMoveVersion(moveSourceVersionId)}
                  disabled={moveSaving || !moveTargetBranchId || moveTargetBranches.length === 0}
                  style={{ background: "var(--gold)", color: "#08080B", border: "none", borderRadius: "var(--radius-unified)", padding: "6px 16px", fontSize: "11px", fontWeight: 600, fontFamily: "var(--font-syne)", cursor: moveSaving || !moveTargetBranchId || moveTargetBranches.length === 0 ? "default" : "pointer", opacity: moveSaving || !moveTargetBranchId || moveTargetBranches.length === 0 ? 0.5 : 1 }}
                >
                  {moveSaving ? "Moving…" : "Move Version"}
                </button>
                <button
                  onClick={() => { setShowMoveVersionDialog(false); setMoveSourceVersionId(null); setMoveSourceVersionReason(""); setMoveTargetBranchId(null); }}
                  style={{ ...btnBase, background: "transparent", color: "var(--text-secondary)", fontFamily: "var(--font-syne)" }}
                >
                  Cancel
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

