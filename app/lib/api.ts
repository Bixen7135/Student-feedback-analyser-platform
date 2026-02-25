/**
 * Typed API client for the FastAPI backend.
 * All calls use relative /api/* paths which Next.js proxies to localhost:8000.
 */

const API_BASE = "/api";

export interface StageStatus {
  name: string;
  status: "pending" | "running" | "completed" | "failed" | "skipped";
  started_at: string | null;
  completed_at: string | null;
  duration_seconds: number | null;
  error: string | null;
}

export interface RunSummary {
  run_id: string;
  created_at: string;
  config_hash: string;
  data_snapshot_id: string;
  random_seed: number;
  stages: Record<string, StageStatus>;
  dataset_id: string | null;
  branch_id: string | null;
  dataset_version: number | null;
  name: string | null;
}

export interface RunDetail extends RunSummary {
  git_commit?: string;
  system_info?: Record<string, string>;
}

export interface PsychometricsMetrics {
  method: string;
  n_obs: number;
  factor_names: string[];
  fit_statistics: Record<string, number | string>;
  reliability: {
    cronbach_alpha?: Record<string, number>;
    mcdonald_omega?: Record<string, number>;
  };
  loadings?: Record<string, Record<string, number>>;
}

export interface ClassificationMetrics {
  task: string;
  model_type: string;
  macro_f1: number;
  accuracy: number;
  per_class_f1: Record<string, number>;
  confusion_matrix: number[][];
  classes: string[];
  n_samples: number;
}

export interface FusionMetrics {
  factor_names: string[];
  survey_only: { mae: number; r_squared: number; per_factor_mae: Record<string, number> };
  text_only: { mae: number; r_squared: number; per_factor_mae: Record<string, number> };
  late_fusion: { mae: number; r_squared: number; per_factor_mae: Record<string, number> };
  delta_mae: Record<string, number>;
  delta_r2: Record<string, number>;
  ablations?: Record<string, unknown>;
}

export interface ContradictionMetrics {
  overall_rate: number;
  n_total: number;
  n_contradictions: number;
  by_type: Record<string, number>;
  stratified_by_language: Record<string, number>;
  stratified_by_detail_level: Record<string, number>;
  disclaimer: string;
}

export interface ArtifactItem {
  name: string;
  path: string;
  type: string;
  stage: string;
  size_bytes: number | null;
  created_at: string | null;
}

export interface SummaryData {
  total_datasets: number;
  total_models: number;
  total_analyses: number;
  total_responses: number;
  n_latent_factors: number;
  n_survey_items: number;
}

// ---------------------------------------------------------------------------
// Generic fetch helper
// ---------------------------------------------------------------------------

async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${text || res.statusText}`);
  }
  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Runs
// ---------------------------------------------------------------------------

export async function fetchRuns(): Promise<RunSummary[]> {
  return apiFetch<RunSummary[]>("/runs");
}

export async function fetchRunDetail(runId: string): Promise<RunDetail> {
  return apiFetch<RunDetail>(`/runs/${runId}`);
}

export async function createRun(options?: {
  data_path?: string;
  seed?: number;
  dataset_id?: string | null;
  branch_id?: string | null;
  dataset_version?: number | null;
  name?: string | null;
}): Promise<RunSummary> {
  return apiFetch<RunSummary>("/runs", {
    method: "POST",
    body: JSON.stringify({ seed: 42, ...options }),
  });
}

export async function deleteRun(runId: string): Promise<void> {
  const res = await fetch(`${API_BASE}/runs/${runId}`, { method: "DELETE" });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`API ${res.status}: ${text || res.statusText}`);
  }
}

export async function startStage(runId: string, stageName: string): Promise<void> {
  await apiFetch(`/runs/${runId}/stages/${stageName}/start`, { method: "POST" });
}

export async function getStageStatus(runId: string, stageName: string): Promise<StageStatus> {
  return apiFetch<StageStatus>(`/runs/${runId}/stages/${stageName}/status`);
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

export async function fetchPsychometricsMetrics(runId: string): Promise<PsychometricsMetrics> {
  return apiFetch<PsychometricsMetrics>(`/runs/${runId}/metrics/psychometrics`);
}

export async function fetchClassificationMetrics(
  runId: string,
  task: string
): Promise<ClassificationMetrics[]> {
  return apiFetch<ClassificationMetrics[]>(`/runs/${runId}/metrics/classification/${task}`);
}

export async function fetchFusionMetrics(runId: string): Promise<FusionMetrics> {
  return apiFetch<FusionMetrics>(`/runs/${runId}/metrics/fusion`);
}

export async function fetchContradictionMetrics(runId: string): Promise<ContradictionMetrics> {
  return apiFetch<ContradictionMetrics>(`/runs/${runId}/metrics/contradiction`);
}

// ---------------------------------------------------------------------------
// Artifacts
// ---------------------------------------------------------------------------

export async function fetchArtifacts(runId: string): Promise<ArtifactItem[]> {
  const data = await apiFetch<{ run_id: string; artifacts: ArtifactItem[] }>(`/runs/${runId}/artifacts`);
  return data.artifacts;
}

export async function downloadReport(runId: string, reportName: string): Promise<Blob> {
  const res = await fetch(`${API_BASE}/runs/${runId}/artifacts/reports/${reportName}`);
  if (!res.ok) throw new Error(`Download failed: ${res.statusText}`);
  return res.blob();
}

export async function fetchSummary(): Promise<SummaryData> {
  return apiFetch<SummaryData>("/summary");
}

// ---------------------------------------------------------------------------
// Dataset types
// ---------------------------------------------------------------------------

export interface ColumnSchema {
  name: string;
  dtype: string;
  n_unique: number;
  n_null: number;
  sample_values: string[];
}

export interface DatasetSummary {
  id: string;
  name: string;
  description: string;
  tags: string[];
  author: string;
  created_at: string;
  current_version: number;
  row_count: number;
  file_size_bytes: number;
  sha256: string;
  status: string;
  schema_info: ColumnSchema[];
  default_branch_id: string | null;
}

export interface DatasetListResponse {
  datasets: DatasetSummary[];
  total: number;
  page: number;
  per_page: number;
}

export interface DatasetPreview {
  dataset_id: string;
  version: number;
  columns: string[];
  total_rows: number;
  offset: number;
  limit: number;
  rows: string[][];
}

export interface DatasetVersion {
  id: string;
  dataset_id: string;
  version: number;
  created_at: string;
  author: string;
  reason: string;
  sha256: string;
  row_count: number;
  file_size_bytes: number;
  branch_id: string | null;
  column_roles: Record<string, string>;
}

export interface DatasetBranch {
  id: string;
  dataset_id: string;
  name: string;
  description: string;
  base_version_id: string | null;
  head_version_id: string | null;
  author: string;
  created_at: string;
  is_default: boolean;
  is_deleted: boolean;
}

export interface DatasetDeleteResult {
  deleted: boolean;
  reason?: string;
  dependencies?: { models: number; analyses: number };
}

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------

export async function fetchDatasets(params?: {
  search?: string;
  tags?: string[];
  sort?: string;
  order?: string;
  page?: number;
  per_page?: number;
}): Promise<DatasetListResponse> {
  const sp = new URLSearchParams();
  if (params?.search) sp.set("search", params.search);
  if (params?.tags?.length) sp.set("tags", JSON.stringify(params.tags));
  if (params?.sort) sp.set("sort", params.sort);
  if (params?.order) sp.set("order", params.order);
  if (params?.page) sp.set("page", String(params.page));
  if (params?.per_page) sp.set("per_page", String(params.per_page));
  const qs = sp.toString();
  return apiFetch<DatasetListResponse>(`/datasets${qs ? `?${qs}` : ""}`);
}

export async function fetchDatasetDetail(datasetId: string): Promise<DatasetSummary> {
  return apiFetch<DatasetSummary>(`/datasets/${datasetId}`);
}

export async function fetchDatasetPreview(
  datasetId: string,
  params?: { version?: number; branch_id?: string; offset?: number; limit?: number }
): Promise<DatasetPreview> {
  const sp = new URLSearchParams();
  if (params?.version != null) sp.set("version", String(params.version));
  if (params?.branch_id) sp.set("branch_id", params.branch_id);
  if (params?.offset != null) sp.set("offset", String(params.offset));
  if (params?.limit != null) sp.set("limit", String(params.limit));
  const qs = sp.toString();
  return apiFetch<DatasetPreview>(`/datasets/${datasetId}/preview${qs ? `?${qs}` : ""}`);
}

export async function fetchDatasetSchema(
  datasetId: string,
  params?: { version_id?: string }
): Promise<{
  dataset_id: string;
  version_id?: string | null;
  version?: number | null;
  columns: ColumnSchema[];
}> {
  const sp = new URLSearchParams();
  if (params?.version_id) sp.set("version_id", params.version_id);
  const qs = sp.toString();
  return apiFetch(`/datasets/${datasetId}/schema${qs ? `?${qs}` : ""}`);
}

export async function fetchDatasetVersions(
  datasetId: string,
  branchId?: string
): Promise<DatasetVersion[]> {
  const qs = branchId ? `?branch_id=${encodeURIComponent(branchId)}` : "";
  return apiFetch<DatasetVersion[]>(`/datasets/${datasetId}/versions${qs}`);
}

export async function fetchColumnRoles(
  datasetId: string,
  params?: { version_id?: string; version?: number }
): Promise<{
  dataset_id: string;
  version_id: string | null;
  version: number | null;
  column_roles: Record<string, string>;
}> {
  const sp = new URLSearchParams();
  if (params?.version_id) sp.set("version_id", params.version_id);
  if (params?.version != null) sp.set("version", String(params.version));
  const qs = sp.toString();
  return apiFetch(`/datasets/${datasetId}/column-roles${qs}`);
}

export async function uploadDataset(
  file: File,
  metadata: { name: string; description?: string; tags?: string[]; author?: string }
): Promise<DatasetSummary> {
  const form = new FormData();
  form.append("file", file);
  form.append("name", metadata.name);
  if (metadata.description) form.append("description", metadata.description);
  if (metadata.tags?.length) form.append("tags", JSON.stringify(metadata.tags));
  if (metadata.author) form.append("author", metadata.author);

  const res = await fetch(`${API_BASE}/datasets/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Upload failed: ${text || res.statusText}`);
  }
  return res.json();
}

export async function updateDatasetMetadata(
  datasetId: string,
  update: { name?: string; description?: string; tags?: string[] }
): Promise<DatasetSummary> {
  return apiFetch<DatasetSummary>(`/datasets/${datasetId}`, {
    method: "PATCH",
    body: JSON.stringify(update),
  });
}

export async function deleteDataset(datasetId: string, force = false): Promise<DatasetDeleteResult> {
  return apiFetch<DatasetDeleteResult>(`/datasets/${datasetId}?force=${force}`, {
    method: "DELETE",
  });
}

export async function createSubset(
  datasetId: string,
  params: {
    name: string;
    description?: string;
    author?: string;
    version?: number;
    filter_config: Record<string, unknown>;
  }
): Promise<DatasetSummary> {
  return apiFetch<DatasetSummary>(`/datasets/${datasetId}/subset`, {
    method: "POST",
    body: JSON.stringify(params),
  });
}

// Phase 6: cell / row mutations

export interface CellChange {
  row_idx: number;
  col: string;
  value: string;
}

export async function updateDatasetCells(
  datasetId: string,
  changes: CellChange[],
  reason = "cell edits",
  author = "",
  branchId?: string
): Promise<DatasetVersion> {
  return apiFetch<DatasetVersion>(`/datasets/${datasetId}/cells`, {
    method: "PATCH",
    body: JSON.stringify({ changes, reason, author, branch_id: branchId ?? null }),
  });
}

export async function addDatasetRows(
  datasetId: string,
  rows: Record<string, string>[],
  reason = "added rows",
  author = "",
  branchId?: string
): Promise<DatasetVersion> {
  return apiFetch<DatasetVersion>(`/datasets/${datasetId}/rows`, {
    method: "POST",
    body: JSON.stringify({ rows, reason, author, branch_id: branchId ?? null }),
  });
}

export async function deleteDatasetRows(
  datasetId: string,
  rowIndices: number[],
  reason = "deleted rows",
  author = "",
  branchId?: string
): Promise<DatasetVersion> {
  return apiFetch<DatasetVersion>(`/datasets/${datasetId}/rows`, {
    method: "DELETE",
    body: JSON.stringify({ row_indices: rowIndices, reason, author, branch_id: branchId ?? null }),
  });
}

export async function renameDatasetColumns(
  datasetId: string,
  renames: Record<string, string>,
  reason = "rename columns",
  author = "",
  branchId?: string
): Promise<DatasetVersion> {
  return apiFetch<DatasetVersion>(`/datasets/${datasetId}/columns`, {
    method: "PATCH",
    body: JSON.stringify({ renames, reason, author, branch_id: branchId ?? null }),
  });
}

export async function createEmptyDataset(params: {
  name: string;
  columns: string[];
  description?: string;
  tags?: string[];
  author?: string;
}): Promise<DatasetSummary> {
  return apiFetch<DatasetSummary>("/datasets/empty", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

// ---------------------------------------------------------------------------
// Branch management
// ---------------------------------------------------------------------------

export async function fetchDatasetBranches(datasetId: string): Promise<DatasetBranch[]> {
  return apiFetch<DatasetBranch[]>(`/datasets/${datasetId}/branches`);
}

export async function createDatasetBranch(
  datasetId: string,
  params: { name: string; description?: string; from_version_id?: string; author?: string }
): Promise<DatasetBranch> {
  return apiFetch<DatasetBranch>(`/datasets/${datasetId}/branches`, {
    method: "POST",
    body: JSON.stringify(params),
  });
}

export async function updateDatasetBranch(
  datasetId: string,
  branchId: string,
  params: { name?: string; description?: string }
): Promise<DatasetBranch> {
  return apiFetch<DatasetBranch>(`/datasets/${datasetId}/branches/${branchId}`, {
    method: "PATCH",
    body: JSON.stringify(params),
  });
}

export async function deleteDatasetBranch(
  datasetId: string,
  branchId: string
): Promise<{ deleted: boolean; branch_id: string }> {
  return apiFetch(`/datasets/${datasetId}/branches/${branchId}`, {
    method: "DELETE",
  });
}

export async function setDefaultBranch(
  datasetId: string,
  branchId: string
): Promise<{ dataset_id: string; default_branch_id: string }> {
  return apiFetch(`/datasets/${datasetId}/branches/${branchId}/set-default`, {
    method: "POST",
  });
}

// ---------------------------------------------------------------------------
// Version metadata editing / deletion
// ---------------------------------------------------------------------------

export async function updateVersionMetadata(
  datasetId: string,
  versionId: string,
  reason: string
): Promise<DatasetVersion> {
  return apiFetch<DatasetVersion>(`/datasets/${datasetId}/versions/${versionId}`, {
    method: "PATCH",
    body: JSON.stringify({ reason }),
  });
}

export async function deleteDatasetVersion(
  datasetId: string,
  versionId: string
): Promise<{ deleted: boolean; version_id: string }> {
  return apiFetch(`/datasets/${datasetId}/versions/${versionId}`, {
    method: "DELETE",
  });
}

export async function copyDatasetVersion(
  datasetId: string,
  versionId: string,
  newReason: string,
  author?: string,
  branchId?: string
): Promise<DatasetVersion> {
  const body: { new_reason: string; author?: string; branch_id?: string } = { new_reason: newReason };
  if (author) body.author = author;
  if (branchId) body.branch_id = branchId;
  return apiFetch<DatasetVersion>(`/datasets/${datasetId}/versions/${versionId}/copy`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function moveDatasetVersion(
  datasetId: string,
  versionId: string,
  targetBranchId: string,
  author?: string
): Promise<DatasetVersion> {
  const body: { target_branch_id: string; author?: string } = { target_branch_id: targetBranchId };
  if (author) body.author = author;
  return apiFetch<DatasetVersion>(`/datasets/${datasetId}/versions/${versionId}/move`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function restoreDatasetVersion(
  datasetId: string,
  versionId: string,
  reason?: string,
  author?: string
): Promise<DatasetVersion> {
  const body: { reason?: string; author?: string } = {};
  if (reason) body.reason = reason;
  if (author) body.author = author;
  return apiFetch<DatasetVersion>(`/datasets/${datasetId}/versions/${versionId}/restore`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function setDatasetVersionAsDefault(
  datasetId: string,
  versionId: string,
  author?: string
): Promise<DatasetVersion> {
  const body: { author?: string } = {};
  if (author) body.author = author;
  return apiFetch<DatasetVersion>(`/datasets/${datasetId}/versions/${versionId}/set-default`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

// ---------------------------------------------------------------------------
// Model types
// ---------------------------------------------------------------------------

export interface ModelSummary {
  id: string;
  name: string;
  task: string;
  model_type: string;
  version: number;
  dataset_id: string | null;
  dataset_version: number | null;
  config: Record<string, unknown>;
  metrics: Record<string, unknown>;
  created_at: string;
  status: string;
  storage_path: string;
  run_id: string | null;
  base_model_id: string | null;
}

export interface ModelLineageResponse {
  model_id: string;
  chain: ModelSummary[];
}

export interface ModelListResponse {
  models: ModelSummary[];
  total: number;
  page: number;
  per_page: number;
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

export async function fetchModels(params?: {
  task?: string;
  model_type?: string;
  dataset_id?: string;
  sort?: string;
  order?: string;
  page?: number;
  per_page?: number;
}): Promise<ModelListResponse> {
  const sp = new URLSearchParams();
  if (params?.task) sp.set("task", params.task);
  if (params?.model_type) sp.set("model_type", params.model_type);
  if (params?.dataset_id) sp.set("dataset_id", params.dataset_id);
  if (params?.sort) sp.set("sort", params.sort);
  if (params?.order) sp.set("order", params.order);
  if (params?.page) sp.set("page", String(params.page));
  if (params?.per_page) sp.set("per_page", String(params.per_page));
  const qs = sp.toString();
  return apiFetch<ModelListResponse>(`/models${qs ? `?${qs}` : ""}`);
}

export async function fetchModelDetail(modelId: string): Promise<ModelSummary> {
  return apiFetch<ModelSummary>(`/models/${modelId}`);
}

export async function fetchModelVersions(modelId: string): Promise<ModelSummary[]> {
  return apiFetch<ModelSummary[]>(`/models/${modelId}/versions`);
}

export async function compareModels(modelIds: string[]): Promise<ModelSummary[]> {
  return apiFetch<ModelSummary[]>("/models/compare", {
    method: "POST",
    body: JSON.stringify({ model_ids: modelIds }),
  });
}

export async function updateModelMetadata(
  modelId: string,
  update: { name?: string }
): Promise<ModelSummary> {
  return apiFetch<ModelSummary>(`/models/${modelId}`, {
    method: "PATCH",
    body: JSON.stringify(update),
  });
}

export async function deleteModel(modelId: string): Promise<{ deleted: boolean; reason?: string; dependencies?: { analyses: number } }> {
  return apiFetch(`/models/${modelId}`, {
    method: "DELETE",
  });
}

export async function fetchModelLineage(modelId: string): Promise<ModelLineageResponse> {
  return apiFetch<ModelLineageResponse>(`/models/${modelId}/lineage`);
}

// ---------------------------------------------------------------------------
// Training types
// ---------------------------------------------------------------------------

export interface TrainingConfigRequest {
  train_ratio?: number;
  val_ratio?: number;
  test_ratio?: number;
  class_balancing?: "none" | "oversample" | "class_weight";
  max_features?: number | null;
  C?: number | null;
  max_iter?: number | null;
  text_col?: string | null;
  label_col?: string | null;
}

export interface StartTrainingRequest {
  dataset_id: string;
  task: "language" | "sentiment" | "detail_level";
  model_type: "tfidf" | "char_ngram";
  config?: TrainingConfigRequest;
  dataset_version?: number | null;
  branch_id?: string | null;
  seed?: number;
  name?: string | null;
  base_model_id?: string | null;
}

export interface TrainingJob {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  dataset_id: string;
  dataset_version: number | null;
  branch_id: string | null;
  task: string;
  model_type: string;
  seed: number;
  name: string | null;
  base_model_id: string | null;
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
  model_id: string | null;
  model_name: string | null;
  model_version: number | null;
  metrics: Record<string, unknown> | null;
  config: Record<string, unknown> | null;
}

export interface TrainingListResponse {
  jobs: TrainingJob[];
  total: number;
}

// ---------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------

export async function startTraining(req: StartTrainingRequest): Promise<TrainingJob> {
  return apiFetch<TrainingJob>("/training/start", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function fetchTrainingJobs(params?: {
  task?: string;
  status?: string;
}): Promise<TrainingListResponse> {
  const sp = new URLSearchParams();
  if (params?.task) sp.set("task", params.task);
  if (params?.status) sp.set("status", params.status);
  const qs = sp.toString();
  return apiFetch<TrainingListResponse>(`/training${qs ? `?${qs}` : ""}`);
}

export async function fetchTrainingStatus(jobId: string): Promise<TrainingJob> {
  return apiFetch<TrainingJob>(`/training/${jobId}/status`);
}

export async function fetchTrainingResult(jobId: string): Promise<TrainingJob> {
  return apiFetch<TrainingJob>(`/training/${jobId}/result`);
}

// ---------------------------------------------------------------------------
// Analysis types
// ---------------------------------------------------------------------------

export interface ModelApplied {
  model_id: string;
  model_name: string;
  task: string;
  model_type: string;
  error: string | null;
  pred_col: string | null;
  conf_col: string | null;
  n_predicted: number;
  class_distribution: Record<string, number>;
  classes: string[];
}

export interface AnalysisSummary {
  analysis_id: string;
  dataset_id: string;
  dataset_version: number | null;
  n_rows: number;
  n_rows_processed: number;
  text_col: string | null;
  models_applied: ModelApplied[];
  created_at: string;
}

export interface AnalysisJob {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  dataset_id: string;
  dataset_version: number | null;
  branch_id: string | null;
  model_ids: string[];
  name: string;
  description: string;
  tags: string[];
  started_at: string | null;
  completed_at: string | null;
  error: string | null;
  result_summary: AnalysisSummary | null;
}

export interface AnalysisRecord {
  id: string;
  name: string;
  description: string;
  tags: string[];
  comments: string;
  dataset_id: string | null;
  dataset_version: number | null;
  branch_id: string | null;
  model_ids: string[];
  created_at: string;
  status: "pending" | "running" | "completed" | "failed";
  run_id: string;
  result_summary: AnalysisSummary | Record<string, unknown>;
}

export interface AnalysisListResponse {
  analyses: AnalysisRecord[];
  total: number;
  page: number;
  per_page: number;
}

export interface StartAnalysisRequest {
  dataset_id: string;
  model_ids: string[];
  name?: string;
  description?: string;
  tags?: string[];
  dataset_version?: number | null;
  branch_id?: string | null;
  text_col?: string | null;
}

export interface AnalysisResults {
  analysis_id: string;
  total_rows: number;
  offset: number;
  limit: number;
  columns: string[];
  rows: string[][];
}

export interface AnalysisComparison {
  run_1: {
    id: string;
    name: string;
    dataset_id: string | null;
    dataset_version: number | null;
    status: string;
    created_at: string;
    n_rows: number | null;
  };
  run_2: {
    id: string;
    name: string;
    dataset_id: string | null;
    dataset_version: number | null;
    status: string;
    created_at: string;
    n_rows: number | null;
  };
  shared_tasks: string[];
  task_comparisons: Array<{
    task: string;
    run_1_model: {
      model_id: string;
      model_name: string;
      model_type: string;
      n_predicted: number;
      class_distribution: Record<string, number>;
    };
    run_2_model: {
      model_id: string;
      model_name: string;
      model_type: string;
      n_predicted: number;
      class_distribution: Record<string, number>;
    };
    distribution_deltas: Record<string, number>;
  }>;
  disagreement: {
    same_dataset: boolean;
    by_task?: Record<string, number>;
    overall?: number | null;
    error?: string;
  } | null;
}

// ---------------------------------------------------------------------------
// Analyses
// ---------------------------------------------------------------------------

export async function startAnalysis(req: StartAnalysisRequest): Promise<AnalysisJob> {
  return apiFetch<AnalysisJob>("/analyses", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function fetchAnalyses(params?: {
  dataset_id?: string;
  model_id?: string;
  status?: string;
  sort?: string;
  order?: string;
  page?: number;
  per_page?: number;
}): Promise<AnalysisListResponse> {
  const sp = new URLSearchParams();
  if (params?.dataset_id) sp.set("dataset_id", params.dataset_id);
  if (params?.model_id) sp.set("model_id", params.model_id);
  if (params?.status) sp.set("status", params.status);
  if (params?.sort) sp.set("sort", params.sort);
  if (params?.order) sp.set("order", params.order);
  if (params?.page) sp.set("page", String(params.page));
  if (params?.per_page) sp.set("per_page", String(params.per_page));
  const qs = sp.toString();
  return apiFetch<AnalysisListResponse>(`/analyses${qs ? `?${qs}` : ""}`);
}

export async function fetchAnalysisDetail(analysisId: string): Promise<AnalysisRecord> {
  return apiFetch<AnalysisRecord>(`/analyses/${analysisId}`);
}

export async function fetchAnalysisStatus(analysisId: string): Promise<AnalysisJob> {
  return apiFetch<AnalysisJob>(`/analyses/${analysisId}/status`);
}

export async function updateAnalysis(
  analysisId: string,
  update: { name?: string; description?: string; tags?: string[]; comments?: string }
): Promise<AnalysisRecord> {
  return apiFetch<AnalysisRecord>(`/analyses/${analysisId}`, {
    method: "PATCH",
    body: JSON.stringify(update),
  });
}

export async function deleteAnalysis(analysisId: string): Promise<{ deleted: boolean }> {
  return apiFetch<{ deleted: boolean }>(`/analyses/${analysisId}`, {
    method: "DELETE",
  });
}

export async function compareAnalyses(
  analysisIds: [string, string]
): Promise<AnalysisComparison> {
  return apiFetch<AnalysisComparison>("/analyses/compare", {
    method: "POST",
    body: JSON.stringify({ analysis_ids: analysisIds }),
  });
}

export async function fetchAnalysisResults(
  analysisId: string,
  params?: {
    offset?: number;
    limit?: number;
    sort_col?: string;
    sort_order?: string;
    // Legacy single-filter (kept for compat)
    filter_col?: string;
    filter_val?: string;
    // Phase 5: multi-filter and search
    filters?: FilterRule[];
    search?: string;
  }
): Promise<AnalysisResults> {
  const sp = new URLSearchParams();
  if (params?.offset != null) sp.set("offset", String(params.offset));
  if (params?.limit != null) sp.set("limit", String(params.limit));
  if (params?.sort_col) sp.set("sort_col", params.sort_col);
  if (params?.sort_order) sp.set("sort_order", params.sort_order);
  if (params?.filter_col) sp.set("filter_col", params.filter_col);
  if (params?.filter_val) sp.set("filter_val", params.filter_val);
  if (params?.filters && params.filters.length > 0)
    sp.set("filters", JSON.stringify(params.filters));
  if (params?.search) sp.set("search", params.search);
  const qs = sp.toString();
  return apiFetch<AnalysisResults>(`/analyses/${analysisId}/results${qs ? `?${qs}` : ""}`);
}

export function getAnalysisExportUrl(analysisId: string, format: "csv" | "json" = "csv"): string {
  return `/api/analyses/${analysisId}/results/export?format=${format}`;
}

// Phase 5: filtered export URL builder
export function getFilteredExportUrl(
  analysisId: string,
  format: "csv" | "json",
  filters?: FilterRule[],
  search?: string,
  sortCol?: string,
  sortOrder?: string
): string {
  const sp = new URLSearchParams({ format });
  if (filters && filters.length > 0) sp.set("filters", JSON.stringify(filters));
  if (search) sp.set("search", search);
  if (sortCol) sp.set("sort_col", sortCol);
  if (sortOrder) sp.set("sort_order", sortOrder);
  return `/api/analyses/${analysisId}/results/export?${sp.toString()}`;
}

// Phase 5: anomaly detection
export interface AnomalyRow {
  row_index: number;
  reasons: string[];
  data: Record<string, string>;
}

export interface AnomalyResponse {
  analysis_id: string;
  anomalies: AnomalyRow[];
  total: number;
  conf_threshold: number;
}

export async function fetchAnalysisAnomalies(
  analysisId: string,
  confThreshold?: number
): Promise<AnomalyResponse> {
  const sp = new URLSearchParams();
  if (confThreshold != null) sp.set("conf_threshold", String(confThreshold));
  const qs = sp.toString();
  return apiFetch<AnomalyResponse>(`/analyses/${analysisId}/anomalies${qs ? `?${qs}` : ""}`);
}

// Phase 5: filter rule type
export interface FilterRule {
  col: string;
  op: "eq" | "ne" | "contains" | "gt" | "lt" | "gte" | "lte";
  val: string;
}

// Phase 5: saved filters
export interface SavedFilter {
  id: string;
  name: string;
  entity_type: string;
  filter_config: {
    filters?: FilterRule[];
    search?: string;
    sort_col?: string;
    sort_order?: string;
  };
  created_at: string;
}

export async function fetchSavedFilters(entityType?: string): Promise<SavedFilter[]> {
  const sp = new URLSearchParams();
  if (entityType) sp.set("entity_type", entityType);
  const qs = sp.toString();
  return apiFetch<SavedFilter[]>(`/saved-filters${qs ? `?${qs}` : ""}`);
}

export async function createSavedFilter(
  name: string,
  entityType: string,
  filterConfig: SavedFilter["filter_config"]
): Promise<SavedFilter> {
  return apiFetch<SavedFilter>("/saved-filters", {
    method: "POST",
    body: JSON.stringify({ name, entity_type: entityType, filter_config: filterConfig }),
  });
}

export async function updateSavedFilter(
  filterId: string,
  update: { name?: string; filter_config?: SavedFilter["filter_config"] }
): Promise<SavedFilter> {
  return apiFetch<SavedFilter>(`/saved-filters/${filterId}`, {
    method: "PUT",
    body: JSON.stringify(update),
  });
}

export async function deleteSavedFilter(filterId: string): Promise<{ deleted: boolean }> {
  return apiFetch<{ deleted: boolean }>(`/saved-filters/${filterId}`, {
    method: "DELETE",
  });
}

// ---------------------------------------------------------------------------
// Phase 3: Analytics & Visualization types + fetch functions
// ---------------------------------------------------------------------------

export interface DistributionData {
  distributions: Record<string, Record<string, number>>;
}

export interface SegmentGroup {
  group: string;
  count: number;
  mean: number;
  median: number;
  std: number;
}

export interface SegmentStatsData {
  group_by: string;
  metric_col: string;
  groups: SegmentGroup[];
}

export interface CrossCompareData {
  analysis_ids: string[];
  columns: string[];
  per_analysis: Record<string, Record<string, Record<string, number>>>;
  disagreement_rates: Record<string, number>;
}

export async function fetchAnalysisDistributions(
  analysisId: string,
  columns: string[]
): Promise<DistributionData> {
  const sp = new URLSearchParams({ columns: columns.join(",") });
  return apiFetch<DistributionData>(`/analyses/${analysisId}/distributions?${sp.toString()}`);
}

export async function fetchAnalysisSegmentStats(
  analysisId: string,
  groupBy: string,
  metricCol: string
): Promise<SegmentStatsData> {
  const sp = new URLSearchParams({ group_by: groupBy, metric_col: metricCol });
  return apiFetch<SegmentStatsData>(`/analyses/${analysisId}/segment-stats?${sp.toString()}`);
}

export async function fetchCrossCompare(
  analysisIds: string[],
  columns: string[]
): Promise<CrossCompareData> {
  return apiFetch<CrossCompareData>("/analyses/cross-compare", {
    method: "POST",
    body: JSON.stringify({ analysis_ids: analysisIds, columns }),
  });
}
