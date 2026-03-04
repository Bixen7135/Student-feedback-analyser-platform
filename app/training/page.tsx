"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import {
  fetchDatasets,
  fetchDatasetSchema,
  fetchColumnRoles,
  fetchDatasetVersions,
  fetchDatasetBranches,
  fetchModels,
  fetchTrainingContract,
  startTraining,
  DatasetSummary,
  DatasetVersion,
  DatasetBranch,
  ModelSummary,
  TrainingConfigRequest,
  TrainingContractResponse,
  TrainingModelType,
  TrainingTask,
  TrainingBalancing,
  TrainingActivation,
  StartTrainingRequest,
} from "@/app/lib/api";
import { formatLocalizedDate, useDateTimeLocale } from "@/app/lib/i18n/date-time";

type Task = TrainingTask;
type ModelType = TrainingModelType;
type Balancing = TrainingBalancing;

const TASK_LABELS: Record<Task, string> = {
  language: "Language Detection",
  sentiment: "Sentiment Classification",
  detail_level: "Detail Level",
};

const TASK_LABEL_COLS: Record<Task, string> = {
  language: "language",
  sentiment: "sentiment",
  detail_level: "detail_level",
};

const MODEL_LABELS: Record<ModelType, string> = {
  tfidf: "TF-IDF + Logistic Regression",
  char_ngram: "Char N-gram + Logistic Regression",
  xlm_roberta: "XLM-RoBERTa",
};

const MODEL_DESCRIPTIONS: Record<ModelType, string> = {
  tfidf: "Sparse word features with logistic regression.",
  char_ngram: "Character n-grams with logistic regression.",
  xlm_roberta: "Transformer fine-tuning for multilingual text.",
};

const STEP_LABELS = [
  "Select Dataset",
  "Select Task",
  "Configure",
  "Review & Launch",
];

const DEFAULT_TRAINING_DATASET_PATH = "/mnt/data/dataset.csv";

const PARAMETER_LABELS: Record<string, string> = {
  train_ratio: "Train ratio",
  val_ratio: "Val ratio",
  test_ratio: "Test ratio",
  class_balancing: "Class balancing",
  max_features: "Max features",
  C: "Regularization (C)",
  max_iter: "Max iterations",
  pretrained_model: "Pretrained model",
  max_seq_length: "Max sequence length",
  batch_size: "Batch size",
  epochs: "Epochs",
  learning_rate: "Learning rate",
  weight_decay: "Weight decay",
  warmup_ratio: "Warmup ratio",
  gradient_accumulation_steps: "Gradient accumulation",
  head_hidden_units: "Head hidden units",
  dropout: "Dropout",
  activation: "Activation",
  text_col: "Text column",
  label_col: "Label column",
  loss: "Loss",
};

const FALLBACK_TRAINING_CONTRACT: TrainingContractResponse = {
  model_types: ["tfidf", "char_ngram", "xlm_roberta"],
  classification_loss: "cross_entropy",
  parameters: [
    { name: "loss", applies_to: "all", required: true, default: "cross_entropy" },
    { name: "train_ratio", applies_to: "all", required: true, default: 0.8 },
    { name: "val_ratio", applies_to: "all", required: true, default: 0.1 },
    { name: "test_ratio", applies_to: "all", required: true, default: 0.1 },
    { name: "class_balancing", applies_to: "all", required: true, default: "class_weight" },
    { name: "text_col", applies_to: "all", required: false, default: null },
    { name: "label_col", applies_to: "all", required: false, default: null },
    { name: "max_features", applies_to: ["tfidf", "char_ngram"], required: false, default: null },
    { name: "C", applies_to: ["tfidf", "char_ngram"], required: false, default: null },
    { name: "max_iter", applies_to: ["tfidf", "char_ngram"], required: false, default: null },
    { name: "pretrained_model", applies_to: ["xlm_roberta"], required: true, default: "xlm-roberta-base" },
    { name: "max_seq_length", applies_to: ["xlm_roberta"], required: true, default: 256 },
    { name: "batch_size", applies_to: ["xlm_roberta"], required: true, default: 16 },
    { name: "epochs", applies_to: ["xlm_roberta"], required: true, default: 3 },
    { name: "learning_rate", applies_to: ["xlm_roberta"], required: true, default: 0.00002 },
    { name: "weight_decay", applies_to: ["xlm_roberta"], required: true, default: 0.01 },
    { name: "warmup_ratio", applies_to: ["xlm_roberta"], required: false, default: null },
    { name: "gradient_accumulation_steps", applies_to: ["xlm_roberta"], required: false, default: null },
    { name: "head_hidden_units", applies_to: ["xlm_roberta"], required: false, default: null },
    { name: "dropout", applies_to: ["xlm_roberta"], required: false, default: null },
    { name: "activation", applies_to: ["xlm_roberta"], required: false, default: null },
  ],
};

// All known system field names that a column can be mapped to.
const SYSTEM_ROLES: { value: string; label: string }[] = [
  { value: "", label: "(not used)" },
  { value: "text", label: "text - feedback text" },
  { value: "sentiment", label: "sentiment - sentiment label" },
  { value: "language", label: "language - language label" },
  { value: "detail_level", label: "detail_level - detail level label" },
  { value: "survey_id", label: "survey_id - record ID" },
  ...Array.from({ length: 9 }, (_, i) => ({
    value: `item_${i + 1}`,
    label: `item_${i + 1} - ordinal survey item`,
  })),
];

const SYSTEM_ROLE_VALUES = new Set(
  SYSTEM_ROLES.map((r) => r.value).filter(Boolean)
);

export default function TrainingPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const dateTimeLocale = useDateTimeLocale();

  // Wizard state
  const [step, setStep] = useState(0);
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [datasetsLoading, setDatasetsLoading] = useState(true);
  const [datasetsError, setDatasetsError] = useState<string | null>(null);

  // Form values
  const [selectedDataset, setSelectedDataset] = useState<DatasetSummary | null>(null);
  const [dsVersions, setDsVersions] = useState<DatasetVersion[]>([]);
  const [selectedVersion, setSelectedVersion] = useState<number | null>(null); // null = latest
  const [dsBranches, setDsBranches] = useState<DatasetBranch[]>([]);
  const [selectedBranch, setSelectedBranch] = useState<string | null>(null); // null = default
  const [selectedTask, setSelectedTask] = useState<Task>("sentiment");
  const [selectedModel, setSelectedModel] = useState<ModelType>("tfidf");
  const [trainName, setTrainName] = useState("");
  const [seed, setSeed] = useState(42);
  const [balancing, setBalancing] = useState<Balancing>("class_weight");
  const [trainRatio, setTrainRatio] = useState(0.80);
  const [valRatio, setValRatio] = useState(0.10);
  const [trainingContract, setTrainingContract] = useState<TrainingContractResponse>(
    FALLBACK_TRAINING_CONTRACT
  );
  const [contractError, setContractError] = useState<string | null>(null);

  // Baseline hyperparameters
  const [baselineMaxFeatures, setBaselineMaxFeatures] = useState("");
  const [baselineC, setBaselineC] = useState("");
  const [baselineMaxIter, setBaselineMaxIter] = useState("");

  // Transformer hyperparameters
  const [pretrainedModel, setPretrainedModel] = useState("xlm-roberta-base");
  const [maxSeqLength, setMaxSeqLength] = useState("256");
  const [batchSize, setBatchSize] = useState("16");
  const [epochs, setEpochs] = useState("3");
  const [learningRate, setLearningRate] = useState("0.00002");
  const [weightDecay, setWeightDecay] = useState("0.01");
  const [warmupRatio, setWarmupRatio] = useState("");
  const [gradientAccumulationSteps, setGradientAccumulationSteps] = useState("");
  const [headHiddenUnits, setHeadHiddenUnits] = useState("");
  const [dropout, setDropout] = useState("");
  const [activation, setActivation] = useState<TrainingActivation | "">("");

  // Column role mapping: { datasetColumnName -> systemRole }
  const [datasetColumns, setDatasetColumns] = useState<string[]>([]);
  const [columnRoles, setColumnRoles] = useState<Record<string, string>>({});

  // Fine-tuning state
  const [fineTuneEnabled, setFineTuneEnabled] = useState(false);
  const [baseModelId, setBaseModelId] = useState<string | null>(null);
  const [availableBaseModels, setAvailableBaseModels] = useState<ModelSummary[]>([]);
  const [baseModelsLoading, setBaseModelsLoading] = useState(false);

  // Launch state
  const [launching, setLaunching] = useState(false);
  const [launchError, setLaunchError] = useState<string | null>(null);
  const [queryPrefillApplied, setQueryPrefillApplied] = useState(false);

  const selectedVersionMeta =
    selectedVersion == null
      ? dsVersions[0] ?? null
      : dsVersions.find((v) => v.version === selectedVersion) ?? null;
  const selectedVersionId = selectedVersionMeta?.id;

  // Derive text_col and label_col from the mapping table
  const textCol =
    Object.entries(columnRoles).find(([, r]) => r === "text")?.[0] ?? "";
  const labelCol =
    Object.entries(columnRoles).find(
      ([, r]) => r === TASK_LABEL_COLS[selectedTask]
    )?.[0] ?? "";

  const modelContractFields = trainingContract.parameters.filter((parameter) => {
    if (parameter.applies_to === "all") {
      return false;
    }
    return parameter.applies_to.includes(selectedModel);
  });

  const baselineContractFields = modelContractFields.filter(
    (parameter) => selectedModel !== "xlm_roberta" && parameter.applies_to !== "all"
  );

  function getParameterLabel(name: string): string {
    return PARAMETER_LABELS[name] ?? name.replace(/_/g, " ");
  }

  function formatSummaryValue(value: unknown): string {
    if (value === null || value === undefined || value === "") return "(not set)";
    if (typeof value === "number") return Number.isFinite(value) ? String(value) : "(not set)";
    if (typeof value === "boolean") return value ? "true" : "false";
    return String(value);
  }

  function parseOptionalNumber(value: string, parser: (input: string) => number): number | undefined {
    const trimmed = value.trim();
    if (!trimmed) return undefined;
    const parsed = parser(trimmed);
    return Number.isFinite(parsed) ? parsed : undefined;
  }

  function buildTrainingConfig(): TrainingConfigRequest {
    const config: TrainingConfigRequest = {
      train_ratio: trainRatio,
      val_ratio: valRatio,
      test_ratio: testRatio,
      class_balancing: balancing,
      text_col: textCol || undefined,
      label_col: labelCol || undefined,
    };

    if (selectedModel === "xlm_roberta") {
      config.pretrained_model = pretrainedModel.trim() || undefined;
      config.max_seq_length = parseOptionalNumber(maxSeqLength, (value) => parseInt(value, 10));
      config.batch_size = parseOptionalNumber(batchSize, (value) => parseInt(value, 10));
      config.epochs = parseOptionalNumber(epochs, (value) => parseInt(value, 10));
      config.learning_rate = parseOptionalNumber(learningRate, Number);
      config.weight_decay = parseOptionalNumber(weightDecay, Number);
      config.warmup_ratio = parseOptionalNumber(warmupRatio, Number);
      config.gradient_accumulation_steps = parseOptionalNumber(
        gradientAccumulationSteps,
        (value) => parseInt(value, 10)
      );
      config.head_hidden_units = parseOptionalNumber(headHiddenUnits, (value) => parseInt(value, 10));
      config.dropout = parseOptionalNumber(dropout, Number);
      config.activation = activation || undefined;
      return config;
    }

    config.max_features = parseOptionalNumber(baselineMaxFeatures, (value) => parseInt(value, 10));
    config.C = parseOptionalNumber(baselineC, Number);
    config.max_iter = parseOptionalNumber(baselineMaxIter, (value) => parseInt(value, 10));
    return config;
  }

  const modelSpecificSummaryRows: Array<[string, string]> =
    selectedModel === "xlm_roberta"
      ? [
          ["Loss", trainingContract.classification_loss],
          ["Pretrained model", formatSummaryValue(pretrainedModel.trim())],
          ["Max sequence length", formatSummaryValue(parseOptionalNumber(maxSeqLength, (value) => parseInt(value, 10)))],
          ["Batch size", formatSummaryValue(parseOptionalNumber(batchSize, (value) => parseInt(value, 10)))],
          ["Epochs", formatSummaryValue(parseOptionalNumber(epochs, (value) => parseInt(value, 10)))],
          ["Learning rate", formatSummaryValue(parseOptionalNumber(learningRate, Number))],
          ["Weight decay", formatSummaryValue(parseOptionalNumber(weightDecay, Number))],
          ["Warmup ratio", formatSummaryValue(parseOptionalNumber(warmupRatio, Number))],
          [
            "Gradient accumulation",
            formatSummaryValue(parseOptionalNumber(gradientAccumulationSteps, (value) => parseInt(value, 10))),
          ],
          ["Head hidden units", formatSummaryValue(parseOptionalNumber(headHiddenUnits, (value) => parseInt(value, 10)))],
          ["Dropout", formatSummaryValue(parseOptionalNumber(dropout, Number))],
          ["Activation", formatSummaryValue(activation)],
        ]
      : [
          ["Loss", trainingContract.classification_loss],
          ["Max features", formatSummaryValue(parseOptionalNumber(baselineMaxFeatures, (value) => parseInt(value, 10)))],
          ["Regularization (C)", formatSummaryValue(parseOptionalNumber(baselineC, Number))],
          ["Max iterations", formatSummaryValue(parseOptionalNumber(baselineMaxIter, (value) => parseInt(value, 10)))],
        ];

  function inferFallbackRole(columnName: string): string {
    const normalized = columnName.trim().toLowerCase();
    if (SYSTEM_ROLE_VALUES.has(normalized)) return normalized;
    if (normalized === "text_feedback") return "text";
    if (normalized === "sentiment_class") return "sentiment";
    return "";
  }

  useEffect(() => {
    fetchDatasets({ sort: "created_at", order: "desc", per_page: 100 })
      .then((r) => {
        setDatasets(r.datasets.filter((d) => d.status === "active"));
        setDatasetsLoading(false);
      })
      .catch((e) => {
        setDatasetsError(e.message);
        setDatasetsLoading(false);
      });
  }, []);

  useEffect(() => {
    fetchTrainingContract()
      .then((contract) => {
        setTrainingContract(contract);
        setContractError(null);
      })
      .catch((error: unknown) => {
        setContractError(error instanceof Error ? error.message : "Unable to load training contract.");
      });
  }, []);

  useEffect(() => {
    if (queryPrefillApplied || datasets.length === 0) return;
    const datasetIdFromQuery = searchParams.get("dataset_id");
    if (!datasetIdFromQuery) {
      setQueryPrefillApplied(true);
      return;
    }
    const ds = datasets.find((d) => d.id === datasetIdFromQuery) ?? null;
    if (ds) setSelectedDataset(ds);
    setQueryPrefillApplied(true);
  }, [datasets, queryPrefillApplied, searchParams]);

  // Fetch versions when dataset changes; auto-select latest
  useEffect(() => {
    if (!selectedDataset) {
      setDsVersions([]);
      setSelectedVersion(null);
      return;
    }
    // When selectedBranch is null, use default_branch_id to get only default branch versions
    const branchToFetch = selectedBranch ?? selectedDataset.default_branch_id ?? undefined;
    fetchDatasetVersions(selectedDataset.id, branchToFetch)
      .then((vs) => {
        setDsVersions(vs);
        setSelectedVersion(null); // default to latest
      })
      .catch(() => {});
  }, [selectedDataset, selectedBranch]);

  // Fetch branches when dataset changes; auto-select default
  useEffect(() => {
    if (!selectedDataset) {
      setDsBranches([]);
      setSelectedBranch(null);
      return;
    }
    fetchDatasetBranches(selectedDataset.id)
      .then((branches) => {
        setDsBranches(branches);
        setSelectedBranch(null); // default to default branch
      })
      .catch(() => {});
  }, [selectedDataset]);

  // Fetch dataset schema and auto-populate column roles whenever dataset/version changes.
  useEffect(() => {
    if (!selectedDataset) {
      setDatasetColumns([]);
      setColumnRoles({});
      return;
    }
    fetchDatasetSchema(
      selectedDataset.id,
      selectedVersionId ? { version_id: selectedVersionId } : undefined
    )
      .then((res) => {
        const cols = res.columns.map((c: { name: string }) => c.name);
        setDatasetColumns(cols);
        return fetchColumnRoles(
          selectedDataset.id,
          selectedVersionId ? { version_id: selectedVersionId } : undefined
        )
          .then(({ column_roles }) => {
            const mapped: Record<string, string> = {};
            for (const col of cols) {
              mapped[col] = column_roles[col] ?? inferFallbackRole(col);
            }
            setColumnRoles(mapped);
          })
          .catch(() => {
            const auto: Record<string, string> = {};
            for (const col of cols) {
              auto[col] = inferFallbackRole(col);
            }
            setColumnRoles(auto);
          });
      })
      .catch(() => {/* schema fetch is best-effort */});
  }, [selectedDataset, selectedVersionId]);

  // Fetch models available for fine-tuning (same task + model type)
  useEffect(() => {
    if (!fineTuneEnabled) return;
    setBaseModelsLoading(true);
    setBaseModelId(null);
    fetchModels({ task: selectedTask, model_type: selectedModel, per_page: 100 })
      .then((r) => {
        setAvailableBaseModels(r.models);
        setBaseModelsLoading(false);
      })
      .catch(() => setBaseModelsLoading(false));
  }, [fineTuneEnabled, selectedTask, selectedModel]);

  // Reset fine-tune state when toggled off
  useEffect(() => {
    if (!fineTuneEnabled) {
      setBaseModelId(null);
      setAvailableBaseModels([]);
    }
  }, [fineTuneEnabled]);

  const testRatio = Math.max(0.02, parseFloat((1 - trainRatio - valRatio).toFixed(3)));

  async function handleLaunch() {
    setLaunching(true);
    setLaunchError(null);
    try {
      const config = buildTrainingConfig();
      const req: StartTrainingRequest = {
        dataset_id: selectedDataset?.id ?? null,
        data_path: selectedDataset ? undefined : DEFAULT_TRAINING_DATASET_PATH,
        dataset_version: selectedDataset ? selectedVersion : null,
        branch_id: selectedDataset ? selectedBranch : null,
        task: selectedTask,
        model_type: selectedModel,
        seed,
        name: trainName.trim() || undefined,
        base_model_id: fineTuneEnabled ? baseModelId : undefined,
        config,
      };
      const job = await startTraining(req);
      router.push(`/training/${job.job_id}`);
    } catch (e: unknown) {
      setLaunchError(e instanceof Error ? e.message : String(e));
      setLaunching(false);
    }
  }

  const card = {
    background: "var(--bg-surface)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius-unified)",
    padding: "24px",
    marginBottom: "16px",
  } as const;

  const inputStyle = {
    background: "var(--bg-base)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius-unified)",
    padding: "8px 12px",
    color: "var(--text-primary)",
    fontSize: "13px",
    fontFamily: "var(--font-jetbrains)",
    width: "100%",
    boxSizing: "border-box" as const,
  } as const;

  const selectStyle = { ...inputStyle };

  const datasetStepSelectStyle = {
    width: "100%",
    background: "var(--bg-elevated)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius-unified)",
    padding: "7px 12px",
    fontFamily: "var(--font-jetbrains)",
    fontSize: "12px",
    color: "var(--text-primary)",
    outline: "none",
  } as const;

  const btnPrimary = {
    background: "var(--gold)",
    color: "#000",
    border: "none",
    borderRadius: "var(--radius-unified)",
    padding: "9px 20px",
    fontSize: "13px",
    fontWeight: 600,
    cursor: "pointer",
    fontFamily: "var(--font-jetbrains)",
  } as const;

  const btnGhost = {
    background: "transparent",
    color: "var(--text-secondary)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius-unified)",
    padding: "9px 20px",
    fontSize: "13px",
    cursor: "pointer",
    fontFamily: "var(--font-jetbrains)",
  } as const;

  const label = {
    fontSize: "11px",
    color: "var(--text-tertiary)",
    fontFamily: "var(--font-syne)",
    fontWeight: 700,
    letterSpacing: "0.1em",
    textTransform: "uppercase" as const,
    marginBottom: "6px",
    display: "block",
  } as const;

  return (
    <div
      className={`page-shell page-standard ${step === 0 ? "page-shell--xs" : "page-shell--sm"} animate-fade-up`}
    >
      {/* Back link */}
      <div style={{ marginBottom: "6px" }}>
        <button
          type="button"
          onClick={() => router.back()}
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "0.35rem",
            color: "var(--text-tertiary)",
            background: "none",
            border: "none",
            padding: 0,
            textDecoration: "none",
            fontFamily: "var(--font-jetbrains)",
            fontSize: "11px",
            cursor: "pointer",
          }}
        >
          <span aria-hidden="true">&larr;</span>
          <span>Back</span>
        </button>
      </div>

      {/* Header */}
      <div style={{ marginBottom: "28px" }}>
        <div style={{ minWidth: 0 }}>
          <h1
            style={{
              fontFamily: "var(--font-syne)",
              fontSize: "20px",
              fontWeight: 700,
              color: "var(--text-primary)",
              marginBottom: "6px",
            }}
          >
            Train a Classifier
          </h1>
          <p style={{ fontSize: "13px", color: "var(--text-secondary)" }}>
            Train a text classifier on an uploaded dataset and register it in the
            model registry.
          </p>
        </div>
      </div>

      {/* Step indicator */}
      <div className="flex flex-wrap items-center gap-y-3" style={{ marginBottom: "28px" }}>
        {STEP_LABELS.map((stepLabel, i) => (
          <div key={i} className="flex items-center">
            <div
              className="flex items-center gap-2"
              style={{ cursor: i < step ? "pointer" : "default" }}
              onClick={() => i < step && setStep(i)}
            >
              <div
                className="flex items-center justify-center rounded-full flex-shrink-0"
                style={{
                  width: "24px",
                  height: "24px",
                  background:
                    i === step
                      ? "var(--gold)"
                      : i < step
                      ? "var(--success)"
                      : "var(--bg-surface)",
                  border:
                    i === step
                      ? "none"
                      : i < step
                      ? "none"
                      : "1px solid var(--border)",
                  color: i <= step ? "#000" : "var(--text-tertiary)",
                  fontSize: "11px",
                  fontWeight: 700,
                  fontFamily: "var(--font-jetbrains)",
                }}
              >
                {i < step ? "✓" : i + 1}
              </div>
              <span
                style={{
                  fontSize: "11px",
                  color:
                    i === step
                      ? "var(--text-primary)"
                      : "var(--text-tertiary)",
                  fontFamily: "var(--font-syne)",
                  fontWeight: i === step ? 700 : 400,
                  whiteSpace: "nowrap",
                }}
              >
                {stepLabel}
              </span>
            </div>
            {i < STEP_LABELS.length - 1 && (
              <div
                style={{
                  width: "32px",
                  height: "1px",
                  background:
                    i < step ? "var(--success)" : "var(--border-dim)",
                  margin: "0 8px",
                  flexShrink: 0,
                }}
              />
            )}
          </div>
        ))}
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Step 0: Select Dataset */}
      {/* ------------------------------------------------------------------ */}
      {step === 0 && (
        <div className="space-y-5 animate-fade-up">
          <div
            className="rounded-xl"
            style={{
              background: "var(--bg-surface)",
              border: "1px solid var(--border-dim)",
              padding: "24px",
            }}
          >
            <div style={{ marginBottom: "20px" }}>
              <label
                style={{
                  fontFamily: "var(--font-syne)",
                  fontSize: "9.5px",
                  fontWeight: 700,
                  letterSpacing: "0.14em",
                  textTransform: "uppercase",
                  color: "var(--text-tertiary)",
                  marginBottom: "8px",
                  display: "block",
                }}
              >
                Dataset
              </label>
              {datasetsLoading ? (
                <div
                  style={{
                    fontSize: "12px",
                    color: "var(--text-tertiary)",
                    fontFamily: "var(--font-jetbrains)",
                  }}
                >
                  Loading datasets...
                </div>
              ) : datasetsError ? (
                <div
                  style={{
                    fontSize: "12px",
                    color: "var(--error)",
                    fontFamily: "var(--font-jetbrains)",
                  }}
                >
                  {datasetsError}
                </div>
              ) : datasets.length === 0 ? (
                <p style={{ fontSize: "13px", color: "var(--text-tertiary)" }}>
                  No datasets uploaded yet.{" "}
                  <Link
                    href="/datasets/upload"
                    style={{ color: "var(--gold)", textDecoration: "none" }}
                  >
                    Upload one first.
                  </Link>
                </p>
              ) : (
                <select
                  style={datasetStepSelectStyle}
                  value={selectedDataset?.id ?? ""}
                  onChange={(e) => {
                    const ds = datasets.find((d) => d.id === e.target.value) ?? null;
                    setSelectedDataset(ds);
                  }}
                >
                  <option value="">- select a dataset -</option>
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>
                      {d.name} ({d.row_count.toLocaleString()} rows, v{d.current_version})
                    </option>
                  ))}
                </select>
              )}
            </div>

            {selectedDataset && dsBranches.length > 0 && (
              <div style={{ marginBottom: "20px" }}>
                <label
                  style={{
                    fontFamily: "var(--font-syne)",
                    fontSize: "9.5px",
                    fontWeight: 700,
                    letterSpacing: "0.14em",
                    textTransform: "uppercase",
                    color: "var(--text-tertiary)",
                    marginBottom: "8px",
                    display: "block",
                  }}
                >
                  Branch
                </label>
                <select
                  style={datasetStepSelectStyle}
                  value={selectedBranch ?? ""}
                  onChange={(e) => setSelectedBranch(e.target.value || null)}
                >
                  <option value="">- default branch -</option>
                  {dsBranches.map((b) => (
                    <option key={b.id} value={b.id}>
                      {b.name}
                      {b.is_default ? " (default)" : ""}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {selectedDataset && dsVersions.length > 0 && (
              <div>
                <label
                  style={{
                    fontFamily: "var(--font-syne)",
                    fontSize: "9.5px",
                    fontWeight: 700,
                    letterSpacing: "0.14em",
                    textTransform: "uppercase",
                    color: "var(--text-tertiary)",
                    marginBottom: "8px",
                    display: "block",
                  }}
                >
                  Version
                </label>
                <select
                  style={datasetStepSelectStyle}
                  value={selectedVersion ?? ""}
                  onChange={(e) =>
                    setSelectedVersion(e.target.value ? Number(e.target.value) : null)
                  }
                >
                  <option value="">- latest -</option>
                  {dsVersions.map((v) => (
                    <option key={v.id} value={v.version}>
                      v{v.version} - {v.reason || "no reason"} ({v.row_count.toLocaleString()} rows)
                    </option>
                  ))}
                </select>
              </div>
            )}
          </div>

          {!selectedDataset && !datasetsLoading && datasets.length > 0 && (
            <div
              style={{
                fontSize: "11px",
                color: "var(--text-tertiary)",
                fontFamily: "var(--font-jetbrains)",
                textAlign: "center",
              }}
            >
              No dataset selected - the pipeline will use the default{" "}
              <span style={{ color: "var(--text-secondary)" }}>
                /mnt/data/dataset.csv
              </span>
            </div>
          )}

          <div>
            <button
              className="w-full rounded-lg flex items-center justify-center gap-2 transition-all duration-150"
              style={{
                background: "var(--gold)",
                color: "#08080B",
                padding: "11px 24px",
                fontSize: "13px",
                fontWeight: 600,
                fontFamily: "var(--font-syne)",
                letterSpacing: "0.05em",
                border: "none",
                cursor: "pointer",
              }}
              onClick={() => setStep(1)}
            >
              Next
              <svg width="13" height="13" viewBox="0 0 13 13" fill="none" aria-hidden="true">
                <path
                  d="M3 6.5H10M7.5 4L10 6.5L7.5 9"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Step 1: Select Task */}
      {/* ------------------------------------------------------------------ */}
      {step === 1 && (
        <div style={card}>
          <span style={label}>Task</span>
          <div className="flex flex-col gap-2" style={{ marginBottom: "20px" }}>
            {(Object.keys(TASK_LABELS) as Task[]).map((t) => (
              <label
                key={t}
                className="flex items-center gap-3"
                style={{
                  padding: "12px 14px",
                  borderRadius: "var(--radius-unified)",
                  border: `1px solid ${selectedTask === t ? "var(--gold)" : "var(--border)"}`,
                  background:
                    selectedTask === t
                      ? "var(--gold-faint)"
                      : "var(--bg-base)",
                  cursor: "pointer",
                }}
              >
                <input
                  type="radio"
                  name="task"
                  value={t}
                  checked={selectedTask === t}
                  onChange={() => setSelectedTask(t)}
                  style={{ accentColor: "var(--gold)" }}
                />
                <div>
                  <div
                    style={{
                      fontSize: "13px",
                      fontWeight: 600,
                      color: "var(--text-primary)",
                    }}
                  >
                    {TASK_LABELS[t]}
                  </div>
                  <div
                    style={{
                      fontSize: "11px",
                      color: "var(--text-tertiary)",
                      fontFamily: "var(--font-jetbrains)",
                    }}
                  >
                    label column: {TASK_LABEL_COLS[t]}
                  </div>
                </div>
              </label>
            ))}
          </div>

          <span style={{ ...label, marginTop: "16px" }}>Model Type</span>
          <div className="flex flex-col gap-2" style={{ marginBottom: "20px" }}>
            {(Object.keys(MODEL_LABELS) as ModelType[]).map((m) => (
              <label
                key={m}
                className="flex items-center gap-3"
                style={{
                  padding: "12px 14px",
                  borderRadius: "var(--radius-unified)",
                  border: `1px solid ${selectedModel === m ? "var(--gold)" : "var(--border)"}`,
                  background:
                    selectedModel === m
                      ? "var(--gold-faint)"
                      : "var(--bg-base)",
                  cursor: "pointer",
                }}
              >
                <input
                  type="radio"
                  name="modelType"
                  value={m}
                  checked={selectedModel === m}
                  onChange={() => setSelectedModel(m)}
                  style={{ accentColor: "var(--gold)" }}
                />
                <div>
                  <div
                    style={{
                      fontSize: "13px",
                      fontWeight: 600,
                      color: "var(--text-primary)",
                    }}
                  >
                    {MODEL_LABELS[m]}
                  </div>
                  <div
                    style={{
                      fontSize: "11px",
                      color: "var(--text-tertiary)",
                      fontFamily: "var(--font-jetbrains)",
                    }}
                  >
                    {MODEL_DESCRIPTIONS[m]}
                  </div>
                </div>
              </label>
            ))}
          </div>

          <div className="flex justify-between">
            <button style={btnGhost} onClick={() => setStep(0)}>
              <span aria-hidden="true">← </span>
              <span>Back</span>
            </button>
            <button style={btnPrimary} onClick={() => setStep(2)}>
              <span>Next</span>
              <span aria-hidden="true"> →</span>
            </button>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Step 2: Configure */}
      {/* ------------------------------------------------------------------ */}
      {step === 2 && (
        <div style={card}>
          <div style={{ marginBottom: "16px" }}>
            <span style={label}>Model Name (optional)</span>
            <input
              style={inputStyle}
              placeholder="e.g. sentiment_tfidf_v1"
              value={trainName}
              onChange={(e) => setTrainName(e.target.value)}
            />
          </div>

          <div className="flex gap-4" style={{ marginBottom: "16px", flexWrap: "wrap" }}>
            <div style={{ flex: "1 1 12rem", minWidth: "min(100%, 12rem)" }}>
              <span style={label}>Train ratio</span>
              <input
                type="number"
                style={inputStyle}
                min={0.5}
                max={0.9}
                step={0.05}
                value={trainRatio}
                onChange={(e) => setTrainRatio(parseFloat(e.target.value))}
              />
            </div>
            <div style={{ flex: "1 1 12rem", minWidth: "min(100%, 12rem)" }}>
              <span style={label}>Val ratio</span>
              <input
                type="number"
                style={inputStyle}
                min={0.02}
                max={0.3}
                step={0.05}
                value={valRatio}
                onChange={(e) => setValRatio(parseFloat(e.target.value))}
              />
            </div>
            <div style={{ flex: "1 1 12rem", minWidth: "min(100%, 12rem)" }}>
              <span style={label}>Test ratio</span>
              <div
                style={{
                  ...inputStyle,
                  color: "var(--text-tertiary)",
                  display: "flex",
                  alignItems: "center",
                }}
              >
                {testRatio.toFixed(2)} (auto)
              </div>
            </div>
          </div>

          <div style={{ marginBottom: "16px" }}>
            <span style={label}>Class balancing</span>
            <select
              style={selectStyle}
              value={balancing}
              onChange={(e) => setBalancing(e.target.value as Balancing)}
            >
              <option value="class_weight">
                class_weight (balanced weights in LR)
              </option>
              <option value="oversample">
                oversample (random oversampling of minority classes)
              </option>
              <option value="none">none (raw class frequencies)</option>
            </select>
          </div>

          <div style={{ marginBottom: "20px" }}>
            <span style={label}>Random seed</span>
            <input
              type="number"
              style={inputStyle}
              min={0}
              value={seed}
              onChange={(e) => setSeed(parseInt(e.target.value, 10))}
            />
          </div>

          <div style={{ marginBottom: "20px" }}>
            <div
              className="flex items-center justify-between gap-3"
              style={{ marginBottom: "10px", flexWrap: "wrap" }}
            >
              <span style={label}>Model Hyperparameters</span>
              <span
                style={{
                  fontSize: "10px",
                  color: "var(--text-tertiary)",
                  fontFamily: "var(--font-jetbrains)",
                }}
              >
                {contractError ? "Using fallback contract" : "Loaded from platform contract"}
              </span>
            </div>

            {selectedModel === "xlm_roberta" ? (
              <div className="flex flex-col gap-4">
                <div style={{ marginBottom: "4px" }}>
                  <span style={label}>{getParameterLabel("pretrained_model")}</span>
                  <input
                    style={inputStyle}
                    value={pretrainedModel}
                    onChange={(e) => setPretrainedModel(e.target.value)}
                  />
                </div>

                <div className="flex gap-4" style={{ flexWrap: "wrap" }}>
                  <div style={{ flex: "1 1 12rem", minWidth: "min(100%, 12rem)" }}>
                    <span style={label}>{getParameterLabel("max_seq_length")}</span>
                    <input
                      type="number"
                      min={1}
                      style={inputStyle}
                      value={maxSeqLength}
                      onChange={(e) => setMaxSeqLength(e.target.value)}
                    />
                  </div>
                  <div style={{ flex: "1 1 12rem", minWidth: "min(100%, 12rem)" }}>
                    <span style={label}>{getParameterLabel("batch_size")}</span>
                    <input
                      type="number"
                      min={1}
                      style={inputStyle}
                      value={batchSize}
                      onChange={(e) => setBatchSize(e.target.value)}
                    />
                  </div>
                  <div style={{ flex: "1 1 12rem", minWidth: "min(100%, 12rem)" }}>
                    <span style={label}>{getParameterLabel("epochs")}</span>
                    <input
                      type="number"
                      min={1}
                      style={inputStyle}
                      value={epochs}
                      onChange={(e) => setEpochs(e.target.value)}
                    />
                  </div>
                </div>

                <div className="flex gap-4" style={{ flexWrap: "wrap" }}>
                  <div style={{ flex: "1 1 12rem", minWidth: "min(100%, 12rem)" }}>
                    <span style={label}>{getParameterLabel("learning_rate")}</span>
                    <input
                      type="number"
                      min={0}
                      step="0.000001"
                      style={inputStyle}
                      value={learningRate}
                      onChange={(e) => setLearningRate(e.target.value)}
                    />
                  </div>
                  <div style={{ flex: "1 1 12rem", minWidth: "min(100%, 12rem)" }}>
                    <span style={label}>{getParameterLabel("weight_decay")}</span>
                    <input
                      type="number"
                      min={0}
                      step="0.001"
                      style={inputStyle}
                      value={weightDecay}
                      onChange={(e) => setWeightDecay(e.target.value)}
                    />
                  </div>
                </div>

                <div
                  style={{
                    border: "1px solid var(--border)",
                    borderRadius: "var(--radius-unified)",
                    padding: "14px",
                    background: "var(--bg-base)",
                  }}
                >
                  <div
                    style={{
                      fontSize: "11px",
                      color: "var(--text-tertiary)",
                      fontFamily: "var(--font-syne)",
                      fontWeight: 700,
                      letterSpacing: "0.1em",
                      textTransform: "uppercase",
                      marginBottom: "12px",
                    }}
                  >
                    Optional Transformer Knobs
                  </div>
                  <div className="flex gap-4" style={{ flexWrap: "wrap" }}>
                    <div style={{ flex: "1 1 12rem", minWidth: "min(100%, 12rem)" }}>
                      <span style={label}>{getParameterLabel("warmup_ratio")}</span>
                      <input
                        type="number"
                        min={0}
                        step="0.01"
                        style={inputStyle}
                        placeholder="Leave empty"
                        value={warmupRatio}
                        onChange={(e) => setWarmupRatio(e.target.value)}
                      />
                    </div>
                    <div style={{ flex: "1 1 12rem", minWidth: "min(100%, 12rem)" }}>
                      <span style={label}>{getParameterLabel("gradient_accumulation_steps")}</span>
                      <input
                        type="number"
                        min={1}
                        style={inputStyle}
                        placeholder="Leave empty"
                        value={gradientAccumulationSteps}
                        onChange={(e) => setGradientAccumulationSteps(e.target.value)}
                      />
                    </div>
                    <div style={{ flex: "1 1 12rem", minWidth: "min(100%, 12rem)" }}>
                      <span style={label}>{getParameterLabel("head_hidden_units")}</span>
                      <input
                        type="number"
                        min={1}
                        style={inputStyle}
                        placeholder="Leave empty"
                        value={headHiddenUnits}
                        onChange={(e) => setHeadHiddenUnits(e.target.value)}
                      />
                    </div>
                    <div style={{ flex: "1 1 12rem", minWidth: "min(100%, 12rem)" }}>
                      <span style={label}>{getParameterLabel("dropout")}</span>
                      <input
                        type="number"
                        min={0}
                        max={1}
                        step="0.05"
                        style={inputStyle}
                        placeholder="Leave empty"
                        value={dropout}
                        onChange={(e) => setDropout(e.target.value)}
                      />
                    </div>
                    <div style={{ flex: "1 1 12rem", minWidth: "min(100%, 12rem)" }}>
                      <span style={label}>{getParameterLabel("activation")}</span>
                      <select
                        style={selectStyle}
                        value={activation}
                        onChange={(e) => setActivation(e.target.value as TrainingActivation | "")}
                      >
                        <option value="">Leave empty</option>
                        <option value="relu">relu</option>
                        <option value="gelu">gelu</option>
                        <option value="tanh">tanh</option>
                      </select>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div
                style={{
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius-unified)",
                  padding: "14px",
                  background: "var(--bg-base)",
                }}
              >
                <div
                  style={{
                    fontSize: "11px",
                    color: "var(--text-tertiary)",
                    fontFamily: "var(--font-jetbrains)",
                    marginBottom: "12px",
                  }}
                >
                  {`Baseline controls match the current training contract for ${selectedModel}.`}
                </div>
                <div className="flex gap-4" style={{ flexWrap: "wrap" }}>
                  {baselineContractFields.some((parameter) => parameter.name === "max_features") && (
                    <div style={{ flex: "1 1 12rem", minWidth: "min(100%, 12rem)" }}>
                      <span style={label}>{getParameterLabel("max_features")}</span>
                      <input
                        type="number"
                        min={1}
                        style={inputStyle}
                        placeholder="Leave empty"
                        value={baselineMaxFeatures}
                        onChange={(e) => setBaselineMaxFeatures(e.target.value)}
                      />
                    </div>
                  )}
                  {baselineContractFields.some((parameter) => parameter.name === "C") && (
                    <div style={{ flex: "1 1 12rem", minWidth: "min(100%, 12rem)" }}>
                      <span style={label}>{getParameterLabel("C")}</span>
                      <input
                        type="number"
                        min={0}
                        step="0.1"
                        style={inputStyle}
                        placeholder="Leave empty"
                        value={baselineC}
                        onChange={(e) => setBaselineC(e.target.value)}
                      />
                    </div>
                  )}
                  {baselineContractFields.some((parameter) => parameter.name === "max_iter") && (
                    <div style={{ flex: "1 1 12rem", minWidth: "min(100%, 12rem)" }}>
                      <span style={label}>{getParameterLabel("max_iter")}</span>
                      <input
                        type="number"
                        min={1}
                        style={inputStyle}
                        placeholder="Leave empty"
                        value={baselineMaxIter}
                        onChange={(e) => setBaselineMaxIter(e.target.value)}
                      />
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Fine-tuning toggle */}
          <div
            style={{
              marginBottom: "20px",
              border: `1px solid ${fineTuneEnabled ? "var(--gold)" : "var(--border)"}`,
              borderRadius: "var(--radius-unified)",
              padding: "14px",
              background: fineTuneEnabled ? "var(--gold-faint)" : "var(--bg-base)",
            }}
          >
            <label
              className="flex items-center gap-3"
              style={{ cursor: "pointer", marginBottom: fineTuneEnabled ? "12px" : 0 }}
            >
              <input
                type="checkbox"
                checked={fineTuneEnabled}
                onChange={(e) => setFineTuneEnabled(e.target.checked)}
                style={{ accentColor: "var(--gold)", width: "16px", height: "16px" }}
              />
              <div>
                <span
                  style={{
                    fontSize: "13px",
                    fontWeight: 600,
                    color: "var(--text-primary)",
                  }}
                >
                  Fine-tune from existing model
                </span>
                <div
                  style={{
                    fontSize: "11px",
                    color: "var(--text-tertiary)",
                    fontFamily: "var(--font-jetbrains)",
                  }}
                >
                  Reuse a compatible registry model of the same task and type. Baselines
                  warm-start linear weights; XLM-R continues fine-tuning from the saved
                  checkpoint.
                </div>
              </div>
            </label>

            {fineTuneEnabled && (
              <div>
                <span style={label}>Base model</span>
                {baseModelsLoading && (
                  <p style={{ fontSize: "12px", color: "var(--text-tertiary)" }}>
                    Loading models...
                  </p>
                )}
                {!baseModelsLoading && availableBaseModels.length === 0 && (
                  <p style={{ fontSize: "12px", color: "var(--text-tertiary)" }}>
                    {`No compatible models found for task "${selectedTask}" / type "${selectedModel}". Train a model first.`}
                  </p>
                )}
                {!baseModelsLoading && availableBaseModels.length > 0 && (
                  <select
                    style={selectStyle}
                    value={baseModelId ?? ""}
                    onChange={(e) =>
                      setBaseModelId(e.target.value || null)
                    }
                  >
                    <option value="">- select base model -</option>
                    {availableBaseModels.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.name} · v{m.version} ·{" "}
                        {formatLocalizedDate(m.created_at, dateTimeLocale)}
                        {m.metrics &&
                        typeof (m.metrics as Record<string, unknown>).val === "object"
                          ? ` · val F1 ${(
                              (
                                (m.metrics as Record<string, unknown>).val as Record<
                                  string,
                                  number
                                >
                              ).macro_f1 ?? 0
                            ).toFixed(3)}`
                          : ""}
                      </option>
                    ))}
                  </select>
                )}
              </div>
            )}
          </div>

          {/* Column role mapping table */}
          {datasetColumns.length > 0 && (
            <div style={{ marginBottom: "16px" }}>
              <span style={label}>Column Roles</span>
              <p style={{ fontSize: "11px", color: "var(--text-tertiary)", marginBottom: "8px", fontFamily: "var(--font-jetbrains)" }}>
                For each column in the dataset, select what it represents in the system.
                Unneeded columns can be left as &quot;(not used)&quot;.
              </p>
              <div
                style={{
                  border: "1px solid var(--border)",
                  borderRadius: "var(--radius-unified)",
                  overflow: "hidden",
                }}
              >
                {/* Header */}
                <div
                  className="flex"
                  style={{
                    background: "var(--bg-base)",
                    padding: "7px 12px",
                    borderBottom: "1px solid var(--border)",
                    flexWrap: "wrap",
                    gap: "4px 12px",
                  }}
                >
                  <span
                    style={{
                      flex: "1 1 12rem",
                      fontSize: "10px",
                      color: "var(--text-tertiary)",
                      fontFamily: "var(--font-syne)",
                      fontWeight: 700,
                      letterSpacing: "0.1em",
                      textTransform: "uppercase",
                    }}
                  >
                    Dataset Column
                  </span>
                  <span
                    style={{
                      flex: "1 1 12rem",
                      fontSize: "10px",
                      color: "var(--text-tertiary)",
                      fontFamily: "var(--font-syne)",
                      fontWeight: 700,
                      letterSpacing: "0.1em",
                      textTransform: "uppercase",
                    }}
                  >
                    System Role
                  </span>
                </div>
                {/* Rows */}
                {datasetColumns.map((col, idx) => (
                  <div
                    key={col}
                    className="flex items-center"
                    style={{
                      padding: "6px 12px",
                      borderBottom:
                        idx < datasetColumns.length - 1
                          ? "1px solid var(--border-dim)"
                          : "none",
                      background:
                        columnRoles[col] === "text" ||
                        columnRoles[col] === TASK_LABEL_COLS[selectedTask]
                          ? "var(--gold-faint)"
                          : "transparent",
                      flexWrap: "wrap",
                      gap: "8px 12px",
                    }}
                  >
                    <span
                      style={{
                        flex: "1 1 12rem",
                        fontSize: "12px",
                        color: "var(--text-primary)",
                        fontFamily: "var(--font-jetbrains)",
                        paddingRight: "8px",
                        wordBreak: "break-all",
                      }}
                    >
                      {col}
                    </span>
                    <div style={{ flex: "1 1 12rem", minWidth: "min(100%, 12rem)" }}>
                      <select
                        value={columnRoles[col] ?? ""}
                        onChange={(e) =>
                          setColumnRoles((prev) => ({
                            ...prev,
                            [col]: e.target.value,
                          }))
                        }
                        style={{
                          ...selectStyle,
                          fontSize: "12px",
                          padding: "5px 8px",
                        }}
                      >
                        {SYSTEM_ROLES.map((r) => (
                          <option key={r.value} value={r.value}>
                            {r.label}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>
                ))}
              </div>

              {/* Derived summary */}
              <div
                style={{
                  marginTop: "8px",
                  fontSize: "11px",
                  color: "var(--text-tertiary)",
                  fontFamily: "var(--font-jetbrains)",
                  display: "flex",
                  gap: "16px",
                }}
              >
                <span>
                  Text:{" "}
                  <span
                    style={{
                      color: textCol
                        ? "var(--success)"
                        : "var(--error, #ef4444)",
                      fontWeight: 600,
                    }}
                  >
                    {textCol || "none - assign text role"}
                  </span>
                </span>
                <span>
                  {`Label (${selectedTask}):`}
                  <span
                    style={{
                      marginLeft: "4px",
                      color: labelCol
                        ? "var(--success)"
                        : "var(--error, #ef4444)",
                      fontWeight: 600,
                    }}
                  >
                    {labelCol ||
                      `none - assign ${TASK_LABEL_COLS[selectedTask]} role`}
                  </span>
                </span>
              </div>
            </div>
          )}

          <div className="flex justify-between">
            <button style={btnGhost} onClick={() => setStep(1)}>
              <span aria-hidden="true">← </span>
              <span>Back</span>
            </button>
            <button style={btnPrimary} onClick={() => setStep(3)}>
              <span>Next</span>
              <span aria-hidden="true"> →</span>
            </button>
          </div>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Step 3: Review & Launch */}
      {/* ------------------------------------------------------------------ */}
      {step === 3 && (
        <div style={card}>
          <div style={{ marginBottom: "20px" }}>
            <div
              style={{
                fontFamily: "var(--font-syne)",
                fontSize: "13px",
                fontWeight: 700,
                color: "var(--text-secondary)",
                marginBottom: "14px",
              }}
            >
              Training Summary
            </div>

            {[
              [
                "Dataset",
                selectedDataset?.name ?? "default",
              ],
              ...(!selectedDataset
                ? [["Source file", DEFAULT_TRAINING_DATASET_PATH]]
                : []),
              [
                "Branch",
                selectedDataset
                  ? (() => {
                      const branch = dsBranches.find((b) => b.id === selectedBranch);
                      return (
                        branch?.name ??
                        dsBranches.find((b) => b.is_default)?.name ??
                        "default"
                      );
                    })()
                  : "(not applicable)",
              ],
              [
                "Version",
                selectedDataset
                  ? selectedVersion
                    ? `v${selectedVersion}`
                    : "(latest on branch)"
                  : "(not applicable)",
              ],
              ["Task", TASK_LABELS[selectedTask]],
              ["Model type", MODEL_LABELS[selectedModel]],
              ["Name", trainName || "(auto-generated)"],
              ...(fineTuneEnabled
                ? [
                    [
                      "Fine-tune from",
                      baseModelId
                        ? (availableBaseModels.find((m) => m.id === baseModelId)?.name ??
                            baseModelId)
                        : "(none selected - fresh training)",
                    ],
                  ]
                : []),
              ["Text column", textCol || "(auto-detect)"],
              ["Label column", labelCol || "(auto-detect)"],
              [
                "Split",
                `${(trainRatio * 100).toFixed(0)} / ${(valRatio * 100).toFixed(0)} / ${(testRatio * 100).toFixed(0)}`,
              ],
              ["Balancing", balancing],
              ["Seed", String(seed)],
              ...modelSpecificSummaryRows,
            ].map(([k, v]) => (
              <div
                key={k}
                className="flex"
                style={{
                  padding: "7px 0",
                  borderBottom: "1px solid var(--border-dim)",
                  fontSize: "13px",
                }}
              >
                <span
                  style={{
                    width: "140px",
                    color: "var(--text-tertiary)",
                    flexShrink: 0,
                    fontFamily: "var(--font-jetbrains)",
                    fontSize: "12px",
                  }}
                >
                  {k}
                </span>
                <span style={{ color: "var(--text-primary)" }}>{v}</span>
              </div>
            ))}
          </div>

          <div
            style={{
              background: "var(--bg-base)",
              borderRadius: "var(--radius-unified)",
              padding: "10px 14px",
              fontSize: "12px",
              color: "var(--text-tertiary)",
              fontFamily: "var(--font-jetbrains)",
              marginBottom: "20px",
              lineHeight: 1.6,
            }}
          >
            Batch only. Training runs in background. Not for real-time or
            individual-level decisions. No causal claims.
          </div>

          {launchError && (
            <div
              style={{
                background: "rgba(239,68,68,0.08)",
                border: "1px solid rgba(239,68,68,0.3)",
                borderRadius: "var(--radius-unified)",
                padding: "10px 14px",
                fontSize: "13px",
                color: "var(--error, #ef4444)",
                marginBottom: "16px",
              }}
            >
              {launchError}
            </div>
          )}

          <div className="flex justify-between">
            <button style={btnGhost} onClick={() => setStep(2)}>
              <span aria-hidden="true">← </span>
              <span>Back</span>
            </button>
            <button
              style={{
                ...btnPrimary,
                opacity: launching ? 0.6 : 1,
                cursor: launching ? "not-allowed" : "pointer",
              }}
              disabled={launching}
              onClick={handleLaunch}
            >
              {launching ? "Launching..." : "Launch Training"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}


