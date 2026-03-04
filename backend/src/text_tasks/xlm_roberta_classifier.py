"""XLM-RoBERTa text classifier with a configurable MLP head."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orjson

from src.text_tasks.base import TextClassifier
from src.utils.reproducibility import set_all_seeds


class XlmRobertaDependencyError(RuntimeError):
    """Raised when optional transformer dependencies are unavailable."""


def ensure_xlm_roberta_runtime_available() -> None:
    """Raise a clean dependency error when transformer extras are unavailable."""
    _TorchRuntime.load()


@dataclass
class _TorchRuntime:
    """Lazy-loaded torch/transformers runtime adapter."""

    torch: Any
    nn: Any
    transformers: Any
    AutoModel: Any
    AutoTokenizer: Any
    AdamW: Any

    @classmethod
    def load(cls) -> "_TorchRuntime":
        try:
            import torch
            import transformers
            from torch import nn
            from torch.optim import AdamW
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - exercised via public error path
            raise XlmRobertaDependencyError(
                "XlmRobertaClassifier requires optional dependencies 'torch' and "
                "'transformers'. Install with: pip install \"sfap-backend[transformers]\""
            ) from None

        return cls(
            torch=torch,
            nn=nn,
            transformers=transformers,
            AutoModel=AutoModel,
            AutoTokenizer=AutoTokenizer,
            AdamW=AdamW,
        )

    def create_tokenizer(self, source: str) -> Any:
        return self._run_quiet_pretrained_load(
            lambda: self.AutoTokenizer.from_pretrained(source)
        )

    def create_encoder(self, source: str) -> Any:
        return self._run_quiet_pretrained_load(
            lambda: self.AutoModel.from_pretrained(source)
        )

    def _run_quiet_pretrained_load(self, loader: Any) -> Any:
        """Suppress noisy Hugging Face load reports during model bootstrap."""
        logging_module = getattr(getattr(self.transformers, "utils", None), "logging", None)
        if logging_module is None:
            return loader()

        previous_verbosity: int | None = None
        if hasattr(logging_module, "get_verbosity"):
            previous_verbosity = logging_module.get_verbosity()

        progress_toggled = False
        if hasattr(logging_module, "disable_progress_bar"):
            logging_module.disable_progress_bar()
            progress_toggled = True

        if hasattr(logging_module, "set_verbosity_error"):
            logging_module.set_verbosity_error()

        try:
            return loader()
        finally:
            if (
                previous_verbosity is not None
                and hasattr(logging_module, "set_verbosity")
            ):
                logging_module.set_verbosity(previous_verbosity)
            if progress_toggled and hasattr(logging_module, "enable_progress_bar"):
                logging_module.enable_progress_bar()

    def build_network(
        self,
        *,
        encoder: Any,
        num_labels: int,
        head_hidden_units: int | None,
        dropout: float,
        activation: str,
    ) -> Any:
        nn = self.nn
        hidden_size = int(getattr(encoder.config, "hidden_size"))

        def _activation(name: str) -> Any:
            if name == "relu":
                return nn.ReLU()
            if name == "tanh":
                return nn.Tanh()
            return nn.GELU()

        class _SequenceClassifier(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.encoder = encoder
                self.classifier = self._build_head()

            def _build_head(self) -> Any:
                layers: list[Any] = []
                input_dim = hidden_size
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                if head_hidden_units is not None:
                    layers.append(nn.Linear(input_dim, head_hidden_units))
                    layers.append(_activation(activation))
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    input_dim = head_hidden_units
                layers.append(nn.Linear(input_dim, num_labels))
                return nn.Sequential(*layers)

            def forward(self, **encoded: Any) -> Any:
                outputs = self.encoder(**encoded)
                last_hidden = getattr(outputs, "last_hidden_state", None)
                if last_hidden is None:
                    last_hidden = outputs[0]
                pooled = last_hidden[:, 0, :]
                return self.classifier(pooled)

        return _SequenceClassifier()

    def train(
        self,
        *,
        network: Any,
        tokenizer: Any,
        texts: list[str],
        label_ids: list[int],
        max_seq_length: int,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        warmup_ratio: float | None,
        gradient_accumulation_steps: int | None,
        seed: int,
    ) -> None:
        torch = self.torch
        nn = self.nn
        set_all_seeds(seed)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        network.to(device)
        network.train()

        optimizer = self.AdamW(
            network.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        grad_accum = max(1, int(gradient_accumulation_steps or 1))
        batches_per_epoch = max(1, math.ceil(len(texts) / max(1, batch_size)))
        total_steps = max(1, math.ceil(batches_per_epoch / grad_accum) * max(1, epochs))
        scheduler = None
        warmup = float(warmup_ratio or 0.0)
        schedule_factory = getattr(self.transformers, "get_linear_schedule_with_warmup", None)
        if callable(schedule_factory):
            scheduler = schedule_factory(
                optimizer,
                num_warmup_steps=int(total_steps * warmup),
                num_training_steps=total_steps,
            )

        for epoch in range(max(1, epochs)):
            order = np.arange(len(texts))
            np.random.default_rng(seed + epoch).shuffle(order)
            optimizer.zero_grad()

            for step, start in enumerate(range(0, len(order), max(1, batch_size)), start=1):
                batch_idx = order[start:start + max(1, batch_size)]
                batch_texts = [texts[int(i)] for i in batch_idx]
                batch_labels = torch.tensor(
                    [label_ids[int(i)] for i in batch_idx],
                    dtype=torch.long,
                    device=device,
                )
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(device)
                    for key, value in encoded.items()
                }
                logits = network(**encoded)
                loss = nn.functional.cross_entropy(logits, batch_labels)
                loss = loss / grad_accum
                loss.backward()

                is_step_boundary = (step % grad_accum) == 0
                is_last_step = step == math.ceil(len(order) / max(1, batch_size))
                if is_step_boundary or is_last_step:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()

            if device == "cuda":
                torch.cuda.empty_cache()

        network.to("cpu")

    def predict_logits(
        self,
        *,
        network: Any,
        tokenizer: Any,
        texts: list[str],
        max_seq_length: int,
        batch_size: int,
    ) -> np.ndarray:
        torch = self.torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        network.to(device)
        network.eval()

        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), max(1, batch_size)):
                batch_texts = texts[start:start + max(1, batch_size)]
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(device)
                    for key, value in encoded.items()
                }
                logits = network(**encoded)
                chunks.append(logits.detach().cpu().numpy())

        network.to("cpu")
        if not chunks:
            return np.zeros((0, 0), dtype=float)
        return np.vstack(chunks)

    def embed_vectors(
        self,
        *,
        network: Any,
        tokenizer: Any,
        texts: list[str],
        max_seq_length: int,
        batch_size: int,
        pooling: str,
    ) -> np.ndarray:
        torch = self.torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        network.to(device)
        network.eval()

        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), max(1, batch_size)):
                batch_texts = texts[start:start + max(1, batch_size)]
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(device)
                    for key, value in encoded.items()
                }
                outputs = network.encoder(**encoded)
                last_hidden = getattr(outputs, "last_hidden_state", None)
                if last_hidden is None:
                    last_hidden = outputs[0]

                if pooling == "cls":
                    pooled = last_hidden[:, 0, :]
                else:
                    attention_mask = encoded.get("attention_mask")
                    if attention_mask is None:
                        pooled = last_hidden.mean(dim=1)
                    else:
                        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
                        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                chunks.append(pooled.detach().cpu().numpy())

        network.to("cpu")
        if not chunks:
            return np.zeros((0, 0), dtype=float)
        return np.vstack(chunks)

    def save_pretrained(self, *, network: Any, tokenizer: Any, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        encoder_dir = path / "encoder"
        tokenizer_dir = path / "tokenizer"
        network.encoder.save_pretrained(encoder_dir)
        tokenizer.save_pretrained(tokenizer_dir)
        self.torch.save(network.classifier.state_dict(), path / "head.pt")

    def load_weights(self, *, network: Any, path: Path) -> None:
        state_dict = self.torch.load(path / "head.pt", map_location="cpu")
        network.classifier.load_state_dict(state_dict)

    def copy_state(self, *, source: Any, target: Any) -> None:
        target.load_state_dict(source.state_dict())


class XlmRobertaClassifier(TextClassifier):
    """XLM-RoBERTa encoder with a configurable MLP classification head."""

    _backend_factory = staticmethod(_TorchRuntime.load)

    def __init__(
        self,
        pretrained_model: str = "xlm-roberta-base",
        max_seq_length: int = 256,
        batch_size: int = 16,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float | None = None,
        gradient_accumulation_steps: int | None = None,
        head_hidden_units: int | None = None,
        dropout: float | None = None,
        activation: str | None = None,
    ) -> None:
        self.pretrained_model = pretrained_model
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.head_hidden_units = head_hidden_units
        self.dropout = 0.1 if dropout is None else float(dropout)
        self.activation = activation or "gelu"

        self._backend: Any | None = None
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._classes: list[str] = []
        self._label_to_id: dict[str, int] = {}

    def _get_backend(self) -> Any:
        if self._backend is None:
            self._backend = self._backend_factory()
        return self._backend

    def fit(self, texts: list[str], labels: list[str], seed: int = 42) -> None:
        classes = sorted({str(label) for label in labels})
        if len(classes) < 2:
            raise ValueError("XlmRobertaClassifier requires at least 2 classes.")

        self._classes = classes
        self._label_to_id = {
            label: idx for idx, label in enumerate(self._classes)
        }
        label_ids = [self._label_to_id[str(label)] for label in labels]

        backend = self._get_backend()
        self._tokenizer = backend.create_tokenizer(self.pretrained_model)
        encoder = backend.create_encoder(self.pretrained_model)
        self._model = backend.build_network(
            encoder=encoder,
            num_labels=len(self._classes),
            head_hidden_units=self.head_hidden_units,
            dropout=self.dropout,
            activation=self.activation,
        )
        backend.train(
            network=self._model,
            tokenizer=self._tokenizer,
            texts=[str(text or "") for text in texts],
            label_ids=label_ids,
            max_seq_length=self.max_seq_length,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            seed=seed,
        )

    def predict(self, texts: list[str]) -> np.ndarray:
        probas = self.predict_proba(texts)
        if probas.size == 0:
            return np.array([], dtype=object)
        indices = probas.argmax(axis=1)
        return np.array([self._classes[int(idx)] for idx in indices], dtype=object)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        assert self._model is not None, "Call fit() first"
        assert self._tokenizer is not None, "Call fit() first"

        backend = self._get_backend()
        logits = backend.predict_logits(
            network=self._model,
            tokenizer=self._tokenizer,
            texts=[str(text or "") for text in texts],
            max_seq_length=self.max_seq_length,
            batch_size=self.batch_size,
        )
        return self._softmax(logits)

    def embed_texts(self, texts: list[str], pooling: str = "mean") -> np.ndarray:
        assert self._model is not None, "Call fit() first"
        assert self._tokenizer is not None, "Call fit() first"
        if pooling not in {"cls", "mean"}:
            raise ValueError(f"Unsupported pooling mode: {pooling}")

        backend = self._get_backend()
        return backend.embed_vectors(
            network=self._model,
            tokenizer=self._tokenizer,
            texts=[str(text or "") for text in texts],
            max_seq_length=self.max_seq_length,
            batch_size=self.batch_size,
            pooling=pooling,
        )

    @property
    def classes_(self) -> list[str]:
        assert self._classes, "Call fit() first"
        return list(self._classes)

    def save(self, path: Path) -> None:
        assert self._model is not None, "Call fit() first"
        assert self._tokenizer is not None, "Call fit() first"

        save_dir = self._normalize_save_dir(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        backend = self._get_backend()
        backend.save_pretrained(
            network=self._model,
            tokenizer=self._tokenizer,
            path=save_dir,
        )

        metadata = {
            "model_type": "xlm_roberta",
            "pretrained_model": self.pretrained_model,
            "max_seq_length": self.max_seq_length,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "head_hidden_units": self.head_hidden_units,
            "dropout": self.dropout,
            "activation": self.activation,
            "classes": self._classes,
            "label_to_id": self._label_to_id,
        }
        (save_dir / "metadata.json").write_bytes(
            orjson.dumps(metadata, option=orjson.OPT_INDENT_2)
        )

    @classmethod
    def load(cls, path: Path) -> "XlmRobertaClassifier":
        load_dir = cls._normalize_load_dir(path)
        metadata = orjson.loads((load_dir / "metadata.json").read_bytes())

        instance = cls(
            pretrained_model=str(metadata.get("pretrained_model", "xlm-roberta-base")),
            max_seq_length=int(metadata.get("max_seq_length", 256)),
            batch_size=int(metadata.get("batch_size", 16)),
            epochs=int(metadata.get("epochs", 3)),
            learning_rate=float(metadata.get("learning_rate", 2e-5)),
            weight_decay=float(metadata.get("weight_decay", 0.01)),
            warmup_ratio=metadata.get("warmup_ratio"),
            gradient_accumulation_steps=metadata.get("gradient_accumulation_steps"),
            head_hidden_units=metadata.get("head_hidden_units"),
            dropout=float(metadata.get("dropout", 0.1)),
            activation=str(metadata.get("activation", "gelu")),
        )
        instance._classes = [str(value) for value in metadata.get("classes", [])]
        instance._label_to_id = {
            str(key): int(value)
            for key, value in (metadata.get("label_to_id") or {}).items()
        }

        backend = instance._get_backend()
        instance._tokenizer = backend.create_tokenizer(str(load_dir / "tokenizer"))
        encoder = backend.create_encoder(str(load_dir / "encoder"))
        instance._model = backend.build_network(
            encoder=encoder,
            num_labels=len(instance._classes),
            head_hidden_units=instance.head_hidden_units,
            dropout=instance.dropout,
            activation=instance.activation,
        )
        backend.load_weights(network=instance._model, path=load_dir)
        return instance

    def get_hyperparameters(self) -> dict[str, Any]:
        return {
            "model_type": "xlm_roberta",
            "pretrained_model": self.pretrained_model,
            "max_seq_length": self.max_seq_length,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "head_hidden_units": self.head_hidden_units,
            "dropout": self.dropout,
            "activation": self.activation,
        }

    def warm_start_from(
        self,
        base_clf: "XlmRobertaClassifier",
        texts: list[str],
        labels: list[str],
        seed: int = 42,
    ) -> bool:
        if not isinstance(base_clf, XlmRobertaClassifier):
            return False
        if self._model is None or self._tokenizer is None:
            return False
        if base_clf._model is None:
            return False
        if self._classes != base_clf.classes_:
            return False

        backend = self._get_backend()
        backend.copy_state(source=base_clf._model, target=self._model)
        label_ids = [self._label_to_id[str(label)] for label in labels]
        backend.train(
            network=self._model,
            tokenizer=self._tokenizer,
            texts=[str(text or "") for text in texts],
            label_ids=label_ids,
            max_seq_length=self.max_seq_length,
            batch_size=self.batch_size,
            epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            seed=seed,
        )
        return True

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        if logits.size == 0:
            return np.zeros((0, 0), dtype=float)
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp = np.exp(shifted)
        denom = np.sum(exp, axis=1, keepdims=True)
        return exp / denom

    @staticmethod
    def _normalize_save_dir(path: Path) -> Path:
        if path.suffix:
            return path.parent / path.stem
        return path

    @staticmethod
    def _normalize_load_dir(path: Path) -> Path:
        if path.is_dir():
            return path
        if path.suffix:
            sibling_dir = path.parent / path.stem
            if sibling_dir.is_dir():
                return sibling_dir
        raise FileNotFoundError(f"Model directory not found: {path}")
