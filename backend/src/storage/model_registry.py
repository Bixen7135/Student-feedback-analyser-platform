"""Model registry — register, list, compare trained models."""
from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import orjson

from src.storage.database import Database
from src.storage.models import ModelMeta
from src.utils.logging import get_logger

log = get_logger(__name__)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class ModelRegistry:
    """Manages trained model storage and metadata."""

    def __init__(self, db: Database, models_dir: Path) -> None:
        self.db = db
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Register
    # ------------------------------------------------------------------

    def register_model(
        self,
        name: str,
        task: str,
        model_type: str,
        source_model_path: Path,
        source_metrics_path: Path | None = None,
        dataset_id: str | None = None,
        dataset_version: int | None = None,
        config: dict | None = None,
        metrics: dict | None = None,
        run_id: str | None = None,
        base_model_id: str | None = None,
        job_id: str | None = None,
        input_signature: dict | None = None,
        preprocess_spec: dict | None = None,
        training_profile: dict | None = None,
    ) -> ModelMeta:
        """Register a trained model in the registry.

        Copies model artifacts to the registry directory and creates a DB record.
        """
        model_id = str(uuid.uuid4())
        now = _utcnow()

        # Determine version (auto-increment for same task+model_type+dataset)
        version = 1
        if dataset_id:
            existing = self.db.fetchone(
                """SELECT MAX(version) as max_v FROM models
                WHERE task = ? AND model_type = ? AND dataset_id = ? AND status = 'active'""",
                (task, model_type, dataset_id),
            )
            if existing and existing["max_v"] is not None:
                version = existing["max_v"] + 1

        # Store artifacts
        version_dir = self.models_dir / model_id / f"v{version}"
        version_dir.mkdir(parents=True, exist_ok=True)

        model_dest = version_dir / "model.joblib"
        shutil.copy2(source_model_path, model_dest)

        if source_metrics_path and source_metrics_path.exists():
            shutil.copy2(source_metrics_path, version_dir / "metrics.json")

        # Save config
        config = config or {}
        (version_dir / "config.json").write_bytes(
            orjson.dumps(config, option=orjson.OPT_INDENT_2)
        )

        # Load metrics from file if not provided
        if metrics is None and source_metrics_path and source_metrics_path.exists():
            metrics = orjson.loads(source_metrics_path.read_bytes())
        metrics = metrics or {}
        input_signature = input_signature or {}
        preprocess_spec = preprocess_spec or {}
        training_profile = training_profile or {}

        (version_dir / "signature.json").write_bytes(
            orjson.dumps(
                {
                    "input_signature": input_signature,
                    "preprocess_spec": preprocess_spec,
                    "training_profile": training_profile,
                },
                option=orjson.OPT_INDENT_2,
            )
        )

        storage_path = str(version_dir)

        self.db.execute(
            """INSERT INTO models
            (id, name, task, model_type, version, dataset_id, dataset_version,
             config, metrics, created_at, status, storage_path, run_id, base_model_id, job_id,
             input_signature, preprocess_spec, training_profile)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (model_id, name, task, model_type, version, dataset_id, dataset_version,
             orjson.dumps(config).decode(), orjson.dumps(metrics).decode(),
             now, "active", storage_path, run_id, base_model_id, job_id,
             orjson.dumps(input_signature).decode(),
             orjson.dumps(preprocess_spec).decode(),
             orjson.dumps(training_profile).decode()),
        )
        self.db.commit()

        meta = ModelMeta(
            id=model_id,
            name=name,
            task=task,
            model_type=model_type,
            version=version,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            config=config,
            metrics=metrics,
            created_at=now,
            storage_path=storage_path,
            run_id=run_id,
            base_model_id=base_model_id,
            job_id=job_id,
            input_signature=input_signature,
            preprocess_spec=preprocess_spec,
            training_profile=training_profile,
        )
        log.info("model_registered", id=model_id, task=task, model_type=model_type, version=version)
        return meta

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_models(
        self,
        task: str | None = None,
        model_type: str | None = None,
        dataset_id: str | None = None,
        run_id: str | None = None,
        include_archived: bool = False,
        sort: str = "created_at",
        order: str = "desc",
        page: int = 1,
        per_page: int = 20,
    ) -> tuple[list[ModelMeta], int]:
        """List models with filters and pagination."""
        conditions: list[str] = []
        params: list[Any] = []

        if not include_archived:
            conditions.append("status = 'active'")

        if task:
            conditions.append("task = ?")
            params.append(task)
        if model_type:
            conditions.append("model_type = ?")
            params.append(model_type)
        if dataset_id:
            conditions.append("dataset_id = ?")
            params.append(dataset_id)
        if run_id:
            conditions.append("run_id = ?")
            params.append(run_id)

        where = " AND ".join(conditions) if conditions else "1 = 1"
        valid_sorts = {"created_at", "name", "task", "model_type", "version"}
        if sort not in valid_sorts:
            sort = "created_at"
        order_dir = "DESC" if order.lower() == "desc" else "ASC"

        count_row = self.db.fetchone(f"SELECT COUNT(*) as cnt FROM models WHERE {where}", tuple(params))
        total = count_row["cnt"] if count_row else 0

        offset = (page - 1) * per_page
        rows = self.db.fetchall(
            f"SELECT * FROM models WHERE {where} ORDER BY {sort} {order_dir} LIMIT ? OFFSET ?",
            (*params, per_page, offset),
        )

        models = [self._row_to_model(r) for r in rows]
        return models, total

    def get_model(self, model_id: str) -> ModelMeta | None:
        """Get a single model by ID."""
        row = self.db.fetchone("SELECT * FROM models WHERE id = ?", (model_id,))
        if row is None:
            return None
        return self._row_to_model(row)

    def get_model_versions(self, task: str, model_type: str, dataset_id: str | None = None) -> list[ModelMeta]:
        """Get all versions of a model for a given task+type+dataset."""
        if dataset_id:
            rows = self.db.fetchall(
                """SELECT * FROM models WHERE task = ? AND model_type = ? AND dataset_id = ?
                AND status = 'active' ORDER BY version DESC""",
                (task, model_type, dataset_id),
            )
        else:
            rows = self.db.fetchall(
                """SELECT * FROM models WHERE task = ? AND model_type = ?
                AND status = 'active' ORDER BY version DESC""",
                (task, model_type),
            )
        return [self._row_to_model(r) for r in rows]

    def compare_models(self, model_ids: list[str]) -> list[ModelMeta]:
        """Get multiple models for comparison."""
        if not model_ids:
            return []
        placeholders = ",".join(["?"] * len(model_ids))
        rows = self.db.fetchall(
            f"SELECT * FROM models WHERE id IN ({placeholders})",
            tuple(model_ids),
        )
        return [self._row_to_model(r) for r in rows]

    def load_model_artifact(self, model_id: str) -> Path:
        """Get the path to a model's joblib file."""
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Model not found: {model_id}")
        model_path = Path(model.storage_path) / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")
        return model_path

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def update_metadata(
        self,
        model_id: str,
        name: str | None = None,
    ) -> ModelMeta | None:
        """Update model metadata (name only for now)."""
        updates: list[str] = []
        params: list[Any] = []

        if name is not None:
            updates.append("name = ?")
            params.append(name)

        if not updates:
            return self.get_model(model_id)

        params.append(model_id)
        self.db.execute(
            f"UPDATE models SET {', '.join(updates)} WHERE id = ?",
            tuple(params),
        )
        self.db.commit()

        updated = self.get_model(model_id)
        if updated:
            log.info("model_updated", id=model_id, name=name)
        return updated

    def delete_model(self, model_id: str) -> dict:
        """Soft-delete a model by setting status to 'archived'."""
        model = self.get_model(model_id)
        if model is None:
            raise ValueError(f"Model not found: {model_id}")

        # Check for dependent analyses using the normalized refs table
        deps = self.db.fetchall(
            """SELECT ar.id, ar.name FROM analysis_runs ar
            JOIN analysis_model_refs amr ON amr.analysis_id = ar.id
            WHERE amr.model_id = ? AND ar.status != 'failed'""",
            (model_id,),
        )

        if deps:
            return {
                "deleted": False,
                "reason": "Model is referenced by one or more analysis runs",
                "dependencies": {"analyses": len(deps)},
            }

        self.db.execute(
            "UPDATE models SET status = 'archived' WHERE id = ?",
            (model_id,),
        )
        self.db.commit()
        log.info("model_deleted", id=model_id)
        return {"deleted": True, "model_id": model_id}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def get_lineage(self, model_id: str) -> list[ModelMeta]:
        """Return the ancestry chain for a model, from current back to the root.

        E.g. [current_model, parent_model, grandparent_model, ..., root_model].
        Stops when base_model_id is None or the referenced model is not found.
        """
        chain: list[ModelMeta] = []
        visited: set[str] = set()
        current_id: str | None = model_id
        while current_id and current_id not in visited:
            visited.add(current_id)
            model = self.get_model(current_id)
            if model is None:
                break
            chain.append(model)
            current_id = model.base_model_id
        return chain

    @staticmethod
    def _row_to_model(row: Any) -> ModelMeta:
        d = dict(row)
        for field in (
            "config",
            "metrics",
            "input_signature",
            "preprocess_spec",
            "training_profile",
        ):
            raw = d.get(field)
            if isinstance(raw, str):
                try:
                    d[field] = orjson.loads(raw)
                except Exception:
                    d[field] = {}
            elif raw is None:
                d[field] = {}
        return ModelMeta(**d)
