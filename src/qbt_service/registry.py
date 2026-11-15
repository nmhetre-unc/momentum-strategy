"""
The model registry: register a trained model as a new immutable version,
read versions back, and load a version's model with its integrity
checked.

register() runs in the caller's transaction under an advisory lock, so
concurrent registrations get consecutive versions without gaps. A model
whose artifact is byte-identical to one already registered on the same
price snapshot is not registered twice: register() returns the existing
version (the same rule as ingesting an unchanged snapshot).

load_model() checks the stored artifact against its SHA-256, and the
stored serving features against theirs, before anything is scored, and
raises IntegrityError on a mismatch. Rows are immutable, so a verified
model can be cached by version for the life of a process.
"""

import hashlib
import threading
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import Connection, RowMapping, func, select, text

from qbt.ml.models import ARTIFACT_FORMAT, FittedModel, from_artifact
from qbt.ml.train import TrainedModel, config_dict, features_hash
from qbt_service.db import model_registry

# An arbitrary constant identifying "model registration" among advisory locks.
REGISTER_LOCK_KEY = 0x0DE1_4E61

# Everything but the artifact bytes, which only load_model() needs.
SUMMARY_COLUMNS = [c for c in model_registry.c if c.name != "artifact"]


class IntegrityError(RuntimeError):
    """A stored artifact or serving input doesn't match its SHA-256."""


@dataclass(frozen=True)
class Registered:
    version: int
    created: bool          # False: an identical model was already registered


def register(conn: Connection, price_series_id: int, trained: TrainedModel,
             request: dict[str, Any] | None = None) -> Registered:
    """Stores `trained` as the next version, or finds the identical one."""
    conn.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": REGISTER_LOCK_KEY})
    existing = conn.execute(
        select(model_registry.c.version).where(
            model_registry.c.price_series_id == price_series_id,
            model_registry.c.artifact_sha256 == trained.artifact_sha256)
    ).scalar_one_or_none()
    if existing is not None:
        return Registered(existing, created=False)

    version = conn.execute(
        select(func.coalesce(func.max(model_registry.c.version), 0) + 1)
    ).scalar_one()
    bar = trained.metrics["bar"]
    conn.execute(model_registry.insert().values(
        version=version, model_type=trained.config.model, target=trained.config.target,
        feature_set=trained.config.feature_set, features=trained.features,
        classes=trained.classes, hyperparameters=trained.hyperparameters,
        training_config=config_dict(trained.config) | {"request": request or {}},
        seed=trained.config.seed, price_series_id=price_series_id,
        train_start=trained.train_first_decision, train_end=trained.train_last_decision,
        train_label_end=trained.train_last_label_end, n_train=trained.n_train,
        oos_accuracy=trained.metrics["out_of_sample"]["accuracy"],
        oos_months=trained.metrics["out_of_sample"]["n_months"],
        baseline_name=bar["name"], baseline_accuracy=bar["accuracy"],
        beats_baseline=bar["beats"], metrics=trained.metrics,
        serving_as_of=trained.serving_as_of, serving_features=trained.serving_features,
        serving_features_sha256=trained.serving_features_sha256,
        artifact_format=ARTIFACT_FORMAT, artifact=trained.artifact,
        artifact_sha256=trained.artifact_sha256, library_versions=trained.library_versions,
    ))
    return Registered(version, created=True)


def list_models(conn: Connection) -> list[RowMapping]:
    return list(conn.execute(
        select(*SUMMARY_COLUMNS, func.octet_length(model_registry.c.artifact)
               .label("artifact_bytes"))
        .order_by(model_registry.c.version)
    ).mappings())


def get_model(conn: Connection, version: int) -> RowMapping | None:
    return conn.execute(
        select(*SUMMARY_COLUMNS, func.octet_length(model_registry.c.artifact)
               .label("artifact_bytes"))
        .where(model_registry.c.version == version)
    ).mappings().one_or_none()


def latest_version(conn: Connection) -> int | None:
    return conn.execute(select(func.max(model_registry.c.version))).scalar_one()


def serving_input(row: RowMapping) -> tuple[date, pd.DataFrame]:
    """
    The stored serving features as a one-row frame in the model's column
    order, after checking them against their SHA-256.
    """
    features = row["serving_features"]
    frame_row = pd.Series([float(features[name]) for name in row["features"]],
                          index=list(row["features"]))
    if features_hash(row["serving_as_of"], frame_row) != bytes(row["serving_features_sha256"]):
        raise IntegrityError(f"model version {row['version']}: the stored serving features "
                             "don't match their SHA-256")
    return row["serving_as_of"], frame_row.to_frame().T


class ModelCache:
    """Verified models by version; rows never change, so entries never go stale."""

    def __init__(self) -> None:
        self._models: dict[int, FittedModel] = {}
        self._lock = threading.Lock()

    def load(self, conn: Connection, version: int) -> FittedModel:
        with self._lock:
            model = self._models.get(version)
        if model is None:
            model = load_model(conn, version)
            with self._lock:
                self._models.setdefault(version, model)
        return model

    def __len__(self) -> int:
        return len(self._models)


def load_model(conn: Connection, version: int) -> FittedModel:
    """Version `version`'s model, after verifying its artifact's SHA-256."""
    row = conn.execute(
        select(model_registry.c.artifact, model_registry.c.artifact_sha256,
               model_registry.c.features)
        .where(model_registry.c.version == version)
    ).one_or_none()
    if row is None:
        raise LookupError(f"no model version {version}")
    artifact = bytes(row.artifact)
    if hashlib.sha256(artifact).digest() != bytes(row.artifact_sha256):
        raise IntegrityError(f"model version {version}: the stored artifact doesn't match "
                             "its SHA-256; refusing to load it")
    model = from_artifact(artifact)
    if list(model.columns) != list(row.features):
        raise IntegrityError(f"model version {version}: the artifact's columns differ from "
                             "the registry's feature list")
    return model
