"""Model registry and prediction log.

model_registry holds one immutable row per trained model version: what
it predicts, from which features and price snapshot, with which
hyperparameters, its out-of-sample accuracy next to the baseline it was
compared with, and the serialized model with its SHA-256. predictions
logs every POST /predict.

Downgrading drops both tables, and with them every registered model and
logged prediction.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-15
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

revision: str = "0004"
down_revision: str | Sequence[str] | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "model_registry",
        sa.Column("version", sa.Integer, nullable=False, autoincrement=False),
        sa.Column("model_type", sa.Text, nullable=False),
        sa.Column("target", sa.Text, nullable=False),
        sa.Column("feature_set", sa.Text, nullable=False),
        sa.Column("features", ARRAY(sa.Text), nullable=False),
        sa.Column("classes", ARRAY(sa.Double), nullable=False),
        sa.Column("hyperparameters", JSONB, nullable=False),
        sa.Column("training_config", JSONB, nullable=False),
        sa.Column("seed", sa.BigInteger, nullable=False),
        sa.Column("price_series_id", sa.BigInteger, nullable=False),
        sa.Column("train_start", sa.Date, nullable=False),
        sa.Column("train_end", sa.Date, nullable=False),
        sa.Column("train_label_end", sa.Date, nullable=False),
        sa.Column("n_train", sa.Integer, nullable=False),
        sa.Column("oos_accuracy", sa.Double, nullable=False),
        sa.Column("oos_months", sa.Integer, nullable=False),
        sa.Column("baseline_name", sa.Text, nullable=False),
        sa.Column("baseline_accuracy", sa.Double, nullable=False),
        sa.Column("beats_baseline", sa.Boolean, nullable=False),
        sa.Column("metrics", JSONB, nullable=False),
        sa.Column("serving_as_of", sa.Date, nullable=False),
        sa.Column("serving_features", JSONB, nullable=False),
        sa.Column("serving_features_sha256", sa.LargeBinary, nullable=False),
        sa.Column("artifact_format", sa.Text, nullable=False),
        sa.Column("artifact", sa.LargeBinary, nullable=False),
        sa.Column("artifact_sha256", sa.LargeBinary, nullable=False),
        sa.Column("library_versions", JSONB, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("version", name=op.f("pk_model_registry")),
        sa.ForeignKeyConstraint(["price_series_id"], ["price_series.id"], ondelete="RESTRICT",
                                name=op.f("fk_model_registry_price_series_id_price_series")),
        sa.UniqueConstraint("price_series_id", "artifact_sha256",
                            name=op.f("uq_model_registry_price_series_id_artifact_sha256")),
        sa.CheckConstraint("version >= 1", name=op.f("ck_model_registry_version_positive")),
        sa.CheckConstraint("model_type IN ('lightgbm')",
                           name=op.f("ck_model_registry_model_type_known")),
        sa.CheckConstraint("target IN ('direction', 'regime')",
                           name=op.f("ck_model_registry_target_known")),
        sa.CheckConstraint("cardinality(features) >= 1",
                           name=op.f("ck_model_registry_has_features")),
        sa.CheckConstraint("cardinality(classes) >= 2",
                           name=op.f("ck_model_registry_has_classes")),
        sa.CheckConstraint("train_start <= train_end AND train_end < train_label_end",
                           name=op.f("ck_model_registry_train_window_order")),
        sa.CheckConstraint("n_train > 0 AND oos_months > 0",
                           name=op.f("ck_model_registry_counts_positive")),
        sa.CheckConstraint(
            "oos_accuracy BETWEEN 0 AND 1 AND baseline_accuracy BETWEEN 0 AND 1",
            name=op.f("ck_model_registry_accuracies_in_range")),
        sa.CheckConstraint("serving_as_of >= train_label_end",
                           name=op.f("ck_model_registry_serving_after_training")),
        sa.CheckConstraint("octet_length(artifact_sha256) = 32",
                           name=op.f("ck_model_registry_artifact_sha256_length")),
        sa.CheckConstraint("octet_length(serving_features_sha256) = 32",
                           name=op.f("ck_model_registry_serving_features_sha256_length")),
    )
    op.create_table(
        "predictions",
        sa.Column("id", sa.BigInteger, sa.Identity(always=True), nullable=False),
        sa.Column("model_version", sa.Integer, nullable=False),
        sa.Column("as_of", sa.Date, nullable=False),
        sa.Column("features_sha256", sa.LargeBinary, nullable=False),
        sa.Column("prediction", sa.Double, nullable=False),
        sa.Column("probabilities", ARRAY(sa.Double), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("now()")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_predictions")),
        sa.ForeignKeyConstraint(["model_version"], ["model_registry.version"],
                                ondelete="RESTRICT",
                                name=op.f("fk_predictions_model_version_model_registry")),
        sa.CheckConstraint("octet_length(features_sha256) = 32",
                           name=op.f("ck_predictions_features_sha256_length")),
    )
    op.create_index("predictions_by_model", "predictions", ["model_version", "created_at"])


def downgrade() -> None:
    op.drop_index("predictions_by_model", table_name=op.f("predictions"))
    op.drop_table("predictions")
    op.drop_table("model_registry")
