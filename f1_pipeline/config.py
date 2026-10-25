"""
Central project configuration for the F1 Fastest Lap pipeline.

This module is the authoritative source for:
- project paths
- primary key
- training-year boundaries
- operational prediction baseline
- canonical prediction column names

No business logic belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
from typing import Dict, Tuple


# ============================================================
# Project-wide immutable constants
# ============================================================

PRIMARY_KEY: Tuple[str, str] = (
    "Year",
    "gp_slug",
)

TRAINING_YEAR_MIN: int = 2017
TRAINING_YEAR_MAX: int = 2024

OPERATIONAL_BASELINE_COLUMN: str = (
    "Pred_GBR"
)

OPERATIONAL_PREDICTION_COLUMN: str = (
    "PredictedFastestLap_sec"
)

PREDICTION_COLUMNS: Tuple[str, ...] = (
    "Pred_GBR",
    "Pred_XGB",
    "Pred_CAT",
    "PredictedFastestLap_sec",
)

PREDICTION_OFFSET_COLUMNS: Tuple[str, ...] = (
    "Pred_GBR_offset_sec",
    "Pred_XGB_offset_sec",
    "Pred_CAT_offset_sec",
    "pred_offset_sec",
)

ACTUAL_COLUMN: str = (
    "ActualFastestLap_sec"
)


# ============================================================
# Central path configuration
# ============================================================

@dataclass(frozen=True)
class PipelineConfig:
    """
    Authoritative path configuration.

    The default root remains compatible with the current
    Google Colab / Google Drive project layout.

    Override through environment variable:

        F1_PROJECT_ROOT=/some/other/path
    """

    root: Path

    @classmethod
    def default(
        cls,
    ) -> "PipelineConfig":

        root = Path(
            os.environ.get(
                "F1_PROJECT_ROOT",
                "/content/drive/MyDrive/F1_ML_Project",
            )
        )

        return cls(
            root=root
        )

    # --------------------------------------------------------
    # Core directories
    # --------------------------------------------------------

    @property
    def data_dir(
        self,
    ) -> Path:
        return self.root / "data"

    @property
    def raw_dir(
        self,
    ) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(
        self,
    ) -> Path:
        return self.data_dir / "processed"

    @property
    def audit_dir(
        self,
    ) -> Path:
        return self.processed_dir / "audit"

    @property
    def backup_dir(
        self,
    ) -> Path:
        return self.processed_dir / "backups"

    @property
    def log_dir(
        self,
    ) -> Path:
        return self.root / "log"

    # --------------------------------------------------------
    # Canonical datasets
    # --------------------------------------------------------

    @property
    def training_path(
        self,
    ) -> Path:
        return (
            self.processed_dir
            / "df_training.csv"
        )

    @property
    def test_path(
        self,
    ) -> Path:
        return (
            self.processed_dir
            / "df_test.csv"
        )

    @property
    def overview_path(
        self,
    ) -> Path:
        return (
            self.processed_dir
            / "df_overview.csv"
        )

    @property
    def event_circuit_references_path(self) -> Path:
        return self.processed_dir / "event_circuit_references.csv"

    @property
    def circuit_info_path(
        self,
    ) -> Path:
        return (
            self.raw_dir
            / "circuit_info.csv"
        )

    # --------------------------------------------------------
    # Registry
    #
    # Authoritative registry:
    # future_event_registry.csv
    #
    # The eligible file is compatibility/output mirror only.
    # It will be implemented in registry.py in phase 3B.
    # --------------------------------------------------------

    @property
    def registry_path(
        self,
    ) -> Path:
        return (
            self.processed_dir
            / "future_event_registry.csv"
        )

    @property
    def registry_eligible_mirror_path(
        self,
    ) -> Path:
        return (
            self.processed_dir
            / "future_event_registry_eligible.csv"
        )

    # --------------------------------------------------------
    # Prediction audit artifacts
    # --------------------------------------------------------

    @property
    def prediction_ledger_path(
        self,
    ) -> Path:
        return (
            self.audit_dir
            / "promotion_grade_shadow_prediction_ledger.csv"
        )

    @property
    def timing_evaluation_path(self) -> Path:
        return self.audit_dir / "actual_race_start_timing_audit.csv"

    @property
    def model_log_path(
        self,
    ) -> Path:
        return (
            self.log_dir
            / "model_log.csv"
        )

    # --------------------------------------------------------
    # Source / UI
    # --------------------------------------------------------

    @property
    def streamlit_app_path(
        self,
    ) -> Path:
        return (
            self.root
            / "app_streamlit.py"
        )

    @property
    def helpers_path(
        self,
    ) -> Path:
        return (
            self.root
            / "utils"
            / "helpers.py"
        )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    def as_dict(
        self,
    ) -> Dict[str, str]:

        return {
            "root":
                str(self.root),

            "training_path":
                str(self.training_path),

            "test_path":
                str(self.test_path),

            "overview_path":
                str(self.overview_path),

            "circuit_info_path":
                str(self.circuit_info_path),

            "registry_path":
                str(self.registry_path),

            "registry_eligible_mirror_path":
                str(
                    self.registry_eligible_mirror_path
                ),

            "prediction_ledger_path":
                str(
                    self.prediction_ledger_path
                ),

            "model_log_path":
                str(self.model_log_path),

            "backup_dir":
                str(self.backup_dir),
        }


# Canonical default configuration instance.
CONFIG = PipelineConfig.default()
