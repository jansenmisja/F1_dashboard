
"""
Formal dataset contracts for the F1 Fastest Lap pipeline.

This module is deliberately independent from Streamlit.

Responsibilities:
- primary-key validation
- event uniqueness
- training-year invariant
- df_training contract
- df_test leakage contract
- df_overview contract

No persistence belongs here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import (
    PRIMARY_KEY,
    TRAINING_YEAR_MIN,
    TRAINING_YEAR_MAX,
    PREDICTION_COLUMNS,
    PREDICTION_OFFSET_COLUMNS,
    ACTUAL_COLUMN,
)


# ============================================================
# Canonical column groups
# ============================================================

TRAINING_TARGET_CANDIDATES = (
    "ActualFastestLap_sec",
    "FastestLap_sec",
    "TargetFastestLap_sec",
)

TEST_FORBIDDEN_PREDICTION_COLUMNS = (
    *PREDICTION_COLUMNS,
    *PREDICTION_OFFSET_COLUMNS,
)

TEST_FORBIDDEN_ACTUAL_COLUMNS = (
    ACTUAL_COLUMN,
)

TEST_FORBIDDEN_KPI_EXACT = (
    "Error",
    "AbsError",
    "APE_pct",
    "APE_pct_GBR",
    "APE_pct_XGB",
    "APE_pct_CAT",
    "Error_GBR",
    "Error_XGB",
    "Error_CAT",
    "AbsError_GBR",
    "AbsError_XGB",
    "AbsError_CAT",
)

# Prefix/suffix patterns catch future KPI variants too.
TEST_FORBIDDEN_KPI_PATTERNS = (
    "error",
    "abserror",
    "ape",
    "kpi",
)

OVERVIEW_REQUIRED_PREDICTION_COLUMNS = (
    "Pred_GBR",
    "Pred_XGB",
    "Pred_CAT",
    "PredictedFastestLap_sec",
)

LIFECYCLE_COLUMN = "RaceStatus"


# ============================================================
# Result object
# ============================================================

@dataclass
class ContractResult:
    name: str
    ok: bool
    errors: List[str] = field(
        default_factory=list
    )
    warnings: List[str] = field(
        default_factory=list
    )
    details: Dict[str, object] = field(
        default_factory=dict
    )

    def raise_if_invalid(
        self,
    ) -> None:

        if self.ok:
            return

        raise ValueError(
            f"{self.name} contract failed: "
            + " | ".join(
                self.errors
            )
        )


# ============================================================
# Generic helpers
# ============================================================

def _is_effectively_empty(
    series: pd.Series,
) -> bool:
    """
    True when a column contains no meaningful values.

    NaN, None and blank strings count as empty.
    """

    if series.empty:
        return True

    non_null = series.dropna()

    if non_null.empty:
        return True

    if (
        non_null
        .astype(str)
        .str.strip()
        .eq("")
        .all()
    ):
        return True

    return False


def get_target_mask(
    df: pd.DataFrame,
    year: int,
    gp_slug: str,
) -> pd.Series:

    missing = [
        col
        for col in PRIMARY_KEY
        if col not in df.columns
    ]

    if missing:
        raise KeyError(
            "Primary-key columns missing: "
            + ", ".join(
                missing
            )
        )

    return (
        pd.to_numeric(
            df["Year"],
            errors="coerce",
        ).eq(
            int(year)
        )
        &
        df["gp_slug"]
        .astype(str)
        .str.strip()
        .eq(
            str(
                gp_slug
            ).strip()
        )
    )




def validate_primary_key_uniqueness(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
) -> ContractResult:
    """
    Validate global uniqueness of the canonical primary key.

    This intentionally performs only:
    - required primary-key column validation
    - global duplicate detection on PRIMARY_KEY

    It does not apply dataset-specific schema,
    lifecycle, leakage, prediction, or KPI rules.
    """

    errors = []

    missing_key = [
        col
        for col in PRIMARY_KEY
        if col not in df.columns
    ]

    if missing_key:

        errors.append(
            f"{dataset_name}: missing primary-key columns: "
            + ", ".join(
                missing_key
            )
        )

        return ContractResult(
            name="primary_key_uniqueness",
            ok=False,
            errors=errors,
            details={
                "duplicate_key_rows": None,
            },
        )

    duplicate_count = int(
        df
        .duplicated(
            list(
                PRIMARY_KEY
            ),
            keep=False,
        )
        .sum()
    )

    if duplicate_count:

        errors.append(
            f"{dataset_name}: contains "
            f"{duplicate_count} rows involved "
            "in duplicate primary keys."
        )

    return ContractResult(
        name="primary_key_uniqueness",
        ok=not errors,
        errors=errors,
        details={
            "duplicate_key_rows":
                duplicate_count,
        },
    )



def validate_row_created_contract(
    df_test: pd.DataFrame,
    df_overview: pd.DataFrame,
    year: int,
    gp_slug: str,
) -> ContractResult:
    """
    Canonical row-created dataset contract.

    Checks exactly:
    - target event occurs once in df_test
    - target event occurs once in df_overview
    - global PRIMARY_KEY uniqueness in df_test
    - global PRIMARY_KEY uniqueness in df_overview

    Does not apply lifecycle, leakage, prediction,
    actual, KPI, or broader dataset-schema rules.
    """

    errors = []
    details = {}

    for dataset_name, df in [
        ("df_test", df_test),
        ("df_overview", df_overview),
    ]:

        event_result = (
            validate_event_uniqueness(
                df=df,
                year=year,
                gp_slug=gp_slug,
                dataset_name=dataset_name,
            )
        )

        details[
            f"{dataset_name}_target_count"
        ] = event_result.details.get(
            "count"
        )

        if not event_result.ok:

            errors.extend(
                event_result.errors
            )

        pk_result = (
            validate_primary_key_uniqueness(
                df=df,
                dataset_name=dataset_name,
            )
        )

        details[
            f"{dataset_name}_duplicate_key_rows"
        ] = pk_result.details.get(
            "duplicate_key_rows"
        )

        if not pk_result.ok:

            errors.extend(
                pk_result.errors
            )

    errors = list(
        dict.fromkeys(
            errors
        )
    )

    return ContractResult(
        name="row_created",
        ok=not errors,
        errors=errors,
        details=details,
    )



def validate_event_uniqueness(
    df: pd.DataFrame,
    year: int,
    gp_slug: str,
    dataset_name: str = "dataset",
) -> ContractResult:

    errors = []

    try:
        mask = get_target_mask(
            df,
            year,
            gp_slug,
        )

        count = int(
            mask.sum()
        )

    except Exception as exc:

        return ContractResult(
            name="event_uniqueness",
            ok=False,
            errors=[
                f"{dataset_name}: {exc}"
            ],
            details={
                "count": None,
            },
        )

    if count != 1:

        errors.append(
            f"{dataset_name}: target count={count}; "
            "expected exactly 1."
        )

    return ContractResult(
        name="event_uniqueness",
        ok=not errors,
        errors=errors,
        details={
            "count": count,
        },
    )


def require_one_event(
    df: pd.DataFrame,
    year: int,
    gp_slug: str,
    dataset_name: str = "dataset",
) -> pd.Series:

    result = validate_event_uniqueness(
        df=df,
        year=year,
        gp_slug=gp_slug,
        dataset_name=dataset_name,
    )

    result.raise_if_invalid()

    return get_target_mask(
        df,
        year,
        gp_slug,
    )


# ============================================================
# Training contract
# ============================================================

def assert_training_year_guard(
    df_training: pd.DataFrame,
) -> None:
    """
    Hard invariant:
    df_training may contain only 2017..2024.

    Any operational 2025+ race is forbidden.
    """

    if "Year" not in df_training.columns:

        raise ValueError(
            "df_training has no Year column."
        )

    years = pd.to_numeric(
        df_training["Year"],
        errors="coerce",
    )

    if years.isna().any():

        bad_count = int(
            years.isna().sum()
        )

        raise ValueError(
            "df_training contains "
            f"{bad_count} invalid Year value(s)."
        )

    bad_mask = (
        (years < TRAINING_YEAR_MIN)
        |
        (years > TRAINING_YEAR_MAX)
    )

    if bad_mask.any():

        bad_years = sorted(
            {
                int(x)
                for x in years[
                    bad_mask
                ].tolist()
            }
        )

        raise ValueError(
            "df_training year guard violated. "
            f"Allowed={TRAINING_YEAR_MIN}.."
            f"{TRAINING_YEAR_MAX}; "
            f"found={bad_years}"
        )


def validate_training_contract(
    df_training: pd.DataFrame,
    require_target: bool = True,
) -> ContractResult:

    errors = []
    warnings = []

    missing_key = [
        col
        for col in PRIMARY_KEY
        if col not in df_training.columns
    ]

    if missing_key:

        errors.append(
            "Missing primary-key columns: "
            + ", ".join(
                missing_key
            )
        )

    try:

        assert_training_year_guard(
            df_training
        )

    except Exception as exc:

        errors.append(
            str(exc)
        )

    target_cols = [
        col
        for col in TRAINING_TARGET_CANDIDATES
        if col in df_training.columns
    ]

    if (
        require_target
        and not target_cols
    ):

        errors.append(
            "No allowed training target found. "
            "Expected one of: "
            + ", ".join(
                TRAINING_TARGET_CANDIDATES
            )
        )

    if len(
        target_cols
    ) > 1:

        warnings.append(
            "Multiple target candidates present: "
            + ", ".join(
                target_cols
            )
        )

    duplicate_count = None

    if not missing_key:

        duplicate_mask = (
            df_training
            .duplicated(
                list(
                    PRIMARY_KEY
                ),
                keep=False,
            )
        )

        duplicate_count = int(
            duplicate_mask.sum()
        )

        if duplicate_count:

            errors.append(
                "df_training contains "
                f"{duplicate_count} rows involved "
                "in duplicate primary keys."
            )

    return ContractResult(
        name="df_training",
        ok=not errors,
        errors=errors,
        warnings=warnings,
        details={
            "rows":
                len(
                    df_training
                ),

            "target_candidates":
                target_cols,

            "duplicate_key_rows":
                duplicate_count,

            "year_min":
                (
                    int(
                        pd.to_numeric(
                            df_training["Year"],
                            errors="coerce",
                        ).min()
                    )
                    if (
                        "Year"
                        in df_training.columns
                        and not df_training.empty
                    )
                    else None
                ),

            "year_max":
                (
                    int(
                        pd.to_numeric(
                            df_training["Year"],
                            errors="coerce",
                        ).max()
                    )
                    if (
                        "Year"
                        in df_training.columns
                        and not df_training.empty
                    )
                    else None
                ),
        },
    )


# ============================================================
# df_test contract
# ============================================================

def _find_test_forbidden_kpi_columns(
    columns: Sequence[str],
) -> List[str]:

    found = []

    exact_lower = {
        col.lower()
        for col in TEST_FORBIDDEN_KPI_EXACT
    }

    for col in columns:

        lower = str(
            col
        ).lower()

        if lower in exact_lower:
            found.append(
                col
            )
            continue

        if any(
            pattern
            in lower
            for pattern in TEST_FORBIDDEN_KPI_PATTERNS
        ):
            found.append(
                col
            )

    return sorted(
        set(
            found
        )
    )


def validate_test_contract(
    df_test: pd.DataFrame,
    *,
    strict_schema: bool = True,
) -> ContractResult:
    """
    Canonical operational df_test contract.

    strict_schema=True:
        prediction/actual/KPI columns may not exist at all.

    strict_schema=False:
        legacy columns may exist, but must be entirely empty.

    The strict mode is the target architecture.
    """

    errors = []
    warnings = []

    missing_key = [
        col
        for col in PRIMARY_KEY
        if col not in df_test.columns
    ]

    if missing_key:

        errors.append(
            "Missing primary-key columns: "
            + ", ".join(
                missing_key
            )
        )

    if (
        LIFECYCLE_COLUMN
        not in df_test.columns
    ):

        errors.append(
            "df_test missing RaceStatus."
        )

    forbidden = set(
        TEST_FORBIDDEN_PREDICTION_COLUMNS
        + TEST_FORBIDDEN_ACTUAL_COLUMNS
    )

    forbidden.update(
        _find_test_forbidden_kpi_columns(
            df_test.columns
        )
    )

    present_forbidden = sorted(
        col
        for col in forbidden
        if col in df_test.columns
    )

    populated_forbidden = []

    for col in present_forbidden:

        if not _is_effectively_empty(
            df_test[
                col
            ]
        ):

            populated_forbidden.append(
                col
            )

    if strict_schema:

        if present_forbidden:

            errors.append(
                "df_test contains forbidden "
                "operational/output columns: "
                + ", ".join(
                    present_forbidden
                )
            )

    else:

        if populated_forbidden:

            errors.append(
                "df_test contains populated "
                "prediction/actual/KPI columns: "
                + ", ".join(
                    populated_forbidden
                )
            )

        legacy_empty = sorted(
            set(
                present_forbidden
            )
            - set(
                populated_forbidden
            )
        )

        if legacy_empty:

            warnings.append(
                "Legacy forbidden columns exist "
                "but are empty: "
                + ", ".join(
                    legacy_empty
                )
            )

    duplicate_count = None

    if not missing_key:

        duplicate_count = int(
            df_test
            .duplicated(
                list(
                    PRIMARY_KEY
                ),
                keep=False,
            )
            .sum()
        )

        if duplicate_count:

            errors.append(
                "df_test contains "
                f"{duplicate_count} rows involved "
                "in duplicate primary keys."
            )

    return ContractResult(
        name="df_test",
        ok=not errors,
        errors=errors,
        warnings=warnings,
        details={
            "rows":
                len(
                    df_test
                ),

            "strict_schema":
                strict_schema,

            "present_forbidden_columns":
                present_forbidden,

            "populated_forbidden_columns":
                populated_forbidden,

            "duplicate_key_rows":
                duplicate_count,
        },
    )


# ============================================================
# df_overview contract
# ============================================================

def validate_overview_contract(
    df_overview: pd.DataFrame,
) -> ContractResult:

    errors = []
    warnings = []

    missing_key = [
        col
        for col in PRIMARY_KEY
        if col not in df_overview.columns
    ]

    if missing_key:

        errors.append(
            "Missing primary-key columns: "
            + ", ".join(
                missing_key
            )
        )

    if (
        LIFECYCLE_COLUMN
        not in df_overview.columns
    ):

        errors.append(
            "df_overview missing RaceStatus."
        )

    missing_prediction_schema = [
        col
        for col in OVERVIEW_REQUIRED_PREDICTION_COLUMNS
        if col not in df_overview.columns
    ]

    # Prediction columns belong in overview, but historical
    # rows do not all need populated predictions.
    if missing_prediction_schema:

        warnings.append(
            "df_overview does not yet contain all "
            "canonical prediction columns: "
            + ", ".join(
                missing_prediction_schema
            )
        )

    if ACTUAL_COLUMN not in df_overview.columns:

        warnings.append(
            "df_overview does not yet contain "
            f"{ACTUAL_COLUMN}."
        )

    duplicate_count = None

    if not missing_key:

        duplicate_count = int(
            df_overview
            .duplicated(
                list(
                    PRIMARY_KEY
                ),
                keep=False,
            )
            .sum()
        )

        if duplicate_count:

            errors.append(
                "df_overview contains "
                f"{duplicate_count} rows involved "
                "in duplicate primary keys."
            )

    return ContractResult(
        name="df_overview",
        ok=not errors,
        errors=errors,
        warnings=warnings,
        details={
            "rows":
                len(
                    df_overview
                ),

            "duplicate_key_rows":
                duplicate_count,

            "missing_prediction_schema":
                missing_prediction_schema,
        },
    )


# ============================================================
# Cross-dataset contract
# ============================================================

def validate_operational_dataset_contracts(
    df_training: pd.DataFrame,
    df_test: pd.DataFrame,
    df_overview: pd.DataFrame,
    *,
    strict_test_schema: bool = True,
) -> Dict[str, ContractResult]:

    return {
        "training":
            validate_training_contract(
                df_training
            ),

        "test":
            validate_test_contract(
                df_test,
                strict_schema=(
                    strict_test_schema
                ),
            ),

        "overview":
            validate_overview_contract(
                df_overview
            ),
    }
