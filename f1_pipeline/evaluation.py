from __future__ import annotations
from f1_pipeline.schemas import (
    require_one_event,
    validate_test_contract,
    validate_overview_contract,
)

from dataclasses import dataclass
from typing import Dict
import math

from pathlib import Path
import pandas as pd

from f1_pipeline.config import CONFIG
from f1_pipeline.io import (
    read_csv_checked,
    safe_commit_csv,
    safe_commit_many,
)
from f1_pipeline.lifecycle import (
    get_event_state,
    transition_event_status,
)
import numpy as np


@dataclass(frozen=True)
class EvaluationMetrics:
    ActualFastestLap_sec: float

    Error: float
    AbsError: float
    APE_pct: float

    Error_GBR: float
    AbsError_GBR: float
    APE_pct_GBR: float

    Error_XGB: float
    AbsError_XGB: float
    APE_pct_XGB: float

    Error_CAT: float
    AbsError_CAT: float
    APE_pct_CAT: float

    target_offset_sec: float
    Error_offset: float
    AbsError_offset: float


def _finite_positive(
    value: float,
    name: str,
) -> float:

    try:
        value = float(value)
    except Exception as exc:
        raise ValueError(
            f"{name} must be numeric"
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            f"{name} must be finite"
        )

    if value <= 0:
        raise ValueError(
            f"{name} must be > 0"
        )

    return value


def _prediction_metrics(
    prediction: float,
    actual: float,
) -> Dict[str, float]:

    error = (
        prediction
        - actual
    )

    abs_error = abs(
        error
    )

    ape = (
        abs_error
        / actual
        * 100.0
    )

    return {
        "error":
            error,

        "abs_error":
            abs_error,

        "ape":
            ape,
    }


def calculate_evaluation_metrics(
    *,
    pred_gbr: float,
    pred_xgb: float,
    pred_cat: float,
    predicted_fastest_lap_sec: float,
    actual_fastest_lap_sec: float,
    record_lap_sec: float,
) -> EvaluationMetrics:
    """
    Pure post-race KPI calculation.

    Canonical operational prediction:
        PredictedFastestLap_sec == Pred_GBR

    No I/O.
    No lifecycle mutation.
    No dataframe mutation.
    """

    pred_gbr = _finite_positive(
        pred_gbr,
        "Pred_GBR",
    )

    pred_xgb = _finite_positive(
        pred_xgb,
        "Pred_XGB",
    )

    pred_cat = _finite_positive(
        pred_cat,
        "Pred_CAT",
    )

    predicted_fastest_lap_sec = (
        _finite_positive(
            predicted_fastest_lap_sec,
            "PredictedFastestLap_sec",
        )
    )

    actual = _finite_positive(
        actual_fastest_lap_sec,
        "ActualFastestLap_sec",
    )

    record = _finite_positive(
        record_lap_sec,
        "RecordLap_sec",
    )

    # Operational invariant
    if not math.isclose(
        predicted_fastest_lap_sec,
        pred_gbr,
        abs_tol=1e-9,
        rel_tol=0.0,
    ):

        raise ValueError(
            "PredictedFastestLap_sec must equal Pred_GBR"
        )

    gbr = _prediction_metrics(
        pred_gbr,
        actual,
    )

    xgb = _prediction_metrics(
        pred_xgb,
        actual,
    )

    cat = _prediction_metrics(
        pred_cat,
        actual,
    )

    operational = _prediction_metrics(
        predicted_fastest_lap_sec,
        actual,
    )

    pred_offset = (
        predicted_fastest_lap_sec
        - record
    )

    target_offset = (
        actual
        - record
    )

    error_offset = (
        pred_offset
        - target_offset
    )

    return EvaluationMetrics(
        ActualFastestLap_sec=
            actual,

        Error=
            operational[
                "error"
            ],

        AbsError=
            operational[
                "abs_error"
            ],

        APE_pct=
            operational[
                "ape"
            ],

        Error_GBR=
            gbr[
                "error"
            ],

        AbsError_GBR=
            gbr[
                "abs_error"
            ],

        APE_pct_GBR=
            gbr[
                "ape"
            ],

        Error_XGB=
            xgb[
                "error"
            ],

        AbsError_XGB=
            xgb[
                "abs_error"
            ],

        APE_pct_XGB=
            xgb[
                "ape"
            ],

        Error_CAT=
            cat[
                "error"
            ],

        AbsError_CAT=
            cat[
                "abs_error"
            ],

        APE_pct_CAT=
            cat[
                "ape"
            ],

        target_offset_sec=
            target_offset,

        Error_offset=
            error_offset,

        AbsError_offset=
            abs(
                error_offset
            ),
    )


def evaluation_metrics_dict(
    **kwargs,
) -> Dict[str, float]:

    metrics = (
        calculate_evaluation_metrics(
            **kwargs
        )
    )

    return {
        field:
            getattr(
                metrics,
                field,
            )
        for field in (
            metrics.__dataclass_fields__
        )
    }

EVALUATION_WRITE_COLUMNS = (
    "ActualFastestLap_sec",

    "Error",
    "AbsError",
    "APE_pct",

    "Error_GBR",
    "AbsError_GBR",
    "APE_pct_GBR",

    "Error_XGB",
    "AbsError_XGB",
    "APE_pct_XGB",

    "Error_CAT",
    "AbsError_CAT",
    "APE_pct_CAT",

    "target_offset_sec",
    "Error_offset",
    "AbsError_offset",
)


def _event_mask(
    df: pd.DataFrame,
    year: int,
    gp_slug: str,
) -> pd.Series:

    if (
        "Year" not in df.columns
        or "gp_slug" not in df.columns
    ):
        raise ValueError(
            "Year/gp_slug missing from dataframe"
        )

    return (
        pd.to_numeric(
            df["Year"],
            errors="coerce",
        ).eq(int(year))
        &
        df["gp_slug"]
        .astype(str)
        .str.strip()
        .eq(str(gp_slug).strip())
    )


def _require_one_event(
    df: pd.DataFrame,
    year: int,
    gp_slug: str,
    name: str,
) -> pd.Series:

    mask = _event_mask(
        df,
        year,
        gp_slug,
    )

    count = int(
        mask.sum()
    )

    if count != 1:
        raise ValueError(
            f"{name}: expected exactly one "
            f"row for {year}/{gp_slug}; "
            f"found {count}"
        )

    return mask


def _value_is_empty(
    value,
) -> bool:

    if pd.isna(value):
        return True

    if isinstance(
        value,
        str,
    ):
        return (
            value.strip()
            .lower()
            in {
                "",
                "nan",
                "none",
                "null",
            }
        )

    return False


def _historical_evaluate_race(
    year: int,
    gp_slug: str,
    actual_fastest_lap_sec: float,
) -> dict:
    """
    Canonical post-race evaluation.

    Contract
    --------
    - requires RaceStatus == predicted;
    - reads predictions from df_overview;
    - writes actual/KPI values only to df_overview;
    - df_test remains leakage-free;
    - lifecycle transition is exclusively
      predicted -> evaluated;
    - rolls overview back if lifecycle transition fails.
    """

    year = int(year)
    gp_slug = str(
        gp_slug
    ).strip()

    # --------------------------------------------------------
    # 1. Canonical pre-state
    # --------------------------------------------------------
    state_before = get_event_state(
        year,
        gp_slug,
    )

    if not state_before.lifecycle_valid:

        raise ValueError(
            "Invalid lifecycle state: "
            + " | ".join(
                state_before.blockers
            )
        )

    if (
        state_before.race_status
        != "predicted"
    ):

        raise ValueError(
            "evaluate_race requires "
            "RaceStatus=predicted; "
            f"current={state_before.race_status}"
        )

    # --------------------------------------------------------
    # 2. Fresh canonical reads
    # --------------------------------------------------------
    df_test = read_csv_checked(
        CONFIG.test_path
    )

    df_overview = read_csv_checked(
        CONFIG.overview_path
    )

    mt = _require_one_event(
        df_test,
        year,
        gp_slug,
        "df_test",
    )

    mo = _require_one_event(
        df_overview,
        year,
        gp_slug,
        "df_overview",
    )

    test_row = (
        df_test
        .loc[mt]
        .iloc[0]
    )

    overview_row = (
        df_overview
        .loc[mo]
        .iloc[0]
    )

    # --------------------------------------------------------
    # 3. Leakage guard
    # --------------------------------------------------------
    for col in EVALUATION_WRITE_COLUMNS:

        if (
            col in df_test.columns
            and not _value_is_empty(
                test_row.get(
                    col
                )
            )
        ):

            raise ValueError(
                "df_test leakage detected "
                f"in {col}"
            )

    # --------------------------------------------------------
    # 4. Prediction inputs
    # --------------------------------------------------------
    required_prediction_cols = (
        "Pred_GBR",
        "Pred_XGB",
        "Pred_CAT",
        "PredictedFastestLap_sec",
        "RecordLap_sec",
    )

    missing = [
        col
        for col in required_prediction_cols
        if col not in df_overview.columns
    ]

    if missing:

        raise ValueError(
            "Missing overview prediction columns: "
            + ", ".join(
                missing
            )
        )

    inputs = {}

    for col in required_prediction_cols:

        value = pd.to_numeric(
            pd.Series(
                [
                    overview_row.get(
                        col
                    )
                ]
            ),
            errors="coerce",
        ).iloc[0]

        if pd.isna(value):

            raise ValueError(
                f"{col} is missing/non-numeric"
            )

        inputs[
            col
        ] = float(
            value
        )

    # --------------------------------------------------------
    # 5. Pure KPI calculation
    # --------------------------------------------------------
    metrics = (
        calculate_evaluation_metrics(
            pred_gbr=
                inputs[
                    "Pred_GBR"
                ],

            pred_xgb=
                inputs[
                    "Pred_XGB"
                ],

            pred_cat=
                inputs[
                    "Pred_CAT"
                ],

            predicted_fastest_lap_sec=
                inputs[
                    "PredictedFastestLap_sec"
                ],

            actual_fastest_lap_sec=
                actual_fastest_lap_sec,

            record_lap_sec=
                inputs[
                    "RecordLap_sec"
                ],
        )
    )

    metric_values = {
        field:
            float(
                getattr(
                    metrics,
                    field,
                )
            )
        for field in EVALUATION_WRITE_COLUMNS
    }

    # --------------------------------------------------------
    # 6. Preserve pre-write overview for rollback
    # --------------------------------------------------------
    overview_before = (
        df_overview.copy(
            deep=True
        )
    )

    # --------------------------------------------------------
    # 7. Ensure/write ONLY overview
    # --------------------------------------------------------
    overview_new = (
        df_overview.copy(
            deep=True
        )
    )

    for col in EVALUATION_WRITE_COLUMNS:

        if col not in overview_new.columns:
            overview_new[
                col
            ] = pd.NA

        overview_new.loc[
            mo,
            col,
        ] = metric_values[
            col
        ]

    # Status blijft hier predicted.
    # Alleen lifecycle mag status wijzigen.
    overview_new.loc[
        mo,
        "RaceStatus",
    ] = "predicted"

    # --------------------------------------------------------
    # 8. Transactional overview commit
    # --------------------------------------------------------
    safe_commit_csv(
        overview_new,
        CONFIG.overview_path,
        backup_dir=(
            CONFIG.backups_dir
            if hasattr(
                CONFIG,
                "backups_dir",
            )
            else None
        ),
        round_decimals=3,
        index=False,
    )

    try:

        # ----------------------------------------------------
        # 9. Readback vóór status promotion
        # ----------------------------------------------------
        overview_check = (
            read_csv_checked(
                CONFIG.overview_path
            )
        )

        mo_check = _require_one_event(
            overview_check,
            year,
            gp_slug,
            "df_overview readback",
        )

        row_check = (
            overview_check
            .loc[mo_check]
            .iloc[0]
        )

        for col, expected in (
            metric_values.items()
        ):

            actual = pd.to_numeric(
                pd.Series(
                    [
                        row_check.get(
                            col
                        )
                    ]
                ),
                errors="coerce",
            ).iloc[0]

            if pd.isna(actual):

                raise RuntimeError(
                    f"Readback missing {col}"
                )

            if not math.isclose(
                float(actual),
                round(
                    float(expected),
                    3,
                ),
                abs_tol=0.001,
                rel_tol=0.0,
            ):

                raise RuntimeError(
                    f"Readback mismatch {col}: "
                    f"expected={expected}, "
                    f"actual={actual}"
                )

        # ----------------------------------------------------
        # 10. Canonical lifecycle transition
        # ----------------------------------------------------
        transition = (
            transition_event_status(
                year=year,
                gp_slug=gp_slug,
                expected_from="predicted",
                to_status="evaluated",
            )
        )

        if not transition.ok:

            raise RuntimeError(
                "Lifecycle transition failed: "
                + str(
                    transition.message
                )
            )

    except Exception:

        # ----------------------------------------------------
        # 11. Rollback overview on any failure after commit
        # ----------------------------------------------------
        safe_commit_csv(
            overview_before,
            CONFIG.overview_path,
            backup_dir=(
                CONFIG.backups_dir
                if hasattr(
                    CONFIG,
                    "backups_dir",
                )
                else None
            ),
            round_decimals=3,
            index=False,
        )

        raise

    # --------------------------------------------------------
    # 12. Final canonical readback
    # --------------------------------------------------------
    final_state = get_event_state(
        year,
        gp_slug,
    )

    if (
        final_state.race_status
        != "evaluated"
    ):

        raise RuntimeError(
            "Final RaceStatus is not evaluated"
        )

    final_test = read_csv_checked(
        CONFIG.test_path
    )

    final_overview = read_csv_checked(
        CONFIG.overview_path
    )

    mt_final = _require_one_event(
        final_test,
        year,
        gp_slug,
        "df_test final",
    )

    mo_final = _require_one_event(
        final_overview,
        year,
        gp_slug,
        "df_overview final",
    )

    final_test_row = (
        final_test
        .loc[mt_final]
        .iloc[0]
    )

    # Hard leakage guard after lifecycle transition.
    for col in EVALUATION_WRITE_COLUMNS:

        if (
            col in final_test.columns
            and not _value_is_empty(
                final_test_row.get(
                    col
                )
            )
        ):

            raise RuntimeError(
                "Post-evaluation df_test leakage "
                f"detected in {col}"
            )

    final_overview_row = (
        final_overview
        .loc[mo_final]
        .iloc[0]
    )

    return {
        "ok":
            True,

        "Year":
            year,

        "gp_slug":
            gp_slug,

        "status_before":
            state_before.race_status,

        "status_after":
            final_state.race_status,

        **{
            col:
                float(
                    final_overview_row[
                        col
                    ]
                )
            for col
            in EVALUATION_WRITE_COLUMNS
        },
    }


def recalc_recordlap_dependents(
    df: pd.DataFrame,
    mask: pd.Series,
) -> pd.DataFrame:
    """
    Pure recalculation of columns derived from RecordLap_sec.

    No I/O, no lifecycle mutation, no Streamlit usage.
    Operates only on columns already present in the dataframe,
    except derived columns explicitly produced here.
    """

    out = df.copy()

    def num(col):

        if col not in out.columns:
            return None

        return pd.to_numeric(
            out.loc[
                mask,
                col,
            ],
            errors="coerce",
        )

    record = num(
        "RecordLap_sec"
    )

    if record is None:
        return out

    if (
        "PoleTime_sec"
        in out.columns
    ):

        out.loc[
            mask,
            "PoleMinusRecord_sec",
        ] = (
            num(
                "PoleTime_sec"
            )
            - record
        )

    if (
        "PredictedFastestLap_sec"
        in out.columns
    ):

        out.loc[
            mask,
            "pred_offset_sec",
        ] = (
            num(
                "PredictedFastestLap_sec"
            )
            - record
        )

    if (
        "ActualFastestLap_sec"
        in out.columns
    ):

        out.loc[
            mask,
            "target_offset_sec",
        ] = (
            num(
                "ActualFastestLap_sec"
            )
            - record
        )

    for model in (
        "GBR",
        "XGB",
        "CAT",
    ):

        pred_col = (
            f"Pred_{model}"
        )

        offset_col = (
            f"Pred_{model}_offset_sec"
        )

        if pred_col in out.columns:

            out.loc[
                mask,
                offset_col,
            ] = (
                num(
                    pred_col
                )
                - record
            )

    if {
        "pred_offset_sec",
        "target_offset_sec",
    }.issubset(
        out.columns
    ):

        out.loc[
            mask,
            "Error_offset",
        ] = (
            num(
                "pred_offset_sec"
            )
            - num(
                "target_offset_sec"
            )
        )

        out.loc[
            mask,
            "AbsError_offset",
        ] = (
            num(
                "Error_offset"
            )
            .abs()
        )

        target_offset = (
            num(
                "target_offset_sec"
            )
            .abs()
        )

        out.loc[
            mask,
            "APE_pct_offset",
        ] = np.where(
            target_offset > 0,
            (
                num(
                    "AbsError_offset"
                )
                / target_offset
                * 100
            ),
            np.nan,
        )

    return out

# ============================================================
# Canonical operational correction/audit writers
# Phase 7H.68a
# ============================================================

from .storage_transactions import transactional as _service_storage_transaction

@_service_storage_transaction
def correct_recordlap(
    year: int,
    gp_slug: str,
    new_recordlap_sec: float,
):

    year = int(year)
    gp_slug = str(gp_slug).strip()
    new_recordlap_sec = float(
        new_recordlap_sec
    )

    if new_recordlap_sec <= 0:
        raise ValueError(
            "RecordLap_sec moet > 0 zijn."
        )

    df_test = read_csv_checked(
        CONFIG.test_path
    )

    df_overview = read_csv_checked(
        CONFIG.overview_path
    )

    circuit_info = read_csv_checked(
        CONFIG.circuit_info_path
    )

    mt = require_one_event(
        df_test,
        year,
        gp_slug,
        dataset_name="df_test",
    )

    mo = require_one_event(
        df_overview,
        year,
        gp_slug,
        dataset_name="df_overview",
    )

    # EVENT_RECORDLAP_REFERENCE_V1
    from .event_circuits import resolve_event_circuit
    reference_row, circuit_reference_slug, event_layout_slug = (
        resolve_event_circuit(year, gp_slug, circuit_info)
    )

    for dataset_name, frame, target in [
        ("df_test", df_test, mt),
        ("df_overview", df_overview, mo),
    ]:
        row = frame.loc[target].iloc[0]
        if str(row["Circuit"]) != str(reference_row["Circuit"]):
            raise ValueError(f"{dataset_name}: circuit reference mismatch.")
        if event_layout_slug is not None:
            if str(row.get("CircuitLayoutSlug")) != event_layout_slug:
                raise ValueError(f"{dataset_name}: event layout mismatch.")

    mci = (
        circuit_info["gp_slug"]
        .astype(str)
        .eq(circuit_reference_slug)
    )

    if int(mci.sum()) != 1:
        raise ValueError(
            "circuit_info target moet exact één keer voorkomen."
        )

    new_test = df_test.copy()
    new_overview = df_overview.copy()
    new_ci = circuit_info.copy()

    if "RecordLap_sec" not in new_test.columns:
        new_test["RecordLap_sec"] = pd.NA

    if "RecordLap_sec" not in new_overview.columns:
        new_overview["RecordLap_sec"] = pd.NA

    new_test.loc[
        mt,
        "RecordLap_sec",
    ] = new_recordlap_sec

    new_overview.loc[
        mo,
        "RecordLap_sec",
    ] = new_recordlap_sec

    new_ci.loc[
        mci,
        "RecordLap_sec",
    ] = new_recordlap_sec

    new_test = recalc_recordlap_dependents(
        new_test,
        mt,
    )

    new_overview = recalc_recordlap_dependents(
        new_overview,
        mo,
    )

    def validate_circuit_info(df):
        mask = (
            df["gp_slug"]
            .astype(str)
            .eq(circuit_reference_slug)
        )

        if int(mask.sum()) != 1:
            return False

        value = pd.to_numeric(
            df.loc[
                mask,
                "RecordLap_sec",
            ],
            errors="coerce",
        ).iloc[0]

        return (
            pd.notna(value)
            and abs(
                float(value)
                - new_recordlap_sec
            )
            <= 0.001
        )

    safe_commit_many(
        {
            CONFIG.circuit_info_path:
                new_ci,
            CONFIG.test_path:
                new_test,
            CONFIG.overview_path:
                new_overview,
        },
        validators={
            CONFIG.circuit_info_path:
                validate_circuit_info,

            CONFIG.test_path:
                lambda df:
                    validate_test_contract(
                        df,
                        strict_schema=True,
                    ),

            CONFIG.overview_path:
                validate_overview_contract,
        },
        backup_dir=CONFIG.backup_dir,
    )

    final_test = read_csv_checked(
        CONFIG.test_path
    )

    final_overview = read_csv_checked(
        CONFIG.overview_path
    )

    final_ci = read_csv_checked(
        CONFIG.circuit_info_path
    )

    require_one_event(
        final_test,
        year,
        gp_slug,
        dataset_name="df_test",
    )

    require_one_event(
        final_overview,
        year,
        gp_slug,
        dataset_name="df_overview",
    )

    if not validate_circuit_info(
        final_ci
    ):
        raise RuntimeError(
            "RecordLap circuit_info readback validatie faalt."
        )

    return {
        "ok": True,
        "year": year,
        "gp_slug": gp_slug,
        "RecordLap_sec":
            new_recordlap_sec,
    }


@_service_storage_transaction
def save_suggested_model_audit(
    year: int,
    gp_slug: str,
    suggestion: Dict,
):

    year = int(year)
    gp_slug = str(gp_slug).strip()

    required = [
        "SuggestedModel",
        "SuggestedModelReason",
        "SuggestedSegment",
        "SuggestedSegment_n",
        "Suggested_APE",
        "Suggested_vs_GBR_APE",
    ]

    missing = [
        key
        for key in required
        if key not in suggestion
    ]

    if missing:
        raise ValueError(
            "SuggestedModel audit mist velden: "
            + ", ".join(missing)
        )

    df_overview = read_csv_checked(
        CONFIG.overview_path
    )

    new_overview = df_overview.copy()

    for col in required:
        if col not in new_overview.columns:
            new_overview[col] = pd.NA

    mo = require_one_event(
        new_overview,
        year,
        gp_slug,
        dataset_name="df_overview",
    )

    protected_cols = [
        col
        for col in [
            "Pred_GBR",
            "Pred_XGB",
            "Pred_CAT",
            "PredictedFastestLap_sec",
        ]
        if col in new_overview.columns
    ]

    before_predictions = {
        col:
            new_overview.loc[
                mo,
                col,
            ].iloc[0]
        for col in protected_cols
    }

    audit_values = {
        "SuggestedModel":
            suggestion["SuggestedModel"],

        "SuggestedModelReason":
            suggestion["SuggestedModelReason"],

        "SuggestedSegment":
            suggestion["SuggestedSegment"],

        "SuggestedSegment_n":
            int(
                suggestion[
                    "SuggestedSegment_n"
                ]
            ),

        "Suggested_APE":
            suggestion["Suggested_APE"],

        "Suggested_vs_GBR_APE":
            suggestion[
                "Suggested_vs_GBR_APE"
            ],
    }

    for col, value in audit_values.items():
        new_overview.loc[
            mo,
            col,
        ] = value

    for col in protected_cols:

        before_value = (
            before_predictions[col]
        )

        after_value = (
            new_overview.loc[
                mo,
                col,
            ].iloc[0]
        )

        if (
            pd.isna(before_value)
            and pd.isna(after_value)
        ):
            continue

        try:
            same = np.isclose(
                float(before_value),
                float(after_value),
                atol=0.001,
            )
        except (
            TypeError,
            ValueError,
        ):
            same = (
                str(before_value)
                == str(after_value)
            )

        if not same:
            raise RuntimeError(
                f"{col} zou door SuggestedModel "
                "audit worden gewijzigd."
            )

    safe_commit_csv(
        new_overview,
        CONFIG.overview_path,
        validator=validate_overview_contract,
        backup_dir=CONFIG.backup_dir,
    )

    reread = read_csv_checked(
        CONFIG.overview_path
    )

    mo_post = require_one_event(
        reread,
        year,
        gp_slug,
        dataset_name="df_overview",
    )

    post_model = str(
        reread.loc[
            mo_post,
            "SuggestedModel",
        ].iloc[0]
    )

    if (
        post_model
        != str(
            suggestion[
                "SuggestedModel"
            ]
        )
    ):
        raise RuntimeError(
            "SuggestedModel readback mismatch."
        )

    for col in protected_cols:

        before_value = (
            before_predictions[col]
        )

        post_value = (
            reread.loc[
                mo_post,
                col,
            ].iloc[0]
        )

        if (
            pd.isna(before_value)
            and pd.isna(post_value)
        ):
            continue

        try:
            same = np.isclose(
                float(before_value),
                float(post_value),
                atol=0.001,
            )
        except (
            TypeError,
            ValueError,
        ):
            same = (
                str(before_value)
                == str(post_value)
            )

        if not same:
            raise RuntimeError(
                "Post-write protected prediction "
                f"field changed: {col}"
            )

    return {
        "ok": True,
        "year": year,
        "gp_slug": gp_slug,
        "SuggestedModel":
            suggestion[
                "SuggestedModel"
            ],
    }

# ACTUAL_RACE_START_V1: separate append-only timing evidence API.
@_service_storage_transaction
def record_actual_race_start(year, gp_slug, run_id, actual_race_start_utc=None,
                            actual_race_start_source=None, actual_race_start_verified_at_utc=None,
                            *, correction_reason=None):
    """Record timing evidence only; do not modify prediction, actual or lifecycle.

    A timing PASS is not by itself promotion-grade: other gates still apply.
    Historic policies are not reclassified by this API.
    """
    import json, hashlib
    from .prediction import _timing_record_hash
    from .readiness import _timing_utc
    event = f"{int(year)}_{str(gp_slug).strip()}"
    ledger = read_csv_checked(CONFIG.prediction_ledger_path)
    mask = ledger["race_event_id"].astype(str).eq(event) & ledger["RunID"].astype(str).eq(str(run_id))
    if int(mask.sum()) != 1:
        raise ValueError("Exactly one event/RunID prediction required")
    prediction = ledger.loc[mask].iloc[0]
    if str(prediction.get("PredictionTimingPolicyVersion", "")) != "actual_race_start_v1":
        raise ValueError("Historical policy preserved; this API cannot reclassify it")
    if str(prediction.get("PredictionHashMethod")) != "csv_payload_manifest_v1" or str(prediction["PredictionRowHash"]) != _timing_record_hash(prediction):
        raise ValueError("Original prediction hash invalid")
    prediction_time = _timing_utc(prediction["PredictionTimestampUTC"], "PredictionTimestampUTC")
    before = {path: path.read_bytes() for path in (CONFIG.prediction_ledger_path, CONFIG.model_log_path,
              CONFIG.test_path, CONFIG.overview_path, CONFIG.training_path) if path.exists()}
    problems = []
    start = verified = None
    try:
        start = _timing_utc(actual_race_start_utc, "ActualRaceStartUTC")
        verified = _timing_utc(actual_race_start_verified_at_utc, "ActualRaceStartVerifiedAtUTC")
        if not actual_race_start_source or not str(actual_race_start_source).strip():
            raise ValueError("ActualRaceStartSource missing")
        if verified < start:
            raise ValueError("Actual start confirmation predates start")
        if verified > pd.Timestamp.now(tz="UTC"):
            raise ValueError("Actual start verification is in the future")
    except (ValueError, TypeError) as exc:
        problems.append(str(exc))
    if problems:
        classification = "timing_unproven"
    elif prediction_time < start:
        classification = "actual_start_timing_pass"
    else:
        classification = "non_promotion_grade_timing"
    path = CONFIG.timing_evaluation_path
    old = read_csv_checked(path) if path.exists() else pd.DataFrame()
    record = {
        "RunID": str(run_id), "race_event_id": event,
        "PredictionRowHash": str(prediction["PredictionRowHash"]),
        "PredictionTimingPolicyVersion": str(prediction["PredictionTimingPolicyVersion"]),
        "PredictionTimestampUTC": str(prediction["PredictionTimestampUTC"]),
        "ActualRaceStartUTC": str(actual_race_start_utc or ""),
        "ActualRaceStartSource": str(actual_race_start_source or ""),
        "ActualRaceStartVerifiedAtUTC": str(actual_race_start_verified_at_utc or ""),
        "TimingEvidenceClass": classification,
        "Reason": "; ".join(problems),
    }
    previous = old.loc[old["RunID"].astype(str).eq(str(run_id))] if "RunID" in old else pd.DataFrame()
    if len(previous):
        last = previous.iloc[-1]
        def text(x): return "" if x is None or pd.isna(x) else str(x)
        if all(text(last.get(key)) == text(value) for key, value in record.items()):
            return {"ok": True, "idempotent": True, "TimingEvidenceClass": classification}
        if not correction_reason or not str(correction_reason).strip():
            raise ValueError("Changed timing evidence requires an append-only correction reason")
        record["OriginalTimingEvidenceClass"] = str(last["TimingEvidenceClass"])
    else:
        record["OriginalTimingEvidenceClass"] = "provisional_announced_start_pass"
    record["CorrectionReason"] = str(correction_reason or "")
    record["EvidenceRecordedAtUTC"] = pd.Timestamp.now(tz="UTC").isoformat()
    record["EvidenceRecordHash"] = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    from .prediction import _append_frame_preserving_history, _check_unchanged_history
    new = _append_frame_preserving_history(old, record)
    def validate(frame):
        _check_unchanged_history(frame, old)
        if str(frame.iloc[-1]["PredictionRowHash"]) != str(prediction["PredictionRowHash"]):
            raise ValueError("Original prediction linkage changed")
        for key, value in record.items():
            observed = frame.iloc[-1].get(key)
            observed = "" if observed is None or pd.isna(observed) else str(observed)
            if observed != str(value):
                raise ValueError(f"Timing evidence readback mismatch: {key}")
    safe_commit_csv(new, path, validator=validate, backup_dir=CONFIG.backup_dir)
    if any(path.read_bytes() != content for path, content in before.items()):
        raise RuntimeError("Protected prediction/data file changed during timing evidence recording")
    return {"ok": True, "idempotent": False, "TimingEvidenceClass": classification,
            "PromotionGradeConfirmed": False, "detail": "Other promotion gates must still be checked"}

# Sandbox proposal: KPI evaluation and timing evidence remain separate.
@_service_storage_transaction
def get_actual_start_timing_summary(year, gp_slug, run_id):
    import hashlib, json
    from .prediction import _timing_record_hash
    from .readiness import _timing_utc
    event=f'{int(year)}_{str(gp_slug).strip()}'
    ledger=read_csv_checked(CONFIG.prediction_ledger_path)
    mask=ledger['race_event_id'].astype(str).eq(event)&ledger['RunID'].astype(str).eq(str(run_id))
    if int(mask.sum())!=1:raise ValueError('Exactly one event/RunID prediction required')
    prediction=ledger.loc[mask].iloc[0]
    if str(prediction.get('PredictionTimingPolicyVersion',''))!='actual_race_start_v1':
        raise ValueError('Historical timing policy is preserved')
    if str(prediction['PredictionRowHash'])!=_timing_record_hash(prediction):
        raise ValueError('Original prediction hash invalid')
    result={'TimingEvidenceClass':'timing_unproven','PredictionRunID':str(run_id),
            'OriginalPredictionRowHash':str(prediction['PredictionRowHash']),
            'PromotionGradeConfirmed':False,'TimingEvidenceRecords':0,
            'TimingDetail':'Actual race start evidence not recorded'}
    if not CONFIG.timing_evaluation_path.exists():return result
    audit=read_csv_checked(CONFIG.timing_evaluation_path)
    required={'RunID','race_event_id','PredictionRowHash','PredictionTimingPolicyVersion','PredictionTimestampUTC',
              'ActualRaceStartUTC','ActualRaceStartSource','ActualRaceStartVerifiedAtUTC',
              'TimingEvidenceClass','Reason','OriginalTimingEvidenceClass','CorrectionReason',
              'EvidenceRecordedAtUTC','EvidenceRecordHash'}
    if not required.issubset(audit.columns):raise ValueError('Timing evidence schema incomplete')
    records=audit.loc[audit['RunID'].astype(str).eq(str(run_id))]
    if records.empty:return result
    def text(value):return '' if value is None or pd.isna(value) else str(value)
    previous_class='provisional_announced_start_pass'
    for _,row in records.iterrows():
        payload={field:text(row[field]) for field in required if field!='EvidenceRecordHash'}
        expected=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if text(row['EvidenceRecordHash'])!=expected:raise ValueError('Timing evidence hash invalid')
        for field,value in [('RunID',str(run_id)),('race_event_id',event),('PredictionRowHash',str(prediction['PredictionRowHash'])),
                            ('PredictionTimingPolicyVersion','actual_race_start_v1'),('PredictionTimestampUTC',str(prediction['PredictionTimestampUTC']))]:
            if payload[field]!=value:raise ValueError('Timing evidence prediction linkage mismatch: '+field)
        if payload['OriginalTimingEvidenceClass']!=previous_class:
            raise ValueError('Timing evidence correction chain invalid')
        if previous_class!='provisional_announced_start_pass' and not payload['CorrectionReason'].strip():
            raise ValueError('Timing evidence correction reason missing')
        recorded=_timing_utc(payload['EvidenceRecordedAtUTC'],'EvidenceRecordedAtUTC')
        if recorded>pd.Timestamp.now(tz='UTC'):raise ValueError('Timing evidence recorded in the future')
        try:
            actual=_timing_utc(payload['ActualRaceStartUTC'],'ActualRaceStartUTC')
            verified=_timing_utc(payload['ActualRaceStartVerifiedAtUTC'],'ActualRaceStartVerifiedAtUTC')
            if not payload['ActualRaceStartSource'].strip() or not actual<=verified<=recorded:
                raise ValueError('Actual start verification unavailable at evidence recording')
            prediction_time=_timing_utc(prediction['PredictionTimestampUTC'],'PredictionTimestampUTC')
            classification='actual_start_timing_pass' if prediction_time<actual else 'non_promotion_grade_timing'
        except (ValueError,TypeError):classification='timing_unproven'
        if payload['TimingEvidenceClass']!=classification:
            raise ValueError('Stored timing classification inconsistent with evidence')
        previous_class=classification
    result.update(TimingEvidenceClass=previous_class,TimingEvidenceRecords=len(records),
        TimingDetail=text(records.iloc[-1]['Reason']) or 'Verified record linkage and strict actual-start comparison')
    return result

@_service_storage_transaction
def evaluate_race(year, gp_slug, actual_fastest_lap_sec):
    # Historical KPI behavior is preserved. New-policy reporting includes timing
    # evidence but does not equate evaluated status with promotion-grade evidence.
    overview=read_csv_checked(CONFIG.overview_path)
    mask=_require_one_event(overview,int(year),str(gp_slug),'df_overview')
    row=overview.loc[mask].iloc[0]
    from .readiness import _timing_policy
    policy=_timing_policy(row)
    if policy in {'','legacy_cutoff_v1'}:
        return _historical_evaluate_race(year,gp_slug,actual_fastest_lap_sec)
    if policy!='actual_race_start_v1':raise ValueError('Unsupported explicit timing policy')
    from .prediction import FINAL_PREDICTION_COLS
    from .readiness import validate_frozen_input_integrity
    test=read_csv_checked(CONFIG.test_path)
    test_mask=_require_one_event(test,int(year),str(gp_slug),'df_test')
    integrity=validate_frozen_input_integrity(test.loc[test_mask].iloc[0],row)
    if not integrity.ok:raise ValueError(integrity.detail)
    ledger=read_csv_checked(CONFIG.prediction_ledger_path)
    matches=ledger.loc[ledger['race_event_id'].astype(str).eq(f'{int(year)}_{str(gp_slug).strip()}')]
    if len(matches)!=1:raise ValueError('Exactly one immutable event prediction required')
    prediction=matches.iloc[0]
    run_id=str(prediction['RunID'])
    for field in FINAL_PREDICTION_COLS:
        if field not in row or not np.isfinite(float(row[field])) or float(row[field])!=float(prediction[field]):
            raise ValueError('Overview prediction differs from immutable ledger: '+field)
    for field in ('RunID','PredictionTimestampUTC','PredictionRowHash'):
        if field in row and pd.notna(row[field]) and str(row[field])!=str(prediction[field]):
            raise ValueError('Overview prediction metadata mismatch: '+field)
    summary=get_actual_start_timing_summary(year,gp_slug,run_id)
    result=_historical_evaluate_race(year,gp_slug,actual_fastest_lap_sec)
    result.update(summary)
    return result
