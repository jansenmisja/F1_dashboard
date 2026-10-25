"""
Prediction service for the F1 Fastest Lap pipeline.

Phase 5B:
- existing production prediction logic mechanically extracted
- paths redirected to central PipelineConfig
- explicit dependency preflight added
- no Streamlit dependency

Business logic is intentionally preserved as much as possible.
Further cleanup happens only after regression tests pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import hashlib
import json
import math
import os
import uuid

import numpy as np
import pandas as pd

from utils import helpers

from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)

try:
    from xgboost import XGBRegressor
    XGBOOST_IMPORT_ERROR = None
except Exception as exc:
    XGBRegressor = None
    XGBOOST_IMPORT_ERROR = repr(exc)

try:
    from catboost import CatBoostRegressor
    CATBOOST_IMPORT_ERROR = None
except Exception as exc:
    CatBoostRegressor = None
    CATBOOST_IMPORT_ERROR = repr(exc)

from .config import (
    CONFIG,
    PRIMARY_KEY,
    OPERATIONAL_BASELINE_COLUMN,
    OPERATIONAL_PREDICTION_COLUMN,
    PREDICTION_COLUMNS,
)

from .io import (
    read_csv_checked,
    safe_commit_csv,
    safe_commit_many,
    round_float_columns,
    io_preflight,
)

from .schemas import (
    get_target_mask,
    require_one_event,
    validate_training_contract,
    validate_test_contract,
    validate_overview_contract,
)

from .lifecycle import (
    get_event_state,
    transition_event_status,
    validate_status_transition,
)

from .registry import (
    get_registry_state,
    build_race_event_id,
    validate_registry_row,
    REGISTRY_REQUIRED_COLUMNS,
)

# Legacy compatibility for mechanically extracted prediction logic.
REGISTRY_REQUIRED_COLS = REGISTRY_REQUIRED_COLUMNS



# Legacy compatibility alias for extracted prediction code
# ============================================================
# Compatibility constants
#
# These are retained temporarily because the extracted functions
# may still reference them internally. Paths themselves now come
# exclusively from CONFIG.
# ============================================================

FINAL_PREDICTION_COLS = (
    "Pred_GBR",
    "Pred_XGB",
    "Pred_CAT",
    "PredictedFastestLap_sec",
    "Pred_GBR_offset_sec",
    "Pred_XGB_offset_sec",
    "Pred_CAT_offset_sec",
    "pred_offset_sec",
)

PREDICTION_LEDGER_REQUIRED_COLS = (
    "RunID",
    "Year",
    "gp_slug",
    "Pred_GBR",
    "Pred_XGB",
    "Pred_CAT",
    "PredictedFastestLap_sec",
    "ImmutableFlag",
)


# ============================================================
# Dependency preflight
# ============================================================

@dataclass
class PredictionDependencyState:
    sklearn_available: bool
    xgboost_available: bool
    catboost_available: bool

    degraded_mode: bool

    CATFallbackUsed: int
    CATFallbackReason: str

    blockers: List[str]


def dependency_preflight(
) -> PredictionDependencyState:

    blockers = []

    # sklearn is mandatory
    try:

        import sklearn  # noqa

        sklearn_available = True

    except Exception as exc:

        sklearn_available = False

        blockers.append(
            "scikit-learn unavailable: "
            + repr(exc)
        )

    # XGBoost is mandatory
    xgboost_available = (
        XGBRegressor
        is not None
    )

    if not xgboost_available:

        blockers.append(
            "xgboost unavailable: "
            + str(
                XGBOOST_IMPORT_ERROR
            )
        )

    # CatBoost may degrade explicitly to fallback
    catboost_available = (
        CatBoostRegressor
        is not None
    )

    degraded_mode = (
        not catboost_available
    )

    if catboost_available:

        fallback_used = 0
        fallback_reason = ""

    else:

        fallback_used = 1

        fallback_reason = (
            "CatBoost unavailable; "
            "HistGradientBoostingRegressor fallback required. "
            + str(
                CATBOOST_IMPORT_ERROR
            )
        )

    return PredictionDependencyState(
        sklearn_available=(
            sklearn_available
        ),

        xgboost_available=(
            xgboost_available
        ),

        catboost_available=(
            catboost_available
        ),

        degraded_mode=(
            degraded_mode
        ),

        CATFallbackUsed=(
            fallback_used
        ),

        CATFallbackReason=(
            fallback_reason
        ),

        blockers=blockers,
    )


def assert_prediction_dependencies(
) -> PredictionDependencyState:

    state = dependency_preflight()

    if state.blockers:

        raise RuntimeError(
            "Prediction dependency preflight failed: "
            + " | ".join(
                state.blockers
            )
        )

    return state


# ============================================================
# Temporary compatibility helpers
# ============================================================


def require_one(
    df,
    year,
    gp_slug,
    dataset_name="dataset",
):
    """
    Legacy compatibility alias.

    Delegates to canonical schemas.require_one_event().
    No duplicated business logic.
    """
    return require_one_event(
        df,
        int(year),
        str(gp_slug).strip(),
        dataset_name=dataset_name,
    )

def safe_read_csv(
    path,
    *args,
    **kwargs,
):
    return read_csv_checked(
        path,
        **kwargs,
    )


def safe_to_csv(
    df,
    path,
    *args,
    **kwargs,
):
    return safe_commit_csv(
        df,
        path,
        backup_dir=CONFIG.backup_dir,
    )


def helper_preflight(
):
    result = io_preflight()

    if not result["ok"]:

        raise RuntimeError(
            "I/O preflight failed: "
            + repr(
                result["checks"]
            )
        )

    return result


def _final_prediction_hash_text(value) -> str:

    import hashlib

    return hashlib.sha256(
        str(value).encode(
            "utf-8"
        )
    ).hexdigest()


def _final_prediction_row_hash(
    row_dict: dict,
) -> str:

    import hashlib
    import json

    payload = {}

    for key in sorted(
        row_dict.keys()
    ):

        if key == "PredictionRowHash":
            continue

        value = row_dict[key]

        if pd.isna(value):
            value = None

        elif isinstance(
            value,
            (
                np.integer,
                np.floating,
            ),
        ):
            value = float(value)

        payload[key] = value

    raw = json.dumps(
        payload,
        sort_keys=True,
        default=str,
        separators=(
            ",",
            ":",
        ),
    )

    return hashlib.sha256(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()


def _resolve_training_target(
    df_training: pd.DataFrame,
) -> str:

    candidates = [
        "ActualFastestLap_sec",
        "FastestLap_sec",
        "TargetFastestLap_sec",
    ]

    for col in candidates:

        if col in df_training.columns:

            numeric = pd.to_numeric(
                df_training[col],
                errors="coerce",
            )

            if numeric.notna().sum() >= 5:
                return col

    raise ValueError(
        "Geen bruikbare training-targetkolom gevonden. "
        "Verwacht ActualFastestLap_sec of equivalent."
    )


def build_final_prediction_features(
    df_training: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
) -> list:
    """FIXED_PRODUCTION_FEATURES_V1: vaste 32-feature referentieset."""
    from pathlib import Path

    path = (
        Path(CONFIG.training_path).parent
        / "audit"
        / "production_safe_fixed_feature_set.csv"
    )
    fixed = pd.read_csv(path)

    required = {"Feature", "FeatureOrder", "ProductionSafeFlag", "FinalDecision"}
    if not required.issubset(fixed.columns):
        raise ValueError("Vaste featurelijst heeft een onverwacht schema.")

    if not pd.to_numeric(
        fixed["ProductionSafeFlag"], errors="coerce"
    ).eq(1).all():
        raise ValueError("Featurelijst bevat niet-goedgekeurde features.")

    if not fixed["FinalDecision"].eq("include_fixed_feature_set").all():
        raise ValueError("Featurelijst bevat onverwachte beslissingen.")

    order = pd.to_numeric(fixed["FeatureOrder"], errors="raise")
    if order.isna().any() or order.duplicated().any():
        raise ValueError("FeatureOrder ontbreekt of is niet uniek.")

    features = (
        fixed.assign(_order=order)
        .sort_values("_order")["Feature"]
        .astype(str).str.strip().tolist()
    )
    if len(features) != 32 or len(set(features)) != 32:
        raise ValueError("Verwacht precies 32 unieke vaste features.")

    if (
        target_col in features
        or "Year" in features
        or "IsSprintWeekend" in features
        or any(name.startswith("Regime2026_") for name in features)
    ):
        raise ValueError("Onverwacht uitgesloten veld in de vaste featurelijst.")

    for label, frame in (("training", df_training), ("target", df_test)):
        missing = [name for name in features if name not in frame.columns]
        if missing:
            raise ValueError(f"{label}: ontbrekende vaste features: {missing}")

    return features


def _prepare_final_prediction_matrix(
    df_training: pd.DataFrame,
    target_df: pd.DataFrame,
    features: list,
    target_col: str,
):
    """FIXED_PRODUCTION_PREPROCESSING_V1: training-only imputatie/encoding."""
    categorical = {
        "Altitude_bucket", "Direction", "Downforce_profile",
        "Track_type", "pole_compound",
    }
    if len(features) != 32 or not categorical.issubset(features):
        raise ValueError("Verwacht de vaste 32 features inclusief 5 categorievelden.")

    y = pd.to_numeric(df_training[target_col], errors="coerce")
    valid = y.notna() & np.isfinite(y)
    training = df_training.loc[valid].copy()
    y_train = y.loc[valid].astype(float)

    if len(training) < 10:
        raise ValueError("Te weinig geldige trainingrijen.")

    train_parts = []
    target_parts = []

    for column in features:
        if column in categorical:
            tr = training[column].astype("string").str.strip()
            te = target_df[column].astype("string").str.strip()
            tr = tr.replace("", pd.NA).fillna("__MISSING__")
            te = te.replace("", pd.NA).fillna("__MISSING__")

            # Categorieën uitsluitend uit training.
            categories = sorted(tr.unique().tolist())
            train_encoded = pd.get_dummies(
                pd.Series(
                    pd.Categorical(tr, categories=categories),
                    index=training.index,
                ),
                prefix=column, dtype=float,
            )
            target_encoded = pd.get_dummies(
                pd.Series(
                    pd.Categorical(te, categories=categories),
                    index=target_df.index,
                ),
                prefix=column, dtype=float,
            ).reindex(columns=train_encoded.columns, fill_value=0.0)

            train_parts.append(train_encoded)
            target_parts.append(target_encoded)
        else:
            tr = pd.to_numeric(training[column], errors="coerce")
            te = pd.to_numeric(target_df[column], errors="coerce")
            tr = tr.replace([np.inf, -np.inf], np.nan)
            te = te.replace([np.inf, -np.inf], np.nan)

            median = tr.median()
            if pd.isna(median):
                raise ValueError(f"Geen geldige trainingwaarden voor {column}")

            train_parts.append(tr.fillna(median).to_frame(column))
            target_parts.append(te.fillna(median).to_frame(column))

    X_train = pd.concat(train_parts, axis=1).astype(float)
    X_target = pd.concat(target_parts, axis=1).astype(float)

    if X_train.columns.duplicated().any():
        raise ValueError("Dubbele kolommen na preprocessing.")
    if not X_train.columns.equals(X_target.columns):
        raise ValueError("Training- en targetkolommen verschillen.")
    if not np.isfinite(X_train.to_numpy()).all():
        raise ValueError("Ongeldige trainingmatrix.")
    if not np.isfinite(X_target.to_numpy()).all():
        raise ValueError("Ongeldige targetmatrix.")

    return X_train, y_train, X_target


def run_three_final_models(
    X_train,
    y_train,
    X_target,
) -> dict:
    """
    Draait verplicht GBR + XGB + CAT.

    CAT:
    - normale CatBoost indien beschikbaar;
    - expliciete HistGradientBoosting fallback bij fout.
    """

    from sklearn.ensemble import (
        GradientBoostingRegressor,
        HistGradientBoostingRegressor,
    )

    # --------------------------------------------------------
    # GBR
    # --------------------------------------------------------
    gbr = GradientBoostingRegressor(
        random_state=42,
        n_estimators=250,
        learning_rate=0.03,
        max_depth=3,
        loss="squared_error",
    )

    gbr.fit(
        X_train,
        y_train,
    )

    pred_gbr = float(
        gbr.predict(
            X_target
        )[0]
    )

    # --------------------------------------------------------
    # XGB — verplicht, geen silent fallback
    # --------------------------------------------------------
    try:

        from xgboost import (
            XGBRegressor
        )

    except Exception as exc:

        raise RuntimeError(
            "XGBoost is niet beschikbaar. "
            "Final prediction wordt geblokkeerd. "
            f"Details: {exc}"
        )

    xgb = XGBRegressor(
        n_estimators=350,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )

    xgb.fit(
        X_train,
        y_train,
    )

    pred_xgb = float(
        xgb.predict(
            X_target
        )[0]
    )

    # --------------------------------------------------------
    # CAT + expliciete fallback
    # --------------------------------------------------------
    cat_fallback_used = False
    cat_fallback_reason = ""
    cat_model_version = ""

    try:

        from catboost import (
            CatBoostRegressor
        )

        cat_model = CatBoostRegressor(
            iterations=350,
            learning_rate=0.03,
            depth=6,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
        )

        cat_model.fit(
            X_train,
            y_train,
        )

        pred_cat = float(
            cat_model.predict(
                X_target
            )[0]
        )

        cat_model_version = (
            "CatBoostRegressor"
        )

    except Exception as exc:

        cat_fallback_used = True
        cat_fallback_reason = (
            f"{type(exc).__name__}: {exc}"
        )

        cat_model = (
            HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=31,
                random_state=42,
            )
        )

        cat_model.fit(
            X_train,
            y_train,
        )

        pred_cat = float(
            cat_model.predict(
                X_target
            )[0]
        )

        cat_model_version = (
            "HistGradientBoostingRegressor_CAT_fallback"
        )

    return {
        "Pred_GBR":
            pred_gbr,

        "Pred_XGB":
            pred_xgb,

        "Pred_CAT":
            pred_cat,

        "ModelVersion_GBR":
            "GradientBoostingRegressor",

        "ModelVersion_XGB":
            "XGBRegressor",

        "ModelVersion_CAT":
            cat_model_version,

        "CATFallbackUsed":
            int(
                cat_fallback_used
            ),

        "CATFallbackReason":
            cat_fallback_reason,
    }


def validate_post_predict_plausibility(
    prediction_values: dict,
    record_lap_sec,
) -> list:
    """
    Hard stops voor technisch onmogelijke outputs.

    Gebruikt RecordLap alleen als plausibility-reference.
    """

    errors = []

    record = pd.to_numeric(
        record_lap_sec,
        errors="coerce",
    )

    if (
        pd.isna(record)
        or float(record) <= 0
    ):
        return [
            "RecordLap_sec ontbreekt of is ongeldig."
        ]

    record = float(
        record
    )

    for col in [
        "Pred_GBR",
        "Pred_XGB",
        "Pred_CAT",
    ]:

        value = pd.to_numeric(
            prediction_values.get(
                col
            ),
            errors="coerce",
        )

        if (
            pd.isna(value)
            or not np.isfinite(
                float(value)
            )
        ):
            errors.append(
                f"{col} is niet finite."
            )

            continue

        value = float(
            value
        )

        if value <= 0:

            errors.append(
                f"{col} moet > 0 zijn."
            )

            continue

        relative_gap = abs(
            value - record
        ) / record

        if relative_gap > 0.20:

            errors.append(
                f"{col} wijkt meer dan 20% af van RecordLap_sec "
                f"({value:.3f} vs {record:.3f})."
            )

    return errors


def _append_immutable_prediction_ledger(
    ledger_row: dict,
):
    """
    Eén immutable predictionrecord per race_event_id.
    Bestaande records worden nooit vervangen.
    """

    ledger_path = Path(
        CONFIG.prediction_ledger_path
    )

    ledger_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if ledger_path.exists():

        ledger_df = (
            helpers.safe_read_csv(
                ledger_path
            )
        )

    else:

        ledger_df = pd.DataFrame(
            columns=
                PREDICTION_LEDGER_REQUIRED_COLS
        )

    ledger_df = ensure_columns(
        ledger_df,
        PREDICTION_LEDGER_REQUIRED_COLS,
    )

    race_event_id = str(
        ledger_row[
            "race_event_id"
        ]
    )

    if len(
        ledger_df
    ):

        same_event = (
            ledger_df[
                "race_event_id"
            ]
            .astype(str)
            .eq(
                race_event_id
            )
        )

        if int(
            same_event.sum()
        ) > 0:

            raise ValueError(
                "Immutable prediction ledger bevat al een "
                f"prediction voor {race_event_id}. "
                "Bestaand record wordt niet overschreven."
            )

    old_hashes = set(
        ledger_df[
            "PredictionRowHash"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    new_df = pd.concat(
        [
            ledger_df,
            pd.DataFrame(
                [ledger_row]
            ),
        ],
        ignore_index=True,
    )

    helpers.safe_to_csv(
        new_df,
        ledger_path,
        index=False,
    )

    reread = helpers.safe_read_csv(
        ledger_path
    )

    if len(
        reread
    ) != len(
        ledger_df
    ) + 1:

        raise RuntimeError(
            "Ledger append-validatie faalt: rowcount incorrect."
        )

    reread_old_hashes = set(
        reread[
            "PredictionRowHash"
        ]
        .dropna()
        .astype(str)
        .tolist()
    )

    if not old_hashes.issubset(
        reread_old_hashes
    ):

        raise RuntimeError(
            "Immutable ledger validatie faalt: "
            "bestaande PredictionRowHash verdwenen."
        )

    new_hash = str(
        ledger_row[
            "PredictionRowHash"
        ]
    )

    if int(
        reread[
            "PredictionRowHash"
        ]
        .astype(str)
        .eq(
            new_hash
        )
        .sum()
    ) != 1:

        raise RuntimeError(
            "Nieuw ledgerrecord niet exact één keer teruggevonden."
        )


def _append_final_prediction_model_log(
    log_row: dict,
):
    """
    Append-only model_log.
    """

    log_path = Path(
        CONFIG.model_log_path
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if log_path.exists():

        log_df = (
            helpers.safe_read_csv(
                log_path
            )
        )

    else:

        log_df = pd.DataFrame()

    old_count = len(
        log_df
    )

    # Nieuwe kolommen toevoegen zonder oude data te verwijderen
    all_cols = list(
        dict.fromkeys(
            list(
                log_df.columns
            )
            + list(
                log_row.keys()
            )
        )
    )

    log_df = ensure_columns(
        log_df,
        all_cols,
    )

    append_row = {
        col:
            log_row.get(
                col,
                np.nan,
            )
        for col in all_cols
    }

    log_df = pd.concat(
        [
            log_df,
            pd.DataFrame(
                [append_row]
            ),
        ],
        ignore_index=True,
    )

    helpers.safe_to_csv(
        log_df,
        log_path,
        index=False,
    )

    reread = helpers.safe_read_csv(
        log_path
    )

    if len(
        reread
    ) != old_count + 1:

        raise RuntimeError(
            "model_log append-validatie faalt."
        )

    if (
        "RunID"
        not in reread.columns
        or int(
            reread[
                "RunID"
            ]
            .astype(str)
            .eq(
                str(
                    log_row[
                        "RunID"
                    ]
                )
            )
            .sum()
        ) != 1
    ):

        raise RuntimeError(
            "Nieuwe model_log RunID niet exact één keer gevonden."
        )


from .storage_transactions import transactional as _service_storage_transaction

@_service_storage_transaction
def run_final_prediction(
    year: int,
    gp_slug: str,
) -> dict:
    """
    Finale pre-race predict-run.

    Stap 7 schrijft:
    - Pred_GBR / Pred_XGB / Pred_CAT;
    - PredictedFastestLap_sec = Pred_GBR;
    - offsets;
    - immutable ledger;
    - append-only model_log.

    RaceStatus blijft hier prediction_ready.
    Promotion naar predicted gebeurt in Step 8.
    """

    # --------------------------------------------------------
    # 1. Hardened preflight
    # --------------------------------------------------------
    preflight = helper_preflight()

    if not preflight.get(
        "ok",
        False,
    ):

        raise RuntimeError(
            "Helper-preflight faalt: "
            + " | ".join(
                preflight.get(
                    "errors",
                    [],
                )
            )
        )

    # --------------------------------------------------------
    # 2. Verse datasets
    # --------------------------------------------------------
    df_training = (
        helpers.safe_read_csv(
            CONFIG.training_path
        )
    )

    df_test = (
        helpers.safe_read_csv(
            CONFIG.test_path
        )
    )

    df_overview = (
        helpers.safe_read_csv(
            CONFIG.overview_path
        )
    )

    mt = require_one(
        df_test,
        year,
        gp_slug,
        "df_test",
    )

    mo = require_one(
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
    # 3. Alleen prediction_ready
    # --------------------------------------------------------
    status_test = str(
        test_row.get(
            "RaceStatus",
            "",
        )
    ).strip()

    status_overview = str(
        overview_row.get(
            "RaceStatus",
            "",
        )
    ).strip()

    if (
        status_test
        != "prediction_ready"
        or status_overview
        != "prediction_ready"
    ):

        raise ValueError(
            "Run final prediction mag alleen bij "
            "RaceStatus == prediction_ready in beide datasets. "
            f"df_test={status_test!r}, "
            f"df_overview={status_overview!r}"
        )

    _assert_no_existing_final_prediction(year, gp_slug)
    if str(test_row.get("PredictionTimingPolicyVersion", "")).strip() != "actual_race_start_v1":
        raise ValueError("New final runs require explicit actual_race_start_v1; historical records are not migrated")
    source_fingerprints = _prediction_source_fingerprints()

    # --------------------------------------------------------
    # 4. Leakage guard opnieuw
    # --------------------------------------------------------
    leakage_errors = (
        validate_leakage_guard(
            df_test,
            df_overview,
            year,
            gp_slug,
        )
    )

    if leakage_errors:

        raise ValueError(
            "Final prediction leakage guard faalt: "
            + " | ".join(
                leakage_errors
            )
        )

    # --------------------------------------------------------
    # 5. Registry moet echt eligible zijn
    # --------------------------------------------------------
    registry_path = Path(
        CONFIG.registry_eligible_mirror_path
    )

    if not registry_path.exists():

        raise ValueError(
            "future_event_registry_eligible.csv ontbreekt."
        )

    eligible_df = (
        helpers.safe_read_csv(
            registry_path
        )
    )

    registry_errors = (
        validate_registry_row(
            eligible_df,
            year,
            gp_slug,
            require_eligible=True,
        )
    )

    if registry_errors:

        raise ValueError(
            "Registry gate faalt: "
            + " | ".join(
                registry_errors
            )
        )

    # --------------------------------------------------------
    # 6. Training target + features
    # --------------------------------------------------------
    target_col = (
        _resolve_training_target(
            df_training
        )
    )

    target_test_df = (
        df_test
        .loc[mt]
        .copy()
    )

    features = (
        build_final_prediction_features(
            df_training,
            target_test_df,
            target_col,
        )
    )

    (
        X_train,
        y_train,
        X_target,
    ) = _prepare_final_prediction_matrix(
        df_training,
        target_test_df,
        features,
        target_col,
    )

    # --------------------------------------------------------
    # 7. Feature-set hash
    # --------------------------------------------------------
    feature_set_hash = (
        _final_prediction_hash_text(
            "|".join(
                sorted(
                    features
                )
            )
        )
    )

    # --------------------------------------------------------
    # 8. Alle drie modellen draaien
    # --------------------------------------------------------
    # FORWARD_RUN_PROVENANCE_V1
    run_provenance = _build_forward_run_provenance(
        df_training, features, target_col, X_train, y_train
    )

    model_result = (
        run_three_final_models(
            X_train,
            y_train,
            X_target,
        )
    )

    # --------------------------------------------------------
    # 9. Post-predict plausibility
    # --------------------------------------------------------
    record_lap = pd.to_numeric(
        test_row.get(
            "RecordLap_sec"
        ),
        errors="coerce",
    )

    plausibility_errors = (
        validate_post_predict_plausibility(
            model_result,
            record_lap,
        )
    )

    if plausibility_errors:

        raise ValueError(
            "Post-predict plausibility faalt: "
            + " | ".join(
                plausibility_errors
            )
        )

    record_lap = float(
        record_lap
    )

    pred_gbr = float(
        model_result[
            "Pred_GBR"
        ]
    )

    pred_xgb = float(
        model_result[
            "Pred_XGB"
        ]
    )

    pred_cat = float(
        model_result[
            "Pred_CAT"
        ]
    )

    prediction_values = {
        "Pred_GBR":
            pred_gbr,

        "Pred_XGB":
            pred_xgb,

        "Pred_CAT":
            pred_cat,

        # Operational canonical baseline
        "PredictedFastestLap_sec":
            pred_gbr,

        "Pred_GBR_offset_sec":
            pred_gbr - record_lap,

        "Pred_XGB_offset_sec":
            pred_xgb - record_lap,

        "Pred_CAT_offset_sec":
            pred_cat - record_lap,

        "pred_offset_sec":
            pred_gbr - record_lap,
    }

    # --------------------------------------------------------
    # 10. Run metadata
    # --------------------------------------------------------
    now_ts = _timing_now_utc()

    timestamp = now_ts.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    run_id = (
        f"{int(year)}_{str(gp_slug).strip()}_"
        f"{now_ts.strftime('%Y%m%dT%H%M%SZ')}"
    )

    race_event_id = (
        build_race_event_id(
            year,
            gp_slug,
        )
    )

    # --------------------------------------------------------
    # 11. Overview in memory uitbreiden
    #     df_test NIET wijzigen
    # --------------------------------------------------------
    df_overview_new = (
        df_overview.copy()
    )

    df_overview_new = (
        ensure_columns(
            df_overview_new,
            FINAL_PREDICTION_COLS,
        )
    )

    mo_new = require_one(
        df_overview_new,
        year,
        gp_slug,
        "df_overview",
    )

    for col, value in (
        prediction_values.items()
    ):

        df_overview_new.loc[
            mo_new,
            col,
        ] = value

    # --------------------------------------------------------
    # 12. Controle dat df_test geen predictionkolommen
    #     krijgt door deze run
    # --------------------------------------------------------
    forbidden_test_cols = [
        col
        for col in FINAL_PREDICTION_COLS
        if col in df_test.columns
    ]

    # Bestaande historische contaminatie blokkeren we
    # alleen voor de targetrij niet door columnschema-mutatie.
    # Deze functie schrijft df_test helemaal niet.
    test_columns_before = tuple(
        df_test.columns
    )

    # --------------------------------------------------------
    # 15. Immutable ledger row
    # --------------------------------------------------------
    ledger_row = {
        "RunID":
            run_id,

        "PredictionTimestampUTC":
            timestamp,

        "race_event_id":
            race_event_id,

        "Year":
            int(year),

        "gp_slug":
            str(
                gp_slug
            ).strip(),

        "Circuit":
            test_row.get(
                "Circuit",
                "",
            ),

        "CircuitLayoutSlug":
            test_row.get(
                "CircuitLayoutSlug",
                "",
            ),

        "EventDateTimeUTC":
            test_row.get(
                "EventDateTimeUTC",
                "",
            ),

        "PredictionCutoffUTC":
            test_row.get(
                "PredictionCutoffUTC",
                "",
            ),

        "FeatureSnapshotTimestampUTC":
            test_row.get(
                "FeatureSnapshotTimestampUTC",
                "",
            ),

        "FeatureSnapshotAfterQualifyingUTC":
            test_row.get(
                "FeatureSnapshotAfterQualifyingUTC",
                "",
            ),

        "ModelVersion_GBR":
            model_result[
                "ModelVersion_GBR"
            ],

        "ModelVersion_XGB":
            model_result[
                "ModelVersion_XGB"
            ],

        "ModelVersion_CAT":
            model_result[
                "ModelVersion_CAT"
            ],

        "CATFallbackUsed":
            int(
                model_result[
                    "CATFallbackUsed"
                ]
            ),

        "CATFallbackReason":
            model_result[
                "CATFallbackReason"
            ],

        "FeatureSetHash":
            feature_set_hash,

        **prediction_values,

        "RecordLap_sec":
            record_lap,

        "PostPredictPlausibilityStatus":
            "passed",

        "ImmutableFlag":
            1,

        "EvaluationStatus":
            "pending",
    }

    ledger_row.update(run_provenance)

    # --------------------------------------------------------
    # 16. model_log append-only
    # --------------------------------------------------------
    log_row = {
        "RunID":
            run_id,

        "TimestampUTC":
            timestamp,

        "Year":
            int(year),

        "gp_slug":
            str(
                gp_slug
            ).strip(),

        "race_event_id":
            race_event_id,

        "RunType":
            "final_pre_race_prediction",

        "RaceStatus":
            "prediction_ready",

        "TrainingRows":
            int(
                len(
                    X_train
                )
            ),

        "FeatureCount":
            int(
                len(
                    features
                )
            ),

        "FeatureSetHash":
            feature_set_hash,

        "Pred_GBR":
            pred_gbr,

        "Pred_XGB":
            pred_xgb,

        "Pred_CAT":
            pred_cat,

        "PredictedFastestLap_sec":
            pred_gbr,

        "CATFallbackUsed":
            int(
                model_result[
                    "CATFallbackUsed"
                ]
            ),

        "CATFallbackReason":
            model_result[
                "CATFallbackReason"
            ],

        "PostPredictPlausibility":
            "passed",

        "Commentaar":
            (
                "CAT fallback gebruikt: "
                + model_result[
                    "CATFallbackReason"
                ]
                if model_result[
                    "CATFallbackUsed"
                ]
                else "GBR + XGB + CAT succesvol"
            ),
    }

    log_row.update(run_provenance)
    for field in ("ModelVersion_GBR", "ModelVersion_XGB", "ModelVersion_CAT"):
        log_row[field] = model_result[field]

    _commit_timed_final_prediction(
        year, gp_slug, df_test, df_overview, df_overview_new,
        ledger_row, log_row, source_fingerprints,
    )
    run_id = ledger_row["RunID"]

    return {
        "ok": True,

        "RunID":
            run_id,

        "RaceStatus":
            "prediction_ready",

        "Pred_GBR":
            round(
                pred_gbr,
                3,
            ),

        "Pred_XGB":
            round(
                pred_xgb,
                3,
            ),

        "Pred_CAT":
            round(
                pred_cat,
                3,
            ),

        "PredictedFastestLap_sec":
            round(
                pred_gbr,
                3,
            ),

        "Pred_GBR_offset_sec":
            round(
                pred_gbr
                - record_lap,
                3,
            ),

        "Pred_XGB_offset_sec":
            round(
                pred_xgb
                - record_lap,
                3,
            ),

        "Pred_CAT_offset_sec":
            round(
                pred_cat
                - record_lap,
                3,
            ),

        "CATFallbackUsed":
            int(
                model_result[
                    "CATFallbackUsed"
                ]
            ),

        "CATFallbackReason":
            model_result[
                "CATFallbackReason"
            ],

        "FeatureCount":
            int(
                len(
                    features
                )
            ),

        "LedgerPath":
            CONFIG.prediction_ledger_path,

        "ModelLogPath":
            CONFIG.model_log_path,
    }

# ============================================================
# PHASE 5C.7 — STATUS TRANSITION COMPATIBILITY
# ============================================================
def _validate_status_transition_errors(
    *args,
    **kwargs,
):
    """
    Legacy adapter.

    Canonical lifecycle.validate_status_transition() may return
    bool, while the extracted legacy artifact validator expects
    an iterable of error strings.
    """

    result = validate_status_transition(
        *args,
        **kwargs,
    )

    # Canonical current behavior
    if isinstance(
        result,
        bool,
    ):

        return (
            []
            if result
            else [
                "Invalid lifecycle status transition"
            ]
        )

    # Legacy-compatible result
    if result is None:

        return []

    if isinstance(
        result,
        (list, tuple, set),
    ):

        return list(
            result
        )

    # Dataclass / result-object semantics
    if hasattr(
        result,
        "ok",
    ):

        if bool(
            result.ok
        ):
            return []

        if hasattr(
            result,
            "errors",
        ):

            errors = getattr(
                result,
                "errors",
            )

            if isinstance(
                errors,
                (list, tuple, set),
            ):

                return list(
                    errors
                )

        return [
            "Invalid lifecycle status transition"
        ]

    # Dict semantics
    if isinstance(
        result,
        dict,
    ):

        if bool(
            result.get(
                "ok",
                False,
            )
        ):

            return []

        errors = result.get(
            "errors",
            [],
        )

        if isinstance(
            errors,
            (list, tuple, set),
        ):

            return list(
                errors
            )

        return [
            str(errors)
        ]

    raise TypeError(
        "Unsupported validate_status_transition "
        f"return type: {type(result).__name__}"
    )


def _validate_final_prediction_artifacts_raw(
    df_test: pd.DataFrame,
    df_overview: pd.DataFrame,
    year: int,
    gp_slug: str,
) -> dict:
    """
    Volledige promotion-grade validatie vóór RaceStatus=predicted.

    Geen writes.
    """

    errors = []
    checks = {}

    # --------------------------------------------------------
    # 1. Target exact 1x
    # --------------------------------------------------------
    mt = get_target_mask(
        df_test,
        year,
        gp_slug,
    )

    mo = get_target_mask(
        df_overview,
        year,
        gp_slug,
    )

    test_count = int(
        mt.sum()
    )

    overview_count = int(
        mo.sum()
    )

    checks[
        "target_unique_df_test"
    ] = (
        test_count == 1
    )

    checks[
        "target_unique_df_overview"
    ] = (
        overview_count == 1
    )

    if test_count != 1:
        errors.append(
            f"df_test target count={test_count}; verwacht exact 1."
        )

    if overview_count != 1:
        errors.append(
            f"df_overview target count={overview_count}; verwacht exact 1."
        )

    if errors:
        return {
            "ok": False,
            "errors": errors,
            "checks": checks,
        }

    rt = (
        df_test
        .loc[mt]
        .iloc[0]
    )

    ro = (
        df_overview
        .loc[mo]
        .iloc[0]
    )

    # --------------------------------------------------------
    # 2. Status moet nog prediction_ready zijn
    # --------------------------------------------------------
    status_test = str(
        rt.get(
            "RaceStatus",
            "",
        )
    ).strip()

    status_overview = str(
        ro.get(
            "RaceStatus",
            "",
        )
    ).strip()

    status_ok = (
        status_test
        == "prediction_ready"
        and status_overview
        == "prediction_ready"
    )

    checks[
        "status_prediction_ready"
    ] = status_ok

    if not status_ok:
        errors.append(
            "Promotion naar predicted mag alleen vanuit "
            "prediction_ready in beide datasets. "
            f"df_test={status_test!r}, "
            f"df_overview={status_overview!r}"
        )

    # --------------------------------------------------------
    # 3. Status transition guard
    # --------------------------------------------------------
    transition_test = (
        _validate_status_transition_errors(
            status_test,
            "predicted",
        )
    )

    transition_overview = (
        _validate_status_transition_errors(
            status_overview,
            "predicted",
        )
    )

    transition_ok = (
        not transition_test
        and not transition_overview
    )

    checks[
        "status_transition"
    ] = transition_ok

    if transition_test:
        errors.extend(
            [
                "df_test: " + err
                for err in transition_test
            ]
        )

    if transition_overview:
        errors.extend(
            [
                "df_overview: " + err
                for err in transition_overview
            ]
        )

    # --------------------------------------------------------
    # 4. df_test mag geen predictionkolommen bevatten
    # --------------------------------------------------------
    forbidden_test_cols = [
        col
        for col in FINAL_PREDICTION_COLS
        if col in df_test.columns
    ]

    test_schema_ok = (
        len(
            forbidden_test_cols
        ) == 0
    )

    checks[
        "df_test_no_prediction_columns"
    ] = test_schema_ok

    if forbidden_test_cols:
        errors.append(
            "df_test bevat predictionkolommen: "
            + str(
                forbidden_test_cols
            )
        )

    # --------------------------------------------------------
    # 5. Overview predictions compleet
    # --------------------------------------------------------
    required_prediction_cols = [
        "Pred_GBR",
        "Pred_XGB",
        "Pred_CAT",
        "PredictedFastestLap_sec",
        "Pred_GBR_offset_sec",
        "Pred_XGB_offset_sec",
        "Pred_CAT_offset_sec",
        "pred_offset_sec",
    ]

    missing_prediction_cols = [
        col
        for col in required_prediction_cols
        if col not in df_overview.columns
    ]

    # Altijd initialiseren zodat latere validatie
    # niet afhankelijk is van locals()/comprehension scope.
    prediction_values = {}

    if missing_prediction_cols:

        errors.append(
            "df_overview mist predictionkolommen: "
            + str(
                missing_prediction_cols
            )
        )

        checks[
            "overview_predictions_complete"
        ] = False

    else:

        prediction_values = {}

        for col in required_prediction_cols:

            value = pd.to_numeric(
                ro.get(
                    col
                ),
                errors="coerce",
            )

            if (
                pd.isna(value)
                or not np.isfinite(
                    float(value)
                )
            ):

                errors.append(
                    f"df_overview {col} is leeg/niet finite."
                )

            else:

                prediction_values[
                    col
                ] = float(
                    value
                )

        checks[
            "overview_predictions_complete"
        ] = (
            len(
                prediction_values
            )
            == len(
                required_prediction_cols
            )
        )

    # --------------------------------------------------------
    # Alleen verdere predictionchecks als waarden bestaan
    # --------------------------------------------------------
    if (
        len(prediction_values)
        == len(required_prediction_cols)
        and all(
            col in prediction_values
            for col in required_prediction_cols
        )
    ):

        # ----------------------------------------------------
        # 6. Operational prediction = GBR
        # ----------------------------------------------------
        operational_ok = np.isclose(
            prediction_values[
                "PredictedFastestLap_sec"
            ],
            prediction_values[
                "Pred_GBR"
            ],
            atol=0.001,
        )

        checks[
            "operational_equals_gbr"
        ] = operational_ok

        if not operational_ok:

            errors.append(
                "PredictedFastestLap_sec is niet gelijk aan Pred_GBR."
            )

        # ----------------------------------------------------
        # 7. Offset consistency
        # ----------------------------------------------------
        record_lap = pd.to_numeric(
            ro.get(
                "RecordLap_sec"
            ),
            errors="coerce",
        )

        if (
            pd.isna(record_lap)
            or float(
                record_lap
            ) <= 0
        ):

            offset_ok = False

            errors.append(
                "RecordLap_sec ontbreekt/ongeldig voor offsetvalidatie."
            )

        else:

            record_lap = float(
                record_lap
            )

            expected_offsets = {
                "Pred_GBR_offset_sec":
                    prediction_values[
                        "Pred_GBR"
                    ]
                    - record_lap,

                "Pred_XGB_offset_sec":
                    prediction_values[
                        "Pred_XGB"
                    ]
                    - record_lap,

                "Pred_CAT_offset_sec":
                    prediction_values[
                        "Pred_CAT"
                    ]
                    - record_lap,

                "pred_offset_sec":
                    prediction_values[
                        "Pred_GBR"
                    ]
                    - record_lap,
            }

            offset_errors = []

            for col, expected in (
                expected_offsets.items()
            ):

                actual = prediction_values[
                    col
                ]

                if not np.isclose(
                    actual,
                    expected,
                    atol=0.002,
                ):

                    offset_errors.append(
                        f"{col}: actual={actual:.3f}, "
                        f"expected={expected:.3f}"
                    )

            offset_ok = (
                not offset_errors
            )

            errors.extend(
                offset_errors
            )

        checks[
            "offsets_consistent"
        ] = offset_ok

        # ----------------------------------------------------
        # 8. Post-predict plausibility opnieuw
        # ----------------------------------------------------
        plausibility_errors = (
            validate_post_predict_plausibility(
                {
                    "Pred_GBR":
                        prediction_values[
                            "Pred_GBR"
                        ],

                    "Pred_XGB":
                        prediction_values[
                            "Pred_XGB"
                        ],

                    "Pred_CAT":
                        prediction_values[
                            "Pred_CAT"
                        ],
                },
                record_lap,
            )
        )

        checks[
            "post_predict_plausibility"
        ] = (
            not plausibility_errors
        )

        if plausibility_errors:

            errors.extend(
                [
                    "Post-predict: " + err
                    for err in plausibility_errors
                ]
            )

    # --------------------------------------------------------
    # 9. Immutable ledger bestaat
    # --------------------------------------------------------
    ledger_path = Path(
        CONFIG.prediction_ledger_path
    )

    if not ledger_path.exists():

        checks[
            "ledger_exists"
        ] = False

        errors.append(
            "Immutable prediction ledger ontbreekt."
        )

        ledger_row = None

    else:

        checks[
            "ledger_exists"
        ] = True

        ledger_df = (
            helpers.safe_read_csv(
                ledger_path
            )
        )

        race_event_id = (
            build_race_event_id(
                year,
                gp_slug,
            )
        )

        if (
            "race_event_id"
            not in ledger_df.columns
        ):

            errors.append(
                "Ledger mist race_event_id."
            )

            ledger_row = None

        else:

            ledger_mask = (
                ledger_df[
                    "race_event_id"
                ]
                .astype(str)
                .eq(
                    race_event_id
                )
            )

            ledger_count = int(
                ledger_mask.sum()
            )

            checks[
                "ledger_exact_one"
            ] = (
                ledger_count == 1
            )

            if ledger_count != 1:

                errors.append(
                    f"Ledger target count={ledger_count}; "
                    "verwacht exact 1 immutable predictionrecord."
                )

                ledger_row = None

            else:

                ledger_row = (
                    ledger_df
                    .loc[
                        ledger_mask
                    ]
                    .iloc[0]
                )

    # --------------------------------------------------------
    # 10. Ledger integriteit
    # --------------------------------------------------------
    run_id = None

    if ledger_row is not None:

        immutable_flag = pd.to_numeric(
            ledger_row.get(
                "ImmutableFlag"
            ),
            errors="coerce",
        )

        immutable_ok = (
            pd.notna(
                immutable_flag
            )
            and int(
                immutable_flag
            ) == 1
        )

        checks[
            "ledger_immutable"
        ] = immutable_ok

        if not immutable_ok:

            errors.append(
                "Ledger ImmutableFlag is niet 1."
            )

        run_id = str(
            ledger_row.get(
                "RunID",
                "",
            )
        ).strip()

        run_id_ok = (
            bool(
                run_id
            )
            and run_id.lower()
            not in {
                "nan",
                "none",
            }
        )

        checks[
            "ledger_run_id"
        ] = run_id_ok

        if not run_id_ok:

            errors.append(
                "Ledger RunID ontbreekt."
            )

        # ----------------------------------------------------
        # Ledger predictions moeten exact overview volgen
        # ----------------------------------------------------
        ledger_prediction_ok = True

        for col in [
            "Pred_GBR",
            "Pred_XGB",
            "Pred_CAT",
            "PredictedFastestLap_sec",
            "Pred_GBR_offset_sec",
            "Pred_XGB_offset_sec",
            "Pred_CAT_offset_sec",
            "pred_offset_sec",
        ]:

            ledger_value = pd.to_numeric(
                ledger_row.get(
                    col
                ),
                errors="coerce",
            )

            overview_value = pd.to_numeric(
                ro.get(
                    col
                ),
                errors="coerce",
            )

            if (
                pd.isna(
                    ledger_value
                )
                or pd.isna(
                    overview_value
                )
                or not np.isclose(
                    float(
                        ledger_value
                    ),
                    float(
                        overview_value
                    ),
                    atol=0.001,
                )
            ):

                ledger_prediction_ok = False

                errors.append(
                    f"Ledger/overview mismatch voor {col}."
                )

        checks[
            "ledger_matches_overview"
        ] = ledger_prediction_ok

        # ----------------------------------------------------
        # CAT fallback auditvelden moeten bestaan
        # ----------------------------------------------------
        fallback_used = pd.to_numeric(
            ledger_row.get(
                "CATFallbackUsed"
            ),
            errors="coerce",
        )

        fallback_reason = str(
            ledger_row.get(
                "CATFallbackReason",
                "",
            )
        ).strip()

        fallback_audit_ok = (
            pd.notna(
                fallback_used
            )
            and int(
                fallback_used
            ) in {
                0,
                1,
            }
        )

        if (
            fallback_audit_ok
            and int(
                fallback_used
            ) == 1
            and (
                not fallback_reason
                or fallback_reason.lower()
                in {
                    "nan",
                    "none",
                }
            )
        ):

            fallback_audit_ok = False

        checks[
            "cat_fallback_audit"
        ] = fallback_audit_ok

        if not fallback_audit_ok:

            errors.append(
                "CAT fallback auditinformatie ontbreekt/incompleet."
            )

    # --------------------------------------------------------
    # 11. model_log match
    # --------------------------------------------------------
    model_log_path = Path(
        CONFIG.model_log_path
    )

    if not model_log_path.exists():

        checks[
            "model_log_exists"
        ] = False

        errors.append(
            "model_log.csv ontbreekt."
        )

    else:

        checks[
            "model_log_exists"
        ] = True

        model_log_df = (
            helpers.safe_read_csv(
                model_log_path
            )
        )

        if (
            not run_id
            or "RunID"
            not in model_log_df.columns
        ):

            checks[
                "model_log_run_match"
            ] = False

            errors.append(
                "model_log RunID-validatie niet mogelijk."
            )

        else:

            run_mask = (
                model_log_df[
                    "RunID"
                ]
                .astype(str)
                .eq(
                    run_id
                )
            )

            run_count = int(
                run_mask.sum()
            )

            model_log_match_ok = (
                run_count == 1
            )

            checks[
                "model_log_run_match"
            ] = model_log_match_ok

            if not model_log_match_ok:

                errors.append(
                    f"model_log RunID count={run_count}; verwacht exact 1."
                )

            else:

                log_row = (
                    model_log_df
                    .loc[
                        run_mask
                    ]
                    .iloc[0]
                )

                # Model-log predictions eveneens vergelijken
                for col in [
                    "Pred_GBR",
                    "Pred_XGB",
                    "Pred_CAT",
                    "PredictedFastestLap_sec",
                ]:

                    log_value = pd.to_numeric(
                        log_row.get(
                            col
                        ),
                        errors="coerce",
                    )

                    overview_value = pd.to_numeric(
                        ro.get(
                            col
                        ),
                        errors="coerce",
                    )

                    if (
                        pd.isna(
                            log_value
                        )
                        or pd.isna(
                            overview_value
                        )
                        or not np.isclose(
                            float(
                                log_value
                            ),
                            float(
                                overview_value
                            ),
                            atol=0.001,
                        )
                    ):

                        errors.append(
                            f"model_log/overview mismatch voor {col}."
                        )

    return {
        "ok":
            len(
                errors
            ) == 0,

        "errors":
            list(
                dict.fromkeys(
                    errors
                )
            ),

        "checks":
            checks,

        "RunID":
            run_id,
    }


@_service_storage_transaction
def promote_prediction_to_predicted(
    year: int,
    gp_slug: str,
) -> dict:
    """
    Official promotion from prediction_ready to predicted.

    Flow:
        1. require canonical lifecycle state prediction_ready
        2. validate complete immutable prediction artifacts
        3. transition via lifecycle.transition_event_status()
        4. verify final readback state predicted

    No direct RaceStatus writes.
    No prediction mutation.
    No ledger/model-log mutation.
    """

    year = int(year)
    gp_slug = str(
        gp_slug
    ).strip()

    # --------------------------------------------------------
    # 1. Canonical state before
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
        != "prediction_ready"
    ):

        raise ValueError(
            "promote_prediction_to_predicted requires "
            "RaceStatus=prediction_ready; "
            f"current={state_before.race_status}"
        )

    # --------------------------------------------------------
    # 2. Artifact validation
    # --------------------------------------------------------
    artifact_result = (
        validate_final_prediction_artifacts(
            year,
            gp_slug,
        )
    )

    if isinstance(
        artifact_result,
        bool,
    ):

        artifact_ok = artifact_result

    elif isinstance(
        artifact_result,
        dict,
    ):

        artifact_ok = bool(
            artifact_result.get(
                "ok",
                artifact_result.get(
                    "all_ok",
                    False,
                ),
            )
        )

    elif hasattr(
        artifact_result,
        "ok",
    ):

        artifact_ok = bool(
            artifact_result.ok
        )

    else:

        artifact_ok = False

    if not artifact_ok:

        raise RuntimeError(
            "Final prediction artifact validation failed: "
            + repr(
                artifact_result
            )
        )

    # --------------------------------------------------------
    # 3. Canonical lifecycle transition
    # --------------------------------------------------------
    transition = (
        transition_event_status(
            year=year,
            gp_slug=gp_slug,
            expected_from=(
                "prediction_ready"
            ),
            to_status=(
                "predicted"
            ),
        )
    )

    if not transition.ok:

        raise RuntimeError(
            "Lifecycle transition to predicted failed."
        )

    # --------------------------------------------------------
    # 4. Final readback
    # --------------------------------------------------------
    state_after = get_event_state(
        year,
        gp_slug,
    )

    if (
        state_after.race_status
        != "predicted"
    ):

        raise RuntimeError(
            "Final RaceStatus readback is not predicted."
        )

    result = {
        "ok":
            True,

        "Year":
            year,

        "gp_slug":
            gp_slug,

        "status_before":
            state_before.race_status,

        "status_after":
            state_after.race_status,

        "artifact_validation":
            artifact_result,
    }

    if isinstance(
        artifact_result,
        dict,
    ):

        if "RunID" in artifact_result:

            result["RunID"] = (
                artifact_result[
                    "RunID"
                ]
            )

    return result

# ============================================================
# PHASE 5C.4 — LEGACY LEAKAGE GUARD
# ============================================================
def _legacy_validate_leakage_guard(
    df_test: pd.DataFrame,
    df_overview: pd.DataFrame,
    year: int,
    gp_slug: str,
) -> list:
    """
    Minimale pre-race leakage guard.

    Controleert:
    - target uniek;
    - ActualFastestLap_sec leeg;
    - event cutoff vóór race;
    - feature snapshot aanwezig;
    - feature snapshot <= cutoff.
    """

    errors = []

    mt = get_target_mask(
        df_test,
        year,
        gp_slug,
    )

    mo = get_target_mask(
        df_overview,
        year,
        gp_slug,
    )

    if int(mt.sum()) != 1:
        errors.append(
            f"df_test target count={int(mt.sum())}; verwacht exact 1."
        )

    if int(mo.sum()) != 1:
        errors.append(
            f"df_overview target count={int(mo.sum())}; verwacht exact 1."
        )

    if errors:
        return errors

    rt = (
        df_test
        .loc[mt]
        .iloc[0]
    )

    ro = (
        df_overview
        .loc[mo]
        .iloc[0]
    )

    # --------------------------------------------------------
    # Actual mag nog niet bekend zijn
    # --------------------------------------------------------
    if not _readiness_actual_empty(
        rt.get(
            "ActualFastestLap_sec"
        )
    ):
        errors.append(
            "df_test ActualFastestLap_sec is al gevuld."
        )

    if not _readiness_actual_empty(
        ro.get(
            "ActualFastestLap_sec"
        )
    ):
        errors.append(
            "df_overview ActualFastestLap_sec is al gevuld."
        )

    # --------------------------------------------------------
    # Timing
    # --------------------------------------------------------
    event_ts = pd.to_datetime(
        rt.get(
            "EventDateTimeUTC"
        ),
        utc=True,
        errors="coerce",
    )

    cutoff_ts = pd.to_datetime(
        rt.get(
            "PredictionCutoffUTC"
        ),
        utc=True,
        errors="coerce",
    )

    if pd.isna(event_ts):
        errors.append(
            "EventDateTimeUTC ontbreekt of is ongeldig."
        )

    if pd.isna(cutoff_ts):
        errors.append(
            "PredictionCutoffUTC ontbreekt of is ongeldig."
        )

    if (
        pd.notna(event_ts)
        and pd.notna(cutoff_ts)
        and cutoff_ts >= event_ts
    ):
        errors.append(
            "PredictionCutoffUTC ligt niet vóór EventDateTimeUTC."
        )

    # --------------------------------------------------------
    # Snapshot
    # --------------------------------------------------------
    snapshot_ts = pd.to_datetime(
        rt.get(
            "FeatureSnapshotTimestampUTC"
        ),
        utc=True,
        errors="coerce",
    )

    snapshot_after_q = pd.to_datetime(
        rt.get(
            "FeatureSnapshotAfterQualifyingUTC"
        ),
        utc=True,
        errors="coerce",
    )

    if pd.isna(snapshot_ts):
        errors.append(
            "FeatureSnapshotTimestampUTC ontbreekt of is ongeldig."
        )

    if pd.isna(snapshot_after_q):
        errors.append(
            "FeatureSnapshotAfterQualifyingUTC ontbreekt of is ongeldig."
        )

    if (
        pd.notna(snapshot_ts)
        and pd.notna(cutoff_ts)
        and snapshot_ts > cutoff_ts
    ):
        errors.append(
            "FeatureSnapshotTimestampUTC ligt ná PredictionCutoffUTC."
        )

    if (
        pd.notna(snapshot_after_q)
        and pd.notna(cutoff_ts)
        and snapshot_after_q > cutoff_ts
    ):
        errors.append(
            "FeatureSnapshotAfterQualifyingUTC ligt ná PredictionCutoffUTC."
        )

    return list(
        dict.fromkeys(
            errors
        )
    )

# ============================================================
# PHASE 5C.5 — LEGACY SUPPORT HELPERS
# ============================================================
def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out

def _readiness_actual_empty(value) -> bool:

    if value is None:
        return True

    if pd.isna(value):
        return True

    if isinstance(value, str):

        return (
            value.strip().lower()
            in {
                "",
                "nan",
                "none",
                "null",
            }
        )

    numeric = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    return pd.isna(numeric)

# ============================================================
# PHASE 5C.1 — PUBLIC ARTIFACT VALIDATOR ADAPTER
# ============================================================
@_service_storage_transaction
def validate_final_prediction_artifacts(
    year: int,
    gp_slug: str,
):
    """
    Public two-argument artifact validator.

    Legacy implementation expects:
        (df_test, df_overview, year, gp_slug)

    Fresh canonical datasets are always read internally.
    """

    year = int(year)
    gp_slug = str(gp_slug).strip()

    fresh_test = read_csv_checked(
        CONFIG.test_path
    )

    fresh_overview = read_csv_checked(
        CONFIG.overview_path
    )

    report = _validate_final_prediction_artifacts_raw(fresh_test, fresh_overview, year, gp_slug)
    mt = get_target_mask(fresh_test, year, gp_slug)
    mo = get_target_mask(fresh_overview, year, gp_slug)
    if int(mt.sum()) == 1 and int(mo.sum()) == 1:
        row = fresh_test.loc[mt].iloc[0]
        if str(row.get("PredictionTimingPolicyVersion", "")) == "actual_race_start_v1":
            from .readiness import validate_frozen_input_integrity
            integrity = validate_frozen_input_integrity(row, fresh_overview.loc[mo].iloc[0])
            problems = list(validate_leakage_guard(fresh_test, fresh_overview, year, gp_slug))
            if not integrity.ok: problems.append(integrity.detail)
            try:
                ledger = read_csv_checked(CONFIG.prediction_ledger_path)
                event = build_race_event_id(year, gp_slug)
                records = ledger.loc[ledger["race_event_id"].astype(str).eq(event)]
                if len(records) != 1: raise ValueError("Expected one immutable event record")
                record = records.iloc[0]
                if str(record["PredictionRowHash"]) != _timing_record_hash(record): raise ValueError("Prediction hash invalid")
                log = read_csv_checked(CONFIG.model_log_path)
                logs = log.loc[log["RunID"].astype(str).eq(str(record["RunID"]))]
                if len(logs) != 1: raise ValueError("Expected one matching model-log record")
                for field in (*TIMING_METADATA_FIELDS, "PredictionTimestampUTC", "PredictionRowHash"):
                    if str(record[field]) != str(logs.iloc[0][field]): raise ValueError("Timing metadata mismatch: " + field)
            except (ValueError, KeyError, FileNotFoundError) as exc:
                problems.append(str(exc))
            report["errors"] = list(dict.fromkeys(report.get("errors", []) + problems))
            report["ok"] = not report["errors"]
    return report


def _build_forward_run_provenance(
    df_training, features, target_col, X_train, y_train
):
    """FORWARD_RUN_PROVENANCE_V1: metadata van werkelijk gebruikte training."""
    import hashlib
    import inspect
    import json
    import platform
    from importlib.metadata import version, PackageNotFoundError

    def digest(text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def source_hash(function):
        return digest(inspect.getsource(function))

    y = pd.to_numeric(df_training[target_col], errors="coerce")
    valid = y.notna() & np.isfinite(y)
    used = df_training.loc[valid].copy()

    if len(used) != len(X_train) or len(used) != len(y_train):
        raise ValueError("Metadata en trainingmatrix hebben andere aantallen.")
    if not used.index.equals(X_train.index):
        raise ValueError("Metadata en trainingmatrix hebben andere rijvolgorde.")
    if not X_train.index.equals(y_train.index):
        raise ValueError("Trainingmatrix en targets hebben andere rijvolgorde.")

    if "Year" not in used.columns or "gp_slug" not in used.columns:
        raise ValueError("Trainingidentiteit mist Year of gp_slug.")

    years = pd.to_numeric(used["Year"], errors="coerce")
    if (
        years.isna().any()
        or not np.isfinite(years).all()
        or not years.eq(np.floor(years)).all()
        or not years.between(2017, 2024).all()
    ):
        raise ValueError(
            "Final prediction vereist uitsluitend trainingjaren 2017–2024."
        )

    if used["gp_slug"].isna().any():
        raise ValueError("Training bevat ontbrekende gp_slug.")
    slugs = used["gp_slug"].astype(str).str.strip()
    if slugs.eq("").any():
        raise ValueError("Training bevat lege gp_slug.")

    keys = pd.DataFrame({
        "Year": years.astype(int),
        "gp_slug": slugs,
    })
    if keys.duplicated().any():
        raise ValueError("Gebruikte training bevat dubbele eventidentiteiten.")

    # Hash van geselecteerde raw features + target + eventidentiteit.
    # Rij- en kolomvolgorde zijn onderdeel van deze representatie.
    raw_columns = list(dict.fromkeys(
        ["Year", "gp_slug"] + list(features) + [target_col]
    ))
    raw = used.loc[:, raw_columns]
    raw_payload = json.dumps({
        "serialization": "pandas_table_json_v1",
        "dtypes": [str(dtype) for dtype in raw.dtypes],
        "data": raw.to_json(
            orient="split", index=False,
            date_format="iso", double_precision=15,
            force_ascii=True,
        ),
    }, sort_keys=True, separators=(",", ":"))

    # Extra hash van de uiteindelijke numerieke matrix en targets.
    # float.hex bewaart de exacte binaire floatwaarde.
    matrix_payload = json.dumps({
        "columns": list(map(str, X_train.columns)),
        "X": [
            [float(value).hex() for value in row]
            for row in X_train.to_numpy()
        ],
        "y": [float(value).hex() for value in y_train.to_numpy()],
    }, sort_keys=True, separators=(",", ":"))

    packages = {}
    for package in ["numpy", "pandas", "scikit-learn", "xgboost", "catboost"]:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not_installed"
    packages["python"] = platform.python_version()

    return {
        "ProvenanceVersion": "forward_run_provenance_v1",
        "FeatureSetVersion": "production_fixed32_categorical_v1",
        "FeatureCount": int(len(features)),
        "FeatureListJSON": json.dumps(list(features), ensure_ascii=True),
        "EncodedFeatureCount": int(X_train.shape[1]),
        "EncodedFeatureListJSON": json.dumps(
            list(map(str, X_train.columns)), ensure_ascii=True
        ),
        "PreprocessingVersion": "training_median_onehot_v1",
        "PreprocessingHash": source_hash(_prepare_final_prediction_matrix),
        "TrainingDataHash": digest(raw_payload),
        "TrainingMatrixHash": digest(matrix_payload),
        "TrainingHashMethod": "selected_raw_pandas_json_v1",
        "TrainingRowCount": int(len(used)),
        "TrainingMinYear": int(years.min()),
        "TrainingMaxYear": int(years.max()),
        "TrainingTarget": str(target_col),
        "ModelCodeHash": source_hash(run_three_final_models),
        "RuntimeVersionsJSON": json.dumps(packages, sort_keys=True),
    }

# ACTUAL_RACE_START_V1_SANDBOX_INTEGRATION_DRAFT
TIMING_METADATA_FIELDS = (
    "PredictionTimingPolicyVersion", "AnnouncedRaceStartUTC",
    "RaceStartSource", "RaceStartVerifiedAtUTC",
    "FeatureSnapshotAfterQualifyingUTC", "FeatureSnapshotTimestampUTC",
    "QualifyingCompletedAtUTC", "LatestInputAvailableAtUTC",
)

def _timing_now_utc():
    return pd.Timestamp.now(tz="UTC")

def _timing_iso(value):
    return pd.Timestamp(value).tz_convert("UTC").isoformat().replace("+00:00", "Z")

def _timing_record_hash(row):
    """Versioned CSV-roundtrip-safe payload hash, retaining an explicit field manifest.

    Historical hashes are never recalculated with this method.
    """
    fields = json.loads(str(row["TimingHashFieldsJSON"]))
    payload = {}
    for field in fields:
        value = row.get(field)
        if value is None or pd.isna(value) or str(value).strip().lower() in {"", "nan", "none", "null"}:
            payload[field] = None
        elif isinstance(value, (bool, np.bool_)):
            payload[field] = "1.000" if value else "0.000"
        else:
            try:
                numeric = float(value)
                if not np.isfinite(numeric):
                    raise ValueError("non-finite hash value")
                payload[field] = f"{numeric:.3f}"
            except (ValueError, TypeError):
                payload[field] = str(value)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _prediction_source_fingerprints():
    paths = (CONFIG.training_path, CONFIG.test_path, CONFIG.overview_path,
             CONFIG.registry_path, CONFIG.registry_eligible_mirror_path,
             CONFIG.prediction_ledger_path, CONFIG.model_log_path)
    return {Path(path): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if Path(path).exists() else None for path in paths}

def _assert_prediction_sources_unchanged(expected):
    for path, original in expected.items():
        current = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if current != original:
            raise RuntimeError(f"Prediction source changed during computation: {path.name}")

@_service_storage_transaction
def _assert_no_existing_final_prediction(year, gp_slug):
    event = build_race_event_id(year, gp_slug)
    for path in (CONFIG.prediction_ledger_path, CONFIG.model_log_path):
        if Path(path).exists():
            frame = read_csv_checked(path)
            if "race_event_id" in frame and frame["race_event_id"].astype(str).eq(event).any():
                if path == CONFIG.model_log_path and "RunType" in frame:
                    mask = frame["race_event_id"].astype(str).eq(event)
                    if not frame.loc[mask, "RunType"].astype(str).eq("final_pre_race_prediction").any():
                        continue
                raise ValueError(f"Immutable final prediction already exists for {event}; forecast or delay does not permit a rerun")

def validate_leakage_guard(df_test, df_overview, year, gp_slug):
    mt = get_target_mask(df_test, year, gp_slug)
    mo = get_target_mask(df_overview, year, gp_slug)
    if int(mt.sum()) != 1 or int(mo.sum()) != 1:
        return ["Expected exactly one target in test and overview"]
    rt, ro = df_test.loc[mt].iloc[0], df_overview.loc[mo].iloc[0]
    if str(rt.get("PredictionTimingPolicyVersion", "")).strip() != "actual_race_start_v1":
        return _legacy_validate_leakage_guard(df_test, df_overview, year, gp_slug)
    from .readiness import validate_event_timing, validate_feature_snapshot, validate_forward_prediction_timing
    gates = [validate_event_timing(rt, ro), validate_feature_snapshot(rt, ro),
             validate_forward_prediction_timing(rt, _timing_now_utc(),
                 qualifying_completed_at_utc=rt.get("QualifyingCompletedAtUTC"),
                 latest_input_available_at_utc=rt.get("LatestInputAvailableAtUTC"))]
    from .readiness import validate_frozen_input_integrity
    gates.append(validate_frozen_input_integrity(rt, ro))
    errors = [g.detail for g in gates if not g.ok]
    if not _readiness_actual_empty(ro.get("ActualFastestLap_sec")):
        errors.append("Overview actual already available")
    forbidden = [c for c in FINAL_PREDICTION_COLS if c in df_test]
    if forbidden:
        errors.append(f"df_test contains forbidden prediction columns: {forbidden}")
    return errors

def _append_frame_preserving_history(old, record):
    cols = list(dict.fromkeys(list(old.columns) + list(record)))
    return pd.concat([old.reindex(columns=cols), pd.DataFrame([record]).reindex(columns=cols)], ignore_index=True)

def _check_unchanged_history(frame, old):
    if len(frame) != len(old) + 1:
        raise ValueError("Append-only row count mismatch")
    def text(value):
        if value is None or pd.isna(value) or str(value).strip().lower() in {"", "nan", "none", "null"}: return None
        try:
            number = float(value)
            if np.isfinite(number): return f"{number:.12g}"
        except (TypeError, ValueError): pass
        return str(value)
    for column in old.columns:
        if [text(v) for v in frame.iloc[:len(old)][column]] != [text(v) for v in old[column]]:
            raise ValueError(f"Historical record changed in {column}")

def _commit_timed_final_prediction(year, gp_slug, df_test, overview_before, overview_new,
                                   ledger_row, log_row, source_fingerprints):
    from .readiness import validate_forward_prediction_timing
    event = build_race_event_id(year, gp_slug)
    rt = df_test.loc[require_one(df_test, year, gp_slug, "df_test")].iloc[0]
    _assert_no_existing_final_prediction(year, gp_slug)
    _assert_prediction_sources_unchanged(source_fingerprints)
    old_ledger = read_csv_checked(CONFIG.prediction_ledger_path) if CONFIG.prediction_ledger_path.exists() else pd.DataFrame()
    old_log = read_csv_checked(CONFIG.model_log_path) if CONFIG.model_log_path.exists() else pd.DataFrame()
    if old_ledger.empty and not len(old_ledger.columns):
        old_ledger = pd.DataFrame(columns=["RunID", "race_event_id", "PredictionRowHash"])
    # Metadata known at prediction time; no post-race values enter this record.
    for field in TIMING_METADATA_FIELDS:
        if field not in rt or pd.isna(rt[field]):
            raise ValueError(f"Required forward timing proof missing: {field}")
        ledger_row[field] = rt[field]
        log_row[field] = rt[field]
    ledger_row["PredictionHashMethod"] = "csv_payload_manifest_v1"
    ledger_row["EvaluationStatus"] = "awaiting_actual"
    ledger_row["TimingEvidenceStatus"] = "provisional_announced_start_pass"
    log_row["TimingEvidenceStatus"] = "provisional_announced_start_pass"
    ledger_row["TimingHashFieldsJSON"] = json.dumps(sorted(set(ledger_row) - {"PredictionRowHash", "TimingHashFieldsJSON"}))
    ledger_row["PredictionRowHash"] = _timing_record_hash(ledger_row)
    frames = {CONFIG.overview_path: overview_new,
              CONFIG.prediction_ledger_path: _append_frame_preserving_history(old_ledger, ledger_row),
              CONFIG.model_log_path: _append_frame_preserving_history(old_log, log_row)}

    def gate(timestamp):
        from .readiness import validate_frozen_input_integrity
        ro = overview_before.loc[require_one(overview_before, year, gp_slug, "df_overview")].iloc[0]
        integrity = validate_frozen_input_integrity(rt, ro)
        if not integrity.ok: raise ValueError(integrity.detail)
        result = validate_forward_prediction_timing(rt, timestamp,
            qualifying_completed_at_utc=rt.get("QualifyingCompletedAtUTC"),
            latest_input_available_at_utc=rt.get("LatestInputAvailableAtUTC"))
        if not result.ok:
            raise ValueError("Final storage timing blocked: " + result.detail)

    def finalize(normalized):
        _assert_prediction_sources_unchanged(source_fingerprints)
        current = _timing_now_utc()
        gate(current)
        timestamp = _timing_iso(current)
        run_id = f"{event}_{current.strftime('%Y%m%dT%H%M%S%fZ')}"
        ledger_row.update(RunID=run_id, PredictionTimestampUTC=timestamp)
        log_row.update(RunID=run_id, TimestampUTC=timestamp, PredictionTimestampUTC=timestamp)
        # Hash the rounded persisted payload, including timing and provenance.
        rounded = round_float_columns(pd.DataFrame([ledger_row]), decimals=3).iloc[0].to_dict()
        rounded["PredictionRowHash"] = _timing_record_hash(rounded)
        ledger_row.update(rounded)
        log_row["PredictionRowHash"] = ledger_row["PredictionRowHash"]
        normalized[CONFIG.prediction_ledger_path] = _append_frame_preserving_history(old_ledger, ledger_row)
        normalized[CONFIG.model_log_path] = _append_frame_preserving_history(old_log, log_row)

    def ledger_validator(frame):
        _check_unchanged_history(frame, old_ledger)
        record = frame.iloc[-1]
        if str(record["race_event_id"]) != event or frame["race_event_id"].astype(str).eq(event).sum() != 1:
            raise ValueError("Ledger event not unique")
        if str(record["PredictionRowHash"]) != _timing_record_hash(record):
            raise ValueError("Prediction hash mismatch")

    def log_validator(frame):
        _check_unchanged_history(frame, old_log)

    def overview_validator(frame):
        result = validate_overview_contract(frame)
        if not result.ok: raise ValueError(str(result.errors))
        mask = require_one(frame, year, gp_slug, "df_overview")
        row = frame.loc[mask].iloc[0]
        if not _readiness_actual_empty(row.get("ActualFastestLap_sec")): raise ValueError("Actual not empty")
        if str(row.get("RaceStatus")) != "prediction_ready": raise ValueError("Lifecycle status changed")
        for col in FINAL_PREDICTION_COLS:
            expected = round(float(ledger_row[col]), 3)
            if not np.isclose(float(row[col]), expected, atol=0.000001, rtol=0): raise ValueError(f"Prediction mismatch: {col}")
        protected = overview_before.loc[~get_target_mask(overview_before, year, gp_slug)]
        if len(protected):
            pd.testing.assert_frame_equal(frame.loc[protected.index, protected.columns].reset_index(drop=True), protected.reset_index(drop=True), check_dtype=False)

    def commit_guard():
        gate(_timing_now_utc())

    def after_commit():
        gate(_timing_now_utc())
        ledger = read_csv_checked(CONFIG.prediction_ledger_path).iloc[-1]
        log = read_csv_checked(CONFIG.model_log_path).iloc[-1]
        for field in (*TIMING_METADATA_FIELDS, "RunID", "PredictionTimestampUTC", "PredictionRowHash"):
            if str(ledger[field]) != str(log[field]):
                raise ValueError(f"Ledger/model-log metadata mismatch: {field}")
        # No external mutation of training/test/registry during finalization.
        _assert_prediction_sources_unchanged({path: value for path, value in source_fingerprints.items()
            if path not in {CONFIG.overview_path, CONFIG.prediction_ledger_path, CONFIG.model_log_path}})

    return safe_commit_many(frames, validators={CONFIG.overview_path: overview_validator,
        CONFIG.prediction_ledger_path: ledger_validator, CONFIG.model_log_path: log_validator},
        backup_dir=CONFIG.backup_dir, prepare_commit=finalize,
        commit_guard=commit_guard, after_commit=after_commit)
