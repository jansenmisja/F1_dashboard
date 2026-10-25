
"""
Pure pre-prediction readiness validation.

No Streamlit.
No CSV writes.
No RaceStatus mutation.

Phase 4C will use these gates for the official
prepare_prediction_ready() orchestration.
"""

from __future__ import annotations
from f1_pipeline.io import safe_commit_csv

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .config import CONFIG
from .io import (
    read_csv_checked,
    io_preflight,
)
from .schemas import (
    require_one_event,
    validate_training_contract,
    validate_test_contract,
    validate_overview_contract,
)
from .lifecycle import (
    get_event_state,
    transition_event_status,
)
from .registry import (
    get_registry_state,
    sync_event_registry,
)
from f1_pipeline.schemas import validate_row_created_contract
from f1_pipeline.config import CONFIG
from f1_pipeline.io import read_csv_checked, safe_commit_many, io_preflight
from f1_pipeline.lifecycle import transition_event_status


# ============================================================
# Canonical readiness fields
# ============================================================

QUALIFYING_NUMERIC = (
    "PoleTime_sec",
    "Q_airtemp_mean",
    "Q_airtemp_max",
    "Q_rain_sum",
    "Q_humidity_mean",
    "pole_c_index",
)

QUALIFYING_TEXT = (
    "pole_compound",
)

EVENT_TIMING_FIELDS = (
    "EventDateTimeUTC",
    "PredictionCutoffUTC",
    "PredictionCutoffPolicy",
    "R_window_start_utc",
    "R_window_end_utc",
    "RaceWindowDurationMinutes",
    "RaceWindowSource",
)

RACE_WEATHER_NUMERIC = (
    "R_airtemp_mean",
    "R_airtemp_max",
    "R_rain_sum",
    "R_humidity_mean",
)

RACE_WEATHER_TEXT = (
    "WeatherSourceMode",
)

SNAPSHOT_FIELDS = (
    "FeatureSnapshotAfterQualifyingUTC",
    "FeatureSnapshotTimestampUTC",
)

ACCEPTED_RECORDLAP_STATUSES = (
    "verified",
    "provisional_model_surrogate",
)


# ============================================================
# Results
# ============================================================

@dataclass
class GateResult:
    name: str
    ok: bool
    detail: str
    values: Dict[str, object] = field(
        default_factory=dict
    )


@dataclass
class ReadinessReport:
    year: int
    gp_slug: str
    race_status: Optional[str]

    ok: bool

    gates: Dict[str, GateResult]
    blockers: List[str]
    warnings: List[str]

    source_eligible: bool

    registry_present: bool
    registry_eligible: bool


# ============================================================
# Generic helpers
# ============================================================

def _present(
    value,
) -> bool:

    if pd.isna(value):
        return False

    if isinstance(value, str):
        return bool(
            value.strip()
        )

    return True


def _numeric(
    value,
) -> bool:

    try:

        x = float(value)

        return bool(
            np.isfinite(x)
        )

    except Exception:

        return False


def _same_value(
    left,
    right,
) -> bool:

    if (
        pd.isna(left)
        and pd.isna(right)
    ):
        return True

    if (
        _numeric(left)
        and _numeric(right)
    ):

        return bool(
            np.isclose(
                float(left),
                float(right),
                atol=0.001,
                rtol=0,
            )
        )

    return (
        str(left).strip()
        ==
        str(right).strip()
    )


def _require_synced_fields(
    test_row: pd.Series,
    overview_row: pd.Series,
    fields,
) -> List[str]:

    problems = []

    for col in fields:

        if (
            col not in test_row.index
            or col not in overview_row.index
        ):

            problems.append(
                f"{col}: column missing"
            )

            continue

        left = test_row[col]
        right = overview_row[col]

        if not _present(left):

            problems.append(
                f"{col}: test value missing"
            )

            continue

        if not _present(right):

            problems.append(
                f"{col}: overview value missing"
            )

            continue

        if not _same_value(
            left,
            right,
        ):

            problems.append(
                f"{col}: test/overview mismatch"
            )

    return problems


# ============================================================
# Individual gates
# ============================================================

def validate_qualifying(
    test_row: pd.Series,
    overview_row: pd.Series,
) -> GateResult:

    fields = (
        *QUALIFYING_NUMERIC,
        *QUALIFYING_TEXT,
    )

    problems = _require_synced_fields(
        test_row,
        overview_row,
        fields,
    )

    for col in QUALIFYING_NUMERIC:

        if (
            col in test_row.index
            and _present(
                test_row[col]
            )
            and not _numeric(
                test_row[col]
            )
        ):

            problems.append(
                f"{col}: not numeric"
            )

    return GateResult(
        name="qualifying",
        ok=not problems,
        detail=(
            "qualifying inputs synchronized"
            if not problems
            else "; ".join(problems)
        ),
    )


def _legacy_validate_event_timing(
    test_row: pd.Series,
    overview_row: pd.Series,
) -> GateResult:

    problems = _require_synced_fields(
        test_row,
        overview_row,
        EVENT_TIMING_FIELDS,
    )

    values = {}

    if not problems:

        event_dt = pd.to_datetime(
            test_row[
                "EventDateTimeUTC"
            ],
            utc=True,
            errors="coerce",
        )

        cutoff_dt = pd.to_datetime(
            test_row[
                "PredictionCutoffUTC"
            ],
            utc=True,
            errors="coerce",
        )

        window_start = pd.to_datetime(
            test_row[
                "R_window_start_utc"
            ],
            utc=True,
            errors="coerce",
        )

        window_end = pd.to_datetime(
            test_row[
                "R_window_end_utc"
            ],
            utc=True,
            errors="coerce",
        )

        if pd.isna(event_dt):
            problems.append(
                "EventDateTimeUTC invalid"
            )

        if pd.isna(cutoff_dt):
            problems.append(
                "PredictionCutoffUTC invalid"
            )

        if pd.isna(window_start):
            problems.append(
                "R_window_start_utc invalid"
            )

        if pd.isna(window_end):
            problems.append(
                "R_window_end_utc invalid"
            )

        if (
            not pd.isna(event_dt)
            and not pd.isna(cutoff_dt)
            and cutoff_dt >= event_dt
        ):

            problems.append(
                "cutoff must be before event start"
            )

        if (
            not pd.isna(window_start)
            and not pd.isna(window_end)
            and window_end <= window_start
        ):

            problems.append(
                "race window end must be after start"
            )

        values = {
            "event":
                str(event_dt),

            "cutoff":
                str(cutoff_dt),

            "window_start":
                str(window_start),

            "window_end":
                str(window_end),
        }

    return GateResult(
        name="event_timing",
        ok=not problems,
        detail=(
            "event timing valid"
            if not problems
            else "; ".join(problems)
        ),
        values=values,
    )


def validate_race_weather(
    test_row: pd.Series,
    overview_row: pd.Series,
) -> GateResult:

    fields = (
        *RACE_WEATHER_NUMERIC,
        *RACE_WEATHER_TEXT,
    )

    problems = _require_synced_fields(
        test_row,
        overview_row,
        fields,
    )

    for col in RACE_WEATHER_NUMERIC:

        if (
            col in test_row.index
            and _present(
                test_row[col]
            )
            and not _numeric(
                test_row[col]
            )
        ):

            problems.append(
                f"{col}: not numeric"
            )

    return GateResult(
        name="race_weather",
        ok=not problems,
        detail=(
            "race weather synchronized"
            if not problems
            else "; ".join(problems)
        ),
    )


def validate_recordlap_readiness(
    test_row: pd.Series,
    overview_row: pd.Series,
) -> GateResult:

    problems = _require_synced_fields(
        test_row,
        overview_row,
        (
            "RecordLap_sec",
            "RecordLapValidationStatus",
        ),
    )

    status = None

    if (
        "RecordLapValidationStatus"
        in test_row.index
    ):

        status = str(
            test_row[
                "RecordLapValidationStatus"
            ]
        ).strip()

    if (
        status
        not in ACCEPTED_RECORDLAP_STATUSES
    ):

        problems.append(
            "RecordLapValidationStatus not accepted"
        )

    if (
        "RecordLap_sec"
        in test_row.index
        and _present(
            test_row[
                "RecordLap_sec"
            ]
        )
        and (
            not _numeric(
                test_row[
                    "RecordLap_sec"
                ]
            )
            or float(
                test_row[
                    "RecordLap_sec"
                ]
            ) <= 0
        )
    ):

        problems.append(
            "RecordLap_sec invalid"
        )

    return GateResult(
        name="recordlap_readiness",
        ok=not problems,
        detail=(
            f"RecordLap readiness accepted: {status}"
            if not problems
            else "; ".join(problems)
        ),
        values={
            "status":
                status,

            "semantics":
                (
                    "verified"
                    if status == "verified"
                    else
                    "provisional_model_surrogate"
                    if status
                    == "provisional_model_surrogate"
                    else
                    "not_accepted"
                ),
        },
    )


def validate_identity_layout(
    test_row: pd.Series,
    overview_row: pd.Series,
) -> GateResult:
    """
    Current operational contract:

    Event identity:
    - exact Year/gp_slug row uniqueness is enforced outside this gate
    - Circuit must be present and synchronized

    Layout:
    - CircuitLayoutSlug must be present and synchronized

    Explicit EventIdentityVerified/LayoutVerified fields can be
    added later without changing the high-level API.
    """

    problems = []

    for col in (
        "Circuit",
        "CircuitLayoutSlug",
    ):

        if (
            col not in test_row.index
            or col not in overview_row.index
        ):

            problems.append(
                f"{col}: column missing"
            )

            continue

        if not _present(
            test_row[col]
        ):

            problems.append(
                f"{col}: missing"
            )

            continue

        if not _same_value(
            test_row[col],
            overview_row[col],
        ):

            problems.append(
                f"{col}: mismatch"
            )

    return GateResult(
        name="identity_layout",
        ok=not problems,
        detail=(
            "event identity accepted via unique Year/gp_slug; "
            "circuit layout accepted via CircuitLayoutSlug"
            if not problems
            else "; ".join(problems)
        ),
    )


def _legacy_validate_feature_snapshot(
    test_row: pd.Series,
    overview_row: pd.Series,
) -> GateResult:

    problems = _require_synced_fields(
        test_row,
        overview_row,
        SNAPSHOT_FIELDS,
    )

    snapshot_dt = None
    cutoff_dt = None

    if not problems:

        snapshot_dt = pd.to_datetime(
            test_row[
                "FeatureSnapshotTimestampUTC"
            ],
            utc=True,
            errors="coerce",
        )

        after_q_dt = pd.to_datetime(
            test_row[
                "FeatureSnapshotAfterQualifyingUTC"
            ],
            utc=True,
            errors="coerce",
        )

        cutoff_dt = pd.to_datetime(
            test_row[
                "PredictionCutoffUTC"
            ],
            utc=True,
            errors="coerce",
        )

        if pd.isna(snapshot_dt):

            problems.append(
                "snapshot timestamp invalid"
            )

        if pd.isna(after_q_dt):

            problems.append(
                "after-qualifying timestamp invalid"
            )

        if pd.isna(cutoff_dt):

            problems.append(
                "cutoff invalid"
            )

        if (
            not pd.isna(snapshot_dt)
            and not pd.isna(after_q_dt)
            and snapshot_dt != after_q_dt
        ):

            problems.append(
                "snapshot timestamps differ"
            )

        if (
            not pd.isna(snapshot_dt)
            and not pd.isna(cutoff_dt)
            and snapshot_dt > cutoff_dt
        ):

            problems.append(
                "snapshot occurs after prediction cutoff"
            )

    return GateResult(
        name="feature_snapshot",
        ok=not problems,
        detail=(
            "feature snapshot valid and before cutoff"
            if not problems
            else "; ".join(problems)
        ),
    )


def validate_race_context_plausibility(
    test_row: pd.Series,
    overview_row: pd.Series,
) -> GateResult:

    problems = []

    if (
        "Regime2026_Flag"
        not in test_row.index
        or "Regime2026_Flag"
        not in overview_row.index
    ):

        problems.append(
            "Regime2026_Flag missing"
        )

    else:

        left = test_row[
            "Regime2026_Flag"
        ]

        right = overview_row[
            "Regime2026_Flag"
        ]

        if not _present(left):

            problems.append(
                "Regime2026_Flag missing value"
            )

        elif not _same_value(
            left,
            right,
        ):

            problems.append(
                "Regime2026_Flag mismatch"
            )

    return GateResult(
        name="race_context_plausibility",
        ok=not problems,
        detail=(
            "race context plausibility inputs present"
            if not problems
            else "; ".join(problems)
        ),
    )


def validate_leakage_guard(
    test_row: pd.Series,
    overview_row: pd.Series,
) -> GateResult:

    problems = []

    # df_test contract itself protects prediction/actual/KPI leakage.
    # Here we additionally require no actual in overview before prediction.
    if (
        "ActualFastestLap_sec"
        in overview_row.index
        and _present(
            overview_row[
                "ActualFastestLap_sec"
            ]
        )
    ):

        problems.append(
            "ActualFastestLap_sec already populated in overview"
        )

    return GateResult(
        name="leakage_guard",
        ok=not problems,
        detail=(
            "no post-race leakage detected"
            if not problems
            else "; ".join(problems)
        ),
    )


# ============================================================
# Official read-only readiness report
# ============================================================

def _build_prediction_readiness_report_raw(
    test_row,
    overview_row,
    *,
    year: int,
    gp_slug: str,
    race_status=None,
    registry_present: bool = False,
    registry_eligible: bool = False,
) -> ReadinessReport:

    year = int(year)
    gp_slug = str(gp_slug).strip()

    blockers = []
    warnings = []

    gates = {
        "qualifying":
            validate_qualifying(
                test_row,
                overview_row,
            ),

        "event_timing":
            validate_event_timing(
                test_row,
                overview_row,
            ),

        "race_weather":
            validate_race_weather(
                test_row,
                overview_row,
            ),

        "recordlap_readiness":
            validate_recordlap_readiness(
                test_row,
                overview_row,
            ),

        "identity_layout":
            validate_identity_layout(
                test_row,
                overview_row,
            ),

        "feature_snapshot":
            validate_feature_snapshot(
                test_row,
                overview_row,
            ),

        "race_context_plausibility":
            validate_race_context_plausibility(
                test_row,
                overview_row,
            ),

        "leakage_guard":
            validate_leakage_guard(
                test_row,
                overview_row,
            ),
    }

    if _timing_policy(test_row) == NEW_TIMING_POLICY:
        gates["frozen_input_integrity"] = validate_frozen_input_integrity(test_row, overview_row)
        gates["forward_timing"] = validate_forward_prediction_timing(test_row, _readiness_now_utc(),
            qualifying_completed_at_utc=test_row.get("QualifyingCompletedAtUTC"),
            latest_input_available_at_utc=test_row.get("LatestInputAvailableAtUTC"))

    for name, gate in gates.items():

        if not gate.ok:

            blockers.append(
                f"{name}: {gate.detail}"
            )

    source_eligible = all(
        gate.ok
        for gate in gates.values()
    )

    return ReadinessReport(
        year=year,
        gp_slug=gp_slug,
        race_status=race_status,
        ok=(
            source_eligible
            and not blockers
        ),
        gates=gates,
        blockers=blockers,
        warnings=warnings,
        source_eligible=source_eligible,
        registry_present=bool(
            registry_present
        ),
        registry_eligible=bool(
            registry_eligible
        ),
    )

from .storage_transactions import transactional as _service_storage_transaction

@_service_storage_transaction
def build_prediction_readiness_report(
    year: int,
    gp_slug: str,
) -> ReadinessReport:

    year = int(year)
    gp_slug = str(
        gp_slug
    ).strip()

    blockers = []
    warnings = []

    # --------------------------------------------------------
    # Infrastructure + global contracts
    # --------------------------------------------------------
    preflight = io_preflight()

    if not preflight["ok"]:

        blockers.append(
            "I/O preflight failed"
        )

    df_training = read_csv_checked(
        CONFIG.training_path
    )

    df_test = read_csv_checked(
        CONFIG.test_path
    )

    df_overview = read_csv_checked(
        CONFIG.overview_path
    )

    training_contract = (
        validate_training_contract(
            df_training
        )
    )

    test_contract = (
        validate_test_contract(
            df_test,
            strict_schema=True,
        )
    )

    overview_contract = (
        validate_overview_contract(
            df_overview
        )
    )

    if not training_contract.ok:

        blockers.extend(
            training_contract.errors
        )

    if not test_contract.ok:

        blockers.extend(
            test_contract.errors
        )

    if not overview_contract.ok:

        blockers.extend(
            overview_contract.errors
        )

    test_mask = require_one_event(
        df_test,
        year,
        gp_slug,
        dataset_name="df_test",
    )

    overview_mask = require_one_event(
        df_overview,
        year,
        gp_slug,
        dataset_name="df_overview",
    )

    test_row = (
        df_test.loc[
            test_mask
        ]
        .iloc[0]
    )

    overview_row = (
        df_overview.loc[
            overview_mask
        ]
        .iloc[0]
    )

    lifecycle = get_event_state(
        year,
        gp_slug,
    )

    registry = get_registry_state(
        year,
        gp_slug,
    )

    raw = _build_prediction_readiness_report_raw(
        test_row,
        overview_row,
        year=year,
        gp_slug=gp_slug,
        race_status=lifecycle.race_status,
        registry_present=registry.registry_present,
        registry_eligible=registry.registry_eligible,
    )

    combined_blockers = (
        list(blockers)
        + list(raw.blockers)
    )

    return ReadinessReport(
        year=raw.year,
        gp_slug=raw.gp_slug,
        race_status=raw.race_status,
        ok=(
            raw.source_eligible
            and not combined_blockers
        ),
        gates=raw.gates,
        blockers=combined_blockers,
        warnings=list(raw.warnings),
        source_eligible=raw.source_eligible,
        registry_present=raw.registry_present,
        registry_eligible=raw.registry_eligible,
    )


def prediction_readiness_ok(
    year: int,
    gp_slug: str,
) -> bool:

    return bool(
        build_prediction_readiness_report(
            year,
            gp_slug,
        ).ok
    )
# ============================================================
# PHASE 4C — OFFICIAL PREPARE ENTRYPOINT
# ============================================================
# ============================================================
# PHASE 4C — OFFICIAL PREPARE ENTRYPOINT
# ============================================================

@dataclass
class PreparePredictionReadyResult:
    ok: bool
    year: int
    gp_slug: str

    status_before: str
    status_after: str

    source_eligible: bool
    registry_present: bool
    registry_eligible: bool

    message: str


@_service_storage_transaction
def prepare_prediction_ready(
    year: int,
    gp_slug: str,
) -> PreparePredictionReadyResult:
    """
    Official high-level promotion into prediction_ready.

    Required current lifecycle state:
        race_context_plausibility_ready

    Flow:
        1. central readiness report
        2. all substantive gates PASS
        3. idempotent authoritative registry sync
        4. registry readback validation
        5. central lifecycle transition
        6. final readback validation

    No caller should manually modify RaceStatus or repair
    registry files before calling this function.
    """

    year = int(year)
    gp_slug = str(
        gp_slug
    ).strip()

    # --------------------------------------------------------
    # 1. CURRENT STATE
    # --------------------------------------------------------
    state_before = get_event_state(
        year,
        gp_slug,
    )

    if not state_before.lifecycle_valid:

        raise ValueError(
            "Event lifecycle is invalid: "
            + " | ".join(
                state_before.blockers
            )
        )

    expected_status = (
        "race_context_plausibility_ready"
    )

    if (
        state_before.race_status
        != expected_status
    ):

        raise ValueError(
            "prepare_prediction_ready requires "
            f"RaceStatus={expected_status}; "
            f"current={state_before.race_status}"
        )

    # --------------------------------------------------------
    # 2. OFFICIAL READINESS REPORT
    # --------------------------------------------------------
    report = (
        build_prediction_readiness_report(
            year,
            gp_slug,
        )
    )

    if not report.ok:

        blocker_text = (
            " | ".join(
                report.blockers
            )
            if report.blockers
            else
            "unknown readiness blocker"
        )

        raise ValueError(
            "Prediction readiness failed: "
            + blocker_text
        )

    if not report.source_eligible:

        raise ValueError(
            "Source eligibility is False."
        )

    # --------------------------------------------------------
    # 3. IDEMPOTENT REGISTRY SYNC
    #
    # source_eligible comes from substantive readiness,
    # not from prior physical registry presence.
    # --------------------------------------------------------
    registry_sync = (
        sync_event_registry(
            year,
            gp_slug,
            source_eligible=True,
        )
    )

    if not (
        registry_sync.source_eligible
        and registry_sync.registry_present
        and registry_sync.registry_eligible
        and registry_sync.authoritative_count == 1
    ):

        raise RuntimeError(
            "Registry sync did not produce an eligible "
            "authoritative event row."
        )

    # --------------------------------------------------------
    # 4. REGISTRY READBACK
    # --------------------------------------------------------
    registry_readback = (
        get_registry_state(
            year,
            gp_slug,
        )
    )

    if not (
        registry_readback.source_eligible
        and registry_readback.registry_present
        and registry_readback.registry_eligible
        and registry_readback.authoritative_count == 1
    ):

        raise RuntimeError(
            "Registry readback validation failed."
        )

    # --------------------------------------------------------
    # 5. RE-RUN SUBSTANTIVE READINESS
    #
    # Prevent promotion if source files changed between
    # readiness and registry synchronization.
    # --------------------------------------------------------
    second_report = (
        build_prediction_readiness_report(
            year,
            gp_slug,
        )
    )

    if not second_report.ok:

        raise RuntimeError(
            "Readiness changed after registry sync: "
            + " | ".join(
                second_report.blockers
            )
        )

    # --------------------------------------------------------
    # 5b. CANONICAL READINESS AUDIT COMMENT
    # --------------------------------------------------------
    _append_prediction_ready_comment(
        year,
        gp_slug,
    )

    # --------------------------------------------------------
    # 6. CENTRAL LIFECYCLE TRANSITION
    # --------------------------------------------------------
    transition = (
        transition_event_status(
            year=year,
            gp_slug=gp_slug,
            expected_from=(
                "race_context_plausibility_ready"
            ),
            to_status=(
                "prediction_ready"
            ),
        )
    )

    if not transition.ok:

        raise RuntimeError(
            "Lifecycle transition failed."
        )

    # --------------------------------------------------------
    # 7. FINAL STATE READBACK
    # --------------------------------------------------------
    state_after = get_event_state(
        year,
        gp_slug,
    )

    registry_after = (
        get_registry_state(
            year,
            gp_slug,
        )
    )

    if (
        state_after.race_status
        != "prediction_ready"
    ):

        raise RuntimeError(
            "Final RaceStatus readback is not "
            "prediction_ready."
        )

    if not (
        registry_after.registry_present
        and registry_after.registry_eligible
    ):

        raise RuntimeError(
            "Final registry state is not eligible."
        )

    return PreparePredictionReadyResult(
        ok=True,

        year=year,
        gp_slug=gp_slug,

        status_before=(
            state_before.race_status
        ),

        status_after=(
            state_after.race_status
        ),

        source_eligible=(
            registry_after.source_eligible
        ),

        registry_present=(
            registry_after.registry_present
        ),

        registry_eligible=(
            registry_after.registry_eligible
        ),

        message=(
            "Event successfully promoted to "
            "prediction_ready"
        ),
    )


def validate_race_weather_audit_rows(
    test_row: pd.Series,
    overview_row: pd.Series,
    *,
    require_status: bool = False,
) -> list[str]:
    """
    Detailed race-weather audit.

    Preserves the stricter legacy audit semantics:
    completeness, synchronization, numeric checks,
    plausibility, weather source, race-window timing,
    and optional race_weather_ready status validation.
    """

    errors: list[str] = []

    audit_cols = [
        "R_airtemp_mean",
        "R_airtemp_max",
        "R_rain_sum",
        "R_humidity_mean",
        "WeatherSourceMode",
        "R_window_start_utc",
        "R_window_end_utc",
    ]

    # --------------------------------------------------------
    # Completeness
    # --------------------------------------------------------
    for dataset_name, row in [
        ("df_test", test_row),
        ("df_overview", overview_row),
    ]:

        missing = []

        for col in audit_cols:

            if col not in row.index:
                missing.append(col)
                continue

            value = row.get(col)

            if pd.isna(value):
                missing.append(col)

            elif (
                isinstance(value, str)
                and not value.strip()
            ):
                missing.append(col)

        if missing:
            errors.append(
                f"{dataset_name}: ontbrekende R-weather velden: {missing}"
            )

    if errors:
        return errors

    # --------------------------------------------------------
    # Numeric synchronization
    # --------------------------------------------------------
    numeric_cols = [
        "R_airtemp_mean",
        "R_airtemp_max",
        "R_rain_sum",
        "R_humidity_mean",
    ]

    for col in numeric_cols:

        vt = pd.to_numeric(
            test_row.get(col),
            errors="coerce",
        )

        vo = pd.to_numeric(
            overview_row.get(col),
            errors="coerce",
        )

        if pd.isna(vt):
            errors.append(
                f"df_test: {col} is niet numeriek."
            )
            continue

        if pd.isna(vo):
            errors.append(
                f"df_overview: {col} is niet numeriek."
            )
            continue

        if not np.isclose(
            float(vt),
            float(vo),
            atol=0.001,
        ):
            errors.append(
                f"{col}: mismatch tussen df_test en df_overview."
            )

    # --------------------------------------------------------
    # Plausibility
    # --------------------------------------------------------
    rain = pd.to_numeric(
        test_row.get("R_rain_sum"),
        errors="coerce",
    )

    humidity = pd.to_numeric(
        test_row.get("R_humidity_mean"),
        errors="coerce",
    )

    air_mean = pd.to_numeric(
        test_row.get("R_airtemp_mean"),
        errors="coerce",
    )

    air_max = pd.to_numeric(
        test_row.get("R_airtemp_max"),
        errors="coerce",
    )

    if (
        pd.notna(rain)
        and rain < 0
    ):
        errors.append(
            "R_rain_sum mag niet negatief zijn."
        )

    if (
        pd.notna(humidity)
        and not (
            0
            <= humidity
            <= 100
        )
    ):
        errors.append(
            "R_humidity_mean moet tussen 0 en 100 liggen."
        )

    if (
        pd.notna(air_mean)
        and pd.notna(air_max)
        and air_max < air_mean
    ):
        errors.append(
            "R_airtemp_max mag niet lager zijn dan R_airtemp_mean."
        )

    # --------------------------------------------------------
    # WeatherSourceMode synchronization
    # --------------------------------------------------------
    source_test = str(
        test_row.get(
            "WeatherSourceMode"
        )
    ).strip()

    source_overview = str(
        overview_row.get(
            "WeatherSourceMode"
        )
    ).strip()

    if (
        source_test
        != source_overview
    ):
        errors.append(
            "WeatherSourceMode mismatch tussen df_test en df_overview."
        )

    # --------------------------------------------------------
    # Race-window timestamps
    # --------------------------------------------------------
    for col in [
        "R_window_start_utc",
        "R_window_end_utc",
    ]:

        ts_test = pd.to_datetime(
            test_row.get(col),
            utc=True,
            errors="coerce",
        )

        ts_overview = pd.to_datetime(
            overview_row.get(col),
            utc=True,
            errors="coerce",
        )

        if pd.isna(
            ts_test
        ):
            errors.append(
                f"df_test: {col} ongeldig."
            )
            continue

        if pd.isna(
            ts_overview
        ):
            errors.append(
                f"df_overview: {col} ongeldig."
            )
            continue

        if (
            ts_test
            != ts_overview
        ):
            errors.append(
                f"{col}: mismatch tussen df_test en df_overview."
            )

    start_ts = pd.to_datetime(
        test_row.get(
            "R_window_start_utc"
        ),
        utc=True,
        errors="coerce",
    )

    end_ts = pd.to_datetime(
        test_row.get(
            "R_window_end_utc"
        ),
        utc=True,
        errors="coerce",
    )

    if (
        pd.notna(
            start_ts
        )
        and pd.notna(
            end_ts
        )
        and end_ts
        <= start_ts
    ):
        errors.append(
            "R_window_end_utc moet na R_window_start_utc liggen."
        )

    # --------------------------------------------------------
    # Optional lifecycle status
    # --------------------------------------------------------
    if require_status:

        status_test = str(
            test_row.get(
                "RaceStatus"
            )
        ).strip()

        status_overview = str(
            overview_row.get(
                "RaceStatus"
            )
        ).strip()

        if (
            status_test
            != "race_weather_ready"
        ):
            errors.append(
                "df_test RaceStatus is niet race_weather_ready."
            )

        if (
            status_overview
            != "race_weather_ready"
        ):
            errors.append(
                "df_overview RaceStatus is niet race_weather_ready."
            )

    return list(
        dict.fromkeys(
            errors
        )
    )


def validate_circuit_regime_rows(
    df_test: pd.DataFrame,
    df_overview: pd.DataFrame,
    year: int,
    gp_slug: str,
) -> list[str]:
    """
    Pure circuit/regime readiness validation.

    Preserves the legacy validate_circuit_regime semantics,
    but delegates row-created dataset integrity to the
    canonical schemas layer.
    """

    errors = []

    row_contract = (
        validate_row_created_contract(
            df_test,
            df_overview,
            year,
            gp_slug,
        )
    )

    if not row_contract.ok:

        return list(
            dict.fromkeys(
                row_contract.errors
            )
        )

    mt = require_one_event(
        df_test,
        year,
        gp_slug,
        "df_test",
    )

    mo = require_one_event(
        df_overview,
        year,
        gp_slug,
        "df_overview",
    )

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

    required = [
        "YearOffset",
        "Length_km",
        "Turns",
        "Turns/Length_km",
        "TurnDensity",
        "avg_speed_kmh",
        "v_proxy_kmh",
        "RecordLap_sec",
        "Latitude",
        "Longitude",
        "Turns/length rank",
        "avg_speed rank",
        "IsSprintWeekend",
    ]

    if int(
        year
    ) >= 2026:

        required += [
            "CornerFraction_est",
            "Regime2026_Flag",
            "Regime2026_AeroComponent_sec",
            "Regime2026_PUComponent_sec",
            "Regime2026_MassComponent_sec",
            "Regime2026_TyreComponent_sec",
            "Regime2026_PriorLapDelta_sec",
        ]

    # --------------------------------------------------------
    # Required structural fields + RecordLap
    # --------------------------------------------------------
    for dataset_name, row in [
        ("df_test", rt),
        ("df_overview", ro),
    ]:

        missing = [
            col
            for col in required
            if (
                col not in row.index
                or pd.isna(
                    row[col]
                )
            )
        ]

        if missing:

            errors.append(
                f"{dataset_name}: "
                "ontbrekende structurele velden: "
                f"{missing}"
            )

        record_lap = pd.to_numeric(
            row.get(
                "RecordLap_sec"
            ),
            errors="coerce",
        )

        if (
            pd.isna(
                record_lap
            )
            or float(
                record_lap
            ) <= 0
        ):

            errors.append(
                f"{dataset_name}: "
                "RecordLap_sec moet groter dan 0 zijn."
            )

    # Preserve legacy short-circuit semantics:
    # only run detailed sync/regime checks if no errors yet.
    if errors:

        return list(
            dict.fromkeys(
                errors
            )
        )

    # --------------------------------------------------------
    # 2026 regime component identity
    # --------------------------------------------------------
    if int(
        year
    ) >= 2026:

        prior = (
            float(
                rt[
                    "Regime2026_AeroComponent_sec"
                ]
            )
            + float(
                rt[
                    "Regime2026_PUComponent_sec"
                ]
            )
            + float(
                rt[
                    "Regime2026_MassComponent_sec"
                ]
            )
            + float(
                rt[
                    "Regime2026_TyreComponent_sec"
                ]
            )
        )

        if not np.isclose(
            prior,
            float(
                rt[
                    "Regime2026_PriorLapDelta_sec"
                ]
            ),
            atol=0.001,
        ):

            errors.append(
                "Regime2026 prior is niet gelijk "
                "aan som componenten."
            )

    # --------------------------------------------------------
    # df_test ↔ df_overview synchronization
    # --------------------------------------------------------
    for col in required:

        if (
            col in rt.index
            and col in ro.index
        ):

            a = rt[col]
            b = ro[col]

            if (
                pd.isna(a)
                and pd.isna(b)
            ):
                continue

            try:

                same = np.isclose(
                    float(a),
                    float(b),
                    atol=0.001,
                )

            except Exception:

                same = (
                    str(a)
                    == str(b)
                )

            if not same:

                errors.append(
                    "Mismatch df_test/df_overview "
                    f"voor {col}."
                )

    return list(
        dict.fromkeys(
            errors
        )
    )


def validate_qualifying_ready_rows(
    df_test: pd.DataFrame,
    df_overview: pd.DataFrame,
    year: int,
    gp_slug: str,
) -> list[str]:
    """
    Pure qualifying-ready validation.

    Preserves legacy validate_qualifying semantics:
    - circuit/regime readiness first
    - required qualifying fields complete in both datasets
    - PoleMinusRecord_sec required
    - SprintFastestLap_sec required when IsSprintWeekend == 1
    - canonical event timing must pass

    Deliberately does not add stronger synchronization rules
    beyond the previous legacy contract.
    """

    errors = []

    # --------------------------------------------------------
    # 1. Circuit/regime prerequisite
    # --------------------------------------------------------
    errors.extend(
        validate_circuit_regime_rows(
            df_test,
            df_overview,
            year,
            gp_slug,
        )
    )

    if errors:

        return list(
            dict.fromkeys(
                errors
            )
        )

    # --------------------------------------------------------
    # 2. Target rows
    # --------------------------------------------------------
    mt = require_one_event(
        df_test,
        year,
        gp_slug,
        "df_test",
    )

    mo = require_one_event(
        df_overview,
        year,
        gp_slug,
        "df_overview",
    )

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

    # Legacy QUALIFYING_COLS reconstructed from canonical fields
    # plus PoleMinusRecord_sec, which is legacy-required but is
    # not part of QUALIFYING_NUMERIC.
    required_base = [
        *QUALIFYING_NUMERIC,
        "PoleMinusRecord_sec",
        *QUALIFYING_TEXT,
    ]

    # --------------------------------------------------------
    # 3. Completeness in both datasets
    # --------------------------------------------------------
    for dataset_name, row in [
        ("df_test", rt),
        ("df_overview", ro),
    ]:

        required = list(
            required_base
        )

        sprint_flag = pd.to_numeric(
            row.get(
                "IsSprintWeekend"
            ),
            errors="coerce",
        )

        if (
            pd.notna(
                sprint_flag
            )
            and int(
                sprint_flag
            ) == 1
        ):

            required.append(
                "SprintFastestLap_sec"
            )

        missing = [
            col
            for col in required
            if (
                col not in row.index
                or pd.isna(
                    row[col]
                )
            )
        ]

        if missing:

            errors.append(
                f"{dataset_name}: "
                f"ontbrekende qualifyingvelden: {missing}"
            )

    # --------------------------------------------------------
    # 4. Canonical event timing
    # --------------------------------------------------------
    timing_gate = (
        validate_event_timing(
            rt,
            ro,
        )
    )

    if not timing_gate.ok:

        errors.append(
            timing_gate.detail
        )

    return list(
        dict.fromkeys(
            errors
        )
    )


def validate_race_weather_ready_rows(
    df_test: pd.DataFrame,
    df_overview: pd.DataFrame,
    year: int,
    gp_slug: str,
) -> list[str]:
    """
    Pure race-weather-ready validation.

    Preserves legacy validate_race_weather semantics:
    - qualifying-ready prerequisite
    - four required race-weather numeric fields
    - numeric validity
    - rain >= 0
    - humidity in [0, 100]
    - R_airtemp_max >= R_airtemp_mean
    - df_test / df_overview synchronization

    Does not add WeatherSourceMode or broader weather-audit
    requirements that were not part of the legacy function.
    """

    errors = []

    # --------------------------------------------------------
    # 1. Qualifying prerequisite
    # --------------------------------------------------------
    errors.extend(
        validate_qualifying_ready_rows(
            df_test,
            df_overview,
            year,
            gp_slug,
        )
    )

    if errors:

        return list(
            dict.fromkeys(
                errors
            )
        )

    # --------------------------------------------------------
    # 2. Target rows
    # --------------------------------------------------------
    mt = require_one_event(
        df_test,
        year,
        gp_slug,
        "df_test",
    )

    mo = require_one_event(
        df_overview,
        year,
        gp_slug,
        "df_overview",
    )

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

    weather_fields = [
        "R_airtemp_mean",
        "R_airtemp_max",
        "R_rain_sum",
        "R_humidity_mean",
    ]

    # --------------------------------------------------------
    # 3. Required + numeric checks
    # --------------------------------------------------------
    numeric_values = {}

    for dataset_name, row in [
        ("df_test", rt),
        ("df_overview", ro),
    ]:

        numeric_values[
            dataset_name
        ] = {}

        for col in weather_fields:

            if (
                col not in row.index
                or pd.isna(
                    row[col]
                )
            ):

                errors.append(
                    f"{dataset_name}: "
                    f"ontbrekend race-weather veld: {col}"
                )

                continue

            value = pd.to_numeric(
                row[col],
                errors="coerce",
            )

            if pd.isna(
                value
            ):

                errors.append(
                    f"{dataset_name}: "
                    f"{col} is niet numeriek."
                )

                continue

            numeric_values[
                dataset_name
            ][
                col
            ] = float(
                value
            )

    if errors:

        return list(
            dict.fromkeys(
                errors
            )
        )

    # --------------------------------------------------------
    # 4. Plausibility per dataset
    # --------------------------------------------------------
    for dataset_name in (
        "df_test",
        "df_overview",
    ):

        values = numeric_values[
            dataset_name
        ]

        if (
            values[
                "R_rain_sum"
            ]
            < 0
        ):

            errors.append(
                f"{dataset_name}: "
                "R_rain_sum mag niet negatief zijn."
            )

        humidity = values[
            "R_humidity_mean"
        ]

        if not (
            0
            <= humidity
            <= 100
        ):

            errors.append(
                f"{dataset_name}: "
                "R_humidity_mean moet tussen 0 en 100 liggen."
            )

        if (
            values[
                "R_airtemp_max"
            ]
            < values[
                "R_airtemp_mean"
            ]
        ):

            errors.append(
                f"{dataset_name}: "
                "R_airtemp_max mag niet lager zijn "
                "dan R_airtemp_mean."
            )

    # --------------------------------------------------------
    # 5. df_test ↔ df_overview sync
    # --------------------------------------------------------
    for col in weather_fields:

        a = numeric_values[
            "df_test"
        ][
            col
        ]

        b = numeric_values[
            "df_overview"
        ][
            col
        ]

        if not np.isclose(
            a,
            b,
            atol=0.001,
        ):

            errors.append(
                "Mismatch df_test/df_overview "
                f"voor {col}."
            )

    return list(
        dict.fromkeys(
            errors
        )
    )


def validate_prediction_ready_rows(
    df_test: pd.DataFrame,
    df_overview: pd.DataFrame,
    year: int,
    gp_slug: str,
) -> list[str]:
    """
    Pure legacy-compatible prediction-ready validation.

    Preserves the remaining legacy semantics after the
    circuit/qualifying/weather refactor.

    No I/O, no Streamlit, no lifecycle mutation.
    """

    errors = []

    # --------------------------------------------------------
    # 1. All earlier readiness layers
    # --------------------------------------------------------
    errors.extend(
        validate_race_weather_ready_rows(
            df_test,
            df_overview,
            year,
            gp_slug,
        )
    )

    if errors:

        return list(
            dict.fromkeys(
                errors
            )
        )

    mt = require_one_event(
        df_test,
        year,
        gp_slug,
        "df_test",
    )

    mo = require_one_event(
        df_overview,
        year,
        gp_slug,
        "df_overview",
    )

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
    # 2. Structural prediction inputs
    # --------------------------------------------------------
    required_structural = [
        "Year",
        "YearOffset",
        "gp_slug",
        "Circuit",
        "Elevation_m",
        "Length_km",
        "Turns",
        "Altitude_m",
        "Altitude_bucket",
        "Downforce_profile",
        "Track_type",
        "Direction",
        "Brake_intensity",
        "Turns/Length_km",
        "TurnDensity",
        "avg_speed_kmh",
        "v_proxy_kmh",
        "RecordLap_sec",
        "IsSprintWeekend",
        "Turns/length rank",
        "avg_speed rank",
        "Latitude",
        "Longitude",
    ]

    if int(
        year
    ) >= 2026:

        required_structural += [
            "CornerFraction_est",
            "Regime2026_Flag",
            "Regime2026_AeroComponent_sec",
            "Regime2026_PUComponent_sec",
            "Regime2026_MassComponent_sec",
            "Regime2026_TyreComponent_sec",
            "Regime2026_PriorLapDelta_sec",
        ]

    # --------------------------------------------------------
    # 3. Qualifying + race-weather inputs
    # --------------------------------------------------------
    required_session = [
        "Q_airtemp_mean",
        "Q_airtemp_max",
        "Q_rain_sum",
        "Q_humidity_mean",
        "PoleTime_sec",
        "PoleMinusRecord_sec",
        "pole_compound",
        "pole_c_index",
        "R_airtemp_mean",
        "R_airtemp_max",
        "R_rain_sum",
        "R_humidity_mean",
    ]

    # --------------------------------------------------------
    # 4. Sprint conditional
    # --------------------------------------------------------
    sprint_flag = pd.to_numeric(
        rt.get(
            "IsSprintWeekend"
        ),
        errors="coerce",
    )

    if pd.isna(
        sprint_flag
    ):

        errors.append(
            "IsSprintWeekend ontbreekt."
        )

    elif int(
        sprint_flag
    ) not in [
        0,
        1,
    ]:

        errors.append(
            "IsSprintWeekend moet 0 of 1 zijn."
        )

    elif int(
        sprint_flag
    ) == 1:

        required_session.append(
            "SprintFastestLap_sec"
        )

    required_all = (
        required_structural
        + required_session
    )

    # --------------------------------------------------------
    # 5. Completeness in both datasets
    # --------------------------------------------------------
    for dataset_name, row in [
        (
            "df_test",
            rt,
        ),
        (
            "df_overview",
            ro,
        ),
    ]:

        missing = [
            col
            for col in required_all
            if (
                col not in row.index
                or pd.isna(
                    row[col]
                )
            )
        ]

        if missing:

            errors.append(
                f"{dataset_name}: "
                "verplichte prediction-input ontbreekt: "
                f"{missing}"
            )

    if errors:

        return list(
            dict.fromkeys(
                errors
            )
        )

    # --------------------------------------------------------
    # 6. Full df_test ↔ df_overview synchronization
    # --------------------------------------------------------
    for col in required_all:

        a = rt[col]
        b = ro[col]

        if (
            pd.isna(a)
            and pd.isna(b)
        ):
            continue

        if (
            pd.isna(a)
            != pd.isna(b)
        ):

            errors.append(
                f"{col}: missing-status verschilt tussen "
                "df_test en df_overview."
            )

            continue

        try:

            same = np.isclose(
                float(a),
                float(b),
                atol=0.001,
            )

        except (
            TypeError,
            ValueError,
        ):

            same = (
                str(a)
                .strip()
                .lower()
                ==
                str(b)
                .strip()
                .lower()
            )

        if not same:

            errors.append(
                f"{col}: mismatch tussen df_test "
                "en df_overview."
            )

    # --------------------------------------------------------
    # 7. Turns / Length formula
    # --------------------------------------------------------
    turns = pd.to_numeric(
        rt[
            "Turns"
        ],
        errors="coerce",
    )

    length_km = pd.to_numeric(
        rt[
            "Length_km"
        ],
        errors="coerce",
    )

    density = pd.to_numeric(
        rt[
            "Turns/Length_km"
        ],
        errors="coerce",
    )

    if (
        pd.notna(
            turns
        )
        and pd.notna(
            length_km
        )
        and length_km > 0
    ):

        expected_density = (
            turns
            / length_km
        )

        if not np.isclose(
            expected_density,
            density,
            atol=0.001,
        ):

            errors.append(
                "Turns/Length_km klopt niet met "
                "Turns / Length_km."
            )

    # --------------------------------------------------------
    # 8. v_proxy == avg_speed
    # --------------------------------------------------------
    avg_speed = pd.to_numeric(
        rt[
            "avg_speed_kmh"
        ],
        errors="coerce",
    )

    v_proxy = pd.to_numeric(
        rt[
            "v_proxy_kmh"
        ],
        errors="coerce",
    )

    if not np.isclose(
        avg_speed,
        v_proxy,
        atol=0.001,
    ):

        errors.append(
            "v_proxy_kmh is niet gelijk aan avg_speed_kmh."
        )

    # --------------------------------------------------------
    # 9. PoleMinusRecord formula
    # --------------------------------------------------------
    pole = pd.to_numeric(
        rt[
            "PoleTime_sec"
        ],
        errors="coerce",
    )

    record = pd.to_numeric(
        rt[
            "RecordLap_sec"
        ],
        errors="coerce",
    )

    pole_minus_record = pd.to_numeric(
        rt[
            "PoleMinusRecord_sec"
        ],
        errors="coerce",
    )

    expected_pole_delta = (
        pole
        - record
    )

    if not np.isclose(
        expected_pole_delta,
        pole_minus_record,
        atol=0.001,
    ):

        errors.append(
            "PoleMinusRecord_sec klopt niet met "
            "PoleTime_sec - RecordLap_sec."
        )

    # --------------------------------------------------------
    # 10. Q-weather plausibility
    # --------------------------------------------------------
    q_mean = pd.to_numeric(
        rt[
            "Q_airtemp_mean"
        ],
        errors="coerce",
    )

    q_max = pd.to_numeric(
        rt[
            "Q_airtemp_max"
        ],
        errors="coerce",
    )

    q_rain = pd.to_numeric(
        rt[
            "Q_rain_sum"
        ],
        errors="coerce",
    )

    q_humidity = pd.to_numeric(
        rt[
            "Q_humidity_mean"
        ],
        errors="coerce",
    )

    if q_max < q_mean:

        errors.append(
            "Q_airtemp_max is lager dan Q_airtemp_mean."
        )

    if q_rain < 0:

        errors.append(
            "Q_rain_sum is negatief."
        )

    if not (
        0
        <= q_humidity
        <= 100
    ):

        errors.append(
            "Q_humidity_mean ligt buiten 0-100."
        )

    # --------------------------------------------------------
    # 11. R-weather plausibility
    #     Retained for exact legacy compatibility even though
    #     weather-ready already validates the same concepts.
    # --------------------------------------------------------
    r_mean = pd.to_numeric(
        rt[
            "R_airtemp_mean"
        ],
        errors="coerce",
    )

    r_max = pd.to_numeric(
        rt[
            "R_airtemp_max"
        ],
        errors="coerce",
    )

    r_rain = pd.to_numeric(
        rt[
            "R_rain_sum"
        ],
        errors="coerce",
    )

    r_humidity = pd.to_numeric(
        rt[
            "R_humidity_mean"
        ],
        errors="coerce",
    )

    if r_max < r_mean:

        errors.append(
            "R_airtemp_max is lager dan R_airtemp_mean."
        )

    if r_rain < 0:

        errors.append(
            "R_rain_sum is negatief."
        )

    if not (
        0
        <= r_humidity
        <= 100
    ):

        errors.append(
            "R_humidity_mean ligt buiten 0-100."
        )

    # --------------------------------------------------------
    # 12. 2026 regime component sum
    # --------------------------------------------------------
    if int(
        year
    ) >= 2026:

        aero = pd.to_numeric(
            rt[
                "Regime2026_AeroComponent_sec"
            ],
            errors="coerce",
        )

        pu = pd.to_numeric(
            rt[
                "Regime2026_PUComponent_sec"
            ],
            errors="coerce",
        )

        mass = pd.to_numeric(
            rt[
                "Regime2026_MassComponent_sec"
            ],
            errors="coerce",
        )

        tyre = pd.to_numeric(
            rt[
                "Regime2026_TyreComponent_sec"
            ],
            errors="coerce",
        )

        prior = pd.to_numeric(
            rt[
                "Regime2026_PriorLapDelta_sec"
            ],
            errors="coerce",
        )

        expected_prior = (
            aero
            + pu
            + mass
            + tyre
        )

        if not np.isclose(
            expected_prior,
            prior,
            atol=0.001,
        ):

            errors.append(
                "Regime2026_PriorLapDelta_sec klopt niet "
                "met Aero + PU + Mass + Tyre."
            )

    return list(
        dict.fromkeys(
            errors
        )
    )

FEATURE_SNAPSHOT_COLS = (
    'FeatureSnapshotAfterQualifyingUTC',
    'FeatureSnapshotTimestampUTC',
)

R_WEATHER_AUDIT_COLS = (
    'R_airtemp_mean',
    'R_airtemp_max',
    'R_rain_sum',
    'R_humidity_mean',
    'WeatherSourceMode',
    'R_window_start_utc',
    'R_window_end_utc',
)


def _ensure_columns_local(
    df: pd.DataFrame,
    columns,
) -> pd.DataFrame:
    """
    Return a copy with all requested columns present.
    Missing columns are initialized with pd.NA.
    """

    out = df.copy()

    for col in columns:

        if col not in out.columns:

            out[col] = pd.NA

    return out


def _legacy_freeze_feature_snapshot(
    year: int,
    gp_slug: str,
    now_utc=None,
):
    """
    Canonical pre-race feature snapshot orchestration.

    Transactionally writes the same snapshot timestamp to
    df_test and df_overview after timing/status/cutoff checks.
    """

    preflight = io_preflight()

    if not preflight.get(
        "ok",
        False,
    ):

        raise RuntimeError(
            "I/O preflight faalt."
        )

    df_test = read_csv_checked(
        CONFIG.test_path
    )

    df_overview = read_csv_checked(
        CONFIG.overview_path
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

    timing_gate = validate_event_timing(
        df_test.loc[
            mt
        ].iloc[0],
        df_overview.loc[
            mo
        ].iloc[0],
    )

    if not timing_gate.ok:

        raise ValueError(
            "Feature snapshot geblokkeerd door "
            "event timing: "
            + timing_gate.detail
        )

    row_test = (
        df_test.loc[
            mt
        ].iloc[0]
    )

    row_overview = (
        df_overview.loc[
            mo
        ].iloc[0]
    )

    status_test = str(
        row_test.get(
            "RaceStatus",
            "",
        )
    ).strip()

    status_overview = str(
        row_overview.get(
            "RaceStatus",
            "",
        )
    ).strip()

    allowed_statuses = {
        "race_weather_ready",
        "race_context_plausibility_ready",
    }

    if status_test not in allowed_statuses:

        raise ValueError(
            "Feature snapshot mag alleen worden "
            "bevroren vanaf race_weather_ready of "
            "race_context_plausibility_ready. "
            f"df_test status={status_test!r}"
        )

    if status_overview not in allowed_statuses:

        raise ValueError(
            "Feature snapshot mag alleen worden "
            "bevroren vanaf race_weather_ready of "
            "race_context_plausibility_ready. "
            f"df_overview status={status_overview!r}"
        )

    if now_utc is None:

        snapshot_ts = pd.Timestamp.now(
            tz="UTC"
        )

    else:

        snapshot_ts = pd.to_datetime(
            now_utc,
            utc=True,
            errors="coerce",
        )

        if pd.isna(
            snapshot_ts
        ):

            raise ValueError(
                "now_utc is ongeldig."
            )

    cutoff_ts = pd.to_datetime(
        row_test[
            "PredictionCutoffUTC"
        ],
        utc=True,
        errors="coerce",
    )

    if pd.isna(
        cutoff_ts
    ):

        raise ValueError(
            "PredictionCutoffUTC is ongeldig."
        )

    if snapshot_ts > cutoff_ts:

        raise ValueError(
            "Feature snapshot kan niet worden bevroren: "
            f"actuele UTC {snapshot_ts.isoformat()} ligt ná "
            f"PredictionCutoffUTC {cutoff_ts.isoformat()}."
        )

    snapshot_value = (
        snapshot_ts
        .floor(
            "s"
        )
        .strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    new_test = _ensure_columns_local(
        df_test,
        FEATURE_SNAPSHOT_COLS,
    )

    new_overview = _ensure_columns_local(
        df_overview,
        FEATURE_SNAPSHOT_COLS,
    )

    mt = require_one_event(
        new_test,
        year,
        gp_slug,
        dataset_name="df_test",
    )

    mo = require_one_event(
        new_overview,
        year,
        gp_slug,
        dataset_name="df_overview",
    )

    for col in FEATURE_SNAPSHOT_COLS:

        new_test.loc[
            mt,
            col,
        ] = snapshot_value

        new_overview.loc[
            mo,
            col,
        ] = snapshot_value

    pre_gate = validate_feature_snapshot(
        new_test.loc[
            mt
        ].iloc[0],
        new_overview.loc[
            mo
        ].iloc[0],
    )

    if not pre_gate.ok:

        raise ValueError(
            "Pre-write feature snapshot "
            "validatie faalt: "
            + pre_gate.detail
        )

    safe_commit_many(
        {
            CONFIG.test_path:
                new_test,

            CONFIG.overview_path:
                new_overview,
        },
        backup_dir=(
            CONFIG.backup_dir
        ),
    )

    final_test = read_csv_checked(
        CONFIG.test_path
    )

    final_overview = read_csv_checked(
        CONFIG.overview_path
    )

    final_mt = require_one_event(
        final_test,
        year,
        gp_slug,
        dataset_name="df_test",
    )

    final_mo = require_one_event(
        final_overview,
        year,
        gp_slug,
        dataset_name="df_overview",
    )

    post_gate = validate_feature_snapshot(
        final_test.loc[
            final_mt
        ].iloc[0],
        final_overview.loc[
            final_mo
        ].iloc[0],
    )

    if not post_gate.ok:

        raise RuntimeError(
            "Post-write feature snapshot "
            "validatie faalt: "
            + post_gate.detail
        )

    return {
        "ok":
            True,

        "FeatureSnapshotAfterQualifyingUTC":
            snapshot_value,

        "FeatureSnapshotTimestampUTC":
            snapshot_value,

        "PredictionCutoffUTC":
            cutoff_ts.strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
    }



@_service_storage_transaction
def save_race_weather_audit(
    year: int,
    gp_slug: str,
    r_airtemp_mean,
    r_airtemp_max,
    r_rain_sum,
    r_humidity_mean,
    weather_source_mode: str,
):
    """
    Canonical race-weather orchestration.

    Weather data is committed transactionally first.
    Lifecycle transition to race_weather_ready happens only
    after weather readback validation succeeds.
    """
    _assert_inputs_not_frozen(year, gp_slug)

    preflight = io_preflight()

    if not preflight.get(
        "ok",
        False,
    ):

        raise RuntimeError(
            "I/O preflight faalt."
        )

    df_test = read_csv_checked(
        CONFIG.test_path
    )

    df_overview = read_csv_checked(
        CONFIG.overview_path
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

    timing_gate = validate_event_timing(
        df_test.loc[
            mt
        ].iloc[0],
        df_overview.loc[
            mo
        ].iloc[0],
    )

    if not timing_gate.ok:

        raise ValueError(
            "Event timing is niet volledig/geldig: "
            + timing_gate.detail
        )

    test_row = (
        df_test.loc[
            mt
        ].iloc[0]
    )

    current_status = str(
        test_row.get(
            "RaceStatus",
            "",
        )
    ).strip()

    if current_status != "qualifying_ready":

        raise ValueError(
            "Race Weather mag alleen worden bevestigd "
            "vanuit qualifying_ready. "
            f"Huidige status: {current_status!r}"
        )

    overview_status = str(
        df_overview.loc[
            mo
        ].iloc[0].get(
            "RaceStatus",
            "",
        )
    ).strip()

    if overview_status != current_status:

        raise ValueError(
            "RaceStatus mismatch vóór weather save: "
            f"test={current_status!r}, "
            f"overview={overview_status!r}"
        )

    values = {
        "R_airtemp_mean":
            float(
                r_airtemp_mean
            ),

        "R_airtemp_max":
            float(
                r_airtemp_max
            ),

        "R_rain_sum":
            float(
                r_rain_sum
            ),

        "R_humidity_mean":
            float(
                r_humidity_mean
            ),

        "WeatherSourceMode":
            str(
                weather_source_mode
            ).strip(),

        "R_window_start_utc":
            str(
                test_row[
                    "R_window_start_utc"
                ]
            ),

        "R_window_end_utc":
            str(
                test_row[
                    "R_window_end_utc"
                ]
            ),
    }

    if not values[
        "WeatherSourceMode"
    ]:

        raise ValueError(
            "WeatherSourceMode mag niet leeg zijn."
        )

    new_test = _ensure_columns_local(
        df_test,
        R_WEATHER_AUDIT_COLS,
    )

    new_overview = _ensure_columns_local(
        df_overview,
        R_WEATHER_AUDIT_COLS,
    )

    new_test = _ensure_columns_local(
        new_test,
        [
            "RaceStatus"
        ],
    )

    new_overview = _ensure_columns_local(
        new_overview,
        [
            "RaceStatus"
        ],
    )

    mt = require_one_event(
        new_test,
        year,
        gp_slug,
        dataset_name="df_test",
    )

    mo = require_one_event(
        new_overview,
        year,
        gp_slug,
        dataset_name="df_overview",
    )

    for col, value in values.items():

        new_test.loc[
            mt,
            col,
        ] = value

        new_overview.loc[
            mo,
            col,
        ] = value

    pre_errors = (
        validate_race_weather_audit_rows(
            new_test.loc[
                mt
            ].iloc[0],
            new_overview.loc[
                mo
            ].iloc[0],
            require_status=False,
        )
    )

    if pre_errors:

        raise ValueError(
            "Pre-write R-weather validatie faalt: "
            + " | ".join(
                pre_errors
            )
        )

    safe_commit_many(
        {
            CONFIG.test_path:
                new_test,

            CONFIG.overview_path:
                new_overview,
        },
        backup_dir=(
            CONFIG.backup_dir
        ),
    )

    check_test = read_csv_checked(
        CONFIG.test_path
    )

    check_overview = read_csv_checked(
        CONFIG.overview_path
    )

    check_mt = require_one_event(
        check_test,
        year,
        gp_slug,
        dataset_name="df_test",
    )

    check_mo = require_one_event(
        check_overview,
        year,
        gp_slug,
        dataset_name="df_overview",
    )

    post_errors = (
        validate_race_weather_audit_rows(
            check_test.loc[
                check_mt
            ].iloc[0],
            check_overview.loc[
                check_mo
            ].iloc[0],
            require_status=False,
        )
    )

    if post_errors:

        raise RuntimeError(
            "Post-write R-weather validatie faalt: "
            + " | ".join(
                post_errors
            )
        )

    transition_event_status(
        year=year,
        gp_slug=gp_slug,
        expected_from="qualifying_ready",
        to_status="race_weather_ready",
    )

    final_test = read_csv_checked(
        CONFIG.test_path
    )

    final_overview = read_csv_checked(
        CONFIG.overview_path
    )

    final_mt = require_one_event(
        final_test,
        year,
        gp_slug,
        dataset_name="df_test",
    )

    final_mo = require_one_event(
        final_overview,
        year,
        gp_slug,
        dataset_name="df_overview",
    )

    final_errors = (
        validate_race_weather_audit_rows(
            final_test.loc[
                final_mt
            ].iloc[0],
            final_overview.loc[
                final_mo
            ].iloc[0],
            require_status=True,
        )
    )

    if final_errors:

        raise RuntimeError(
            "Finale R-weather validatie faalt: "
            + " | ".join(
                final_errors
            )
        )

    return {
        "ok":
            True,

        "RaceStatus":
            "race_weather_ready",

        "values":
            values,
    }

# ============================================================
# Canonical pre-race preparation helpers
# Migrated from app_streamlit.py in Phase 7H.67
# ============================================================

def determine_year_offset(
    year: int,
    df_test: pd.DataFrame,
    df_overview: pd.DataFrame,
    df_training: pd.DataFrame,
) -> Tuple[float, str]:
    for source_name, df in [
        ("df_test", df_test),
        ("df_overview", df_overview),
        ("df_training", df_training),
    ]:
        if {"Year", "YearOffset"}.issubset(df.columns):
            values = (
                df.loc[pd.to_numeric(df["Year"], errors="coerce").eq(int(year)), "YearOffset"]
                .dropna()
                .unique()
            )
            if len(values) > 0:
                return float(values[0]), source_name
    raise ValueError(
        f"Geen bestaande Year→YearOffset-mapping gevonden voor {year}. "
        "De app gebruikt geen max+1 of handmatige extrapolatie."
    )

def corner_fraction(v_proxy: float) -> float:
    if v_proxy < 200:
        return 0.62
    if v_proxy <= 220:
        return 0.55
    return 0.48

def component_from_existing_2026(
    col: str,
    target_cf: float,
    sources: List[Tuple[str, pd.DataFrame]],
) -> Tuple[float, str]:
    for source_name, df in sources:
        if not {"Year", "CornerFraction_est", col}.issubset(df.columns):
            continue
        tmp = df.copy()
        tmp["Year"] = pd.to_numeric(tmp["Year"], errors="coerce")
        tmp["CornerFraction_est"] = pd.to_numeric(tmp["CornerFraction_est"], errors="coerce")
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.loc[
            tmp["Year"].ge(2026)
            & tmp["CornerFraction_est"].notna()
            & tmp[col].notna()
        ]
        if tmp.empty:
            continue

        exact = tmp.loc[np.isclose(tmp["CornerFraction_est"], target_cf)]
        if not exact.empty:
            return float(exact[col].iloc[0]), f"{source_name}: exacte CornerFraction_est-match"

        pts = (
            tmp[["CornerFraction_est", col]]
            .drop_duplicates()
            .sort_values("CornerFraction_est")
        )
        if len(pts) >= 2:
            value = float(
                np.interp(
                    target_cf,
                    pts["CornerFraction_est"].to_numpy(float),
                    pts[col].to_numpy(float),
                )
            )
            return value, f"{source_name}: interpolatie op bestaande 2026-waarden"

    raise ValueError(f"Geen geldige bron gevonden voor {col}.")

def unique_circuit_reference(
    df_test: pd.DataFrame,
    circuit_info: pd.DataFrame,
) -> pd.DataFrame:
    cols = ["gp_slug", "Circuit", "Turns", "Length_km", "avg_speed_kmh"]
    pieces = []
    for df in [df_test, circuit_info]:
        available = [c for c in cols if c in df.columns]
        if "gp_slug" in available:
            pieces.append(df[available].copy())
    ref = pd.concat(pieces, ignore_index=True, sort=False)
    ref = ref.drop_duplicates(subset=["gp_slug"], keep="last")
    ref["Turns"] = pd.to_numeric(ref.get("Turns"), errors="coerce")
    ref["Length_km"] = pd.to_numeric(ref.get("Length_km"), errors="coerce")
    ref["avg_speed_kmh"] = pd.to_numeric(ref.get("avg_speed_kmh"), errors="coerce")
    ref["Turns/Length_km"] = ref["Turns"] / ref["Length_km"]
    return ref

def rank_for_new_race(
    gp_slug: str,
    df_test: pd.DataFrame,
    circuit_info: pd.DataFrame,
) -> Tuple[float, float]:
    ref = unique_circuit_reference(df_test, circuit_info)
    if gp_slug not in set(ref["gp_slug"].astype(str)):
        raise ValueError("Nieuwe GP ontbreekt in de unieke circuitreferentieset.")

    ref["Turns/length rank"] = ref["Turns/Length_km"].rank(
        method="min", ascending=False
    )
    ref["avg_speed rank"] = ref["avg_speed_kmh"].rank(
        method="min", ascending=False
    )
    row = ref.loc[ref["gp_slug"].astype(str).eq(gp_slug)].iloc[0]
    return float(row["Turns/length rank"]), float(row["avg_speed rank"])

def circuit_row_by_slug(circuit_info: pd.DataFrame, gp_slug: str) -> pd.Series:
    match = circuit_info.loc[circuit_info["gp_slug"].astype(str).eq(gp_slug)]
    if len(match) != 1:
        raise ValueError(
            f"circuit_info: gp_slug={gp_slug} moet exact één keer voorkomen; gevonden {len(match)}."
        )
    return match.iloc[0]

def value_from_row(row: pd.Series, names: List[str], default=np.nan):
    lookup = {str(c).lower(): c for c in row.index}
    for name in names:
        if name.lower() in lookup:
            return row[lookup[name.lower()]]
    return default

def build_structural_features(
    year: int,
    gp_slug: str,
    is_sprint: int,
    df_test: pd.DataFrame,
    df_overview: pd.DataFrame,
    df_training: pd.DataFrame,
    circuit_info: pd.DataFrame,
) -> Tuple[Dict, List[str]]:
    # EVENT_CIRCUIT_SELECTION_V1
    from .event_circuits import resolve_event_circuit
    ci, circuit_reference_slug, event_layout_slug = (
        resolve_event_circuit(year, gp_slug, circuit_info)
    )
    log = []

    year_offset, source = determine_year_offset(
        year, df_test, df_overview, df_training
    )
    log.append(f"YearOffset via {source}")

    data = {
        "YearOffset": year_offset,
        "Circuit": value_from_row(ci, ["Circuit"]),
        "Elevation_m": value_from_row(ci, ["Elevation_m"]),
        "Length_km": value_from_row(ci, ["Length_km"]),
        "Turns": value_from_row(ci, ["Turns"]),
        "Altitude_m": value_from_row(ci, ["Altitude_m"]),
        "Altitude_bucket": value_from_row(ci, ["Altitude_bucket"]),
        "Downforce_profile": value_from_row(ci, ["Downforce_profile"]),
        "Track_type": value_from_row(ci, ["Track_type"]),
        "Direction": value_from_row(ci, ["Direction"]),
        "Brake_intensity": value_from_row(ci, ["Brake_intensity"]),
        "avg_speed_kmh": value_from_row(ci, ["avg_speed_kmh"]),
        "RecordLap_sec": value_from_row(ci, ["RecordLap_sec"]),
        "Latitude": value_from_row(ci, ["Latitude", "latitude", "lat"]),
        "Longitude": value_from_row(ci, ["Longitude", "longitude", "lon", "lng"]),
        "IsSprintWeekend": int(is_sprint),
    }

    for required in ["Length_km", "Turns", "avg_speed_kmh", "RecordLap_sec", "Latitude", "Longitude"]:
        if pd.isna(data[required]):
            raise ValueError(f"Verplicht circuitveld ontbreekt: {required}")

    data["Turns/Length_km"] = float(data["Turns"]) / float(data["Length_km"])
    data["TurnDensity"] = data["Turns/Length_km"]
    data["v_proxy_kmh"] = float(data["avg_speed_kmh"])

    # Scope the ranking reference to this event without rewriting the circuit master.
    event_rank_reference = circuit_info.copy()
    reference_mask = event_rank_reference["gp_slug"].astype(str).eq(circuit_reference_slug)
    for field in ("Length_km", "Turns", "avg_speed_kmh"):
        event_rank_reference.loc[reference_mask, field] = ci[field]
    tl_rank, speed_rank = rank_for_new_race(
        circuit_reference_slug, df_test, event_rank_reference
    )
    data["Turns/length rank"] = tl_rank
    data["avg_speed rank"] = speed_rank

    if year >= 2026:
        cf = corner_fraction(data["v_proxy_kmh"])
        data["CornerFraction_est"] = cf
        data["Regime2026_Flag"] = 1

        sources = [
            ("df_test", df_test),
            ("df_overview", df_overview),
            ("df_training", df_training),
        ]
        for col in [
            "Regime2026_AeroComponent_sec",
            "Regime2026_PUComponent_sec",
            "Regime2026_TyreComponent_sec",
        ]:
            value, method = component_from_existing_2026(col, cf, sources)
            data[col] = value
            log.append(f"{col}: {method}")

        data["Regime2026_MassComponent_sec"] = -0.35
        data["Regime2026_PriorLapDelta_sec"] = (
            data["Regime2026_AeroComponent_sec"]
            + data["Regime2026_PUComponent_sec"]
            + data["Regime2026_MassComponent_sec"]
            + data["Regime2026_TyreComponent_sec"]
        )
    else:
        data.update({
            "CornerFraction_est": np.nan,
            "Regime2026_Flag": 0,
            "Regime2026_AeroComponent_sec": 0.0,
            "Regime2026_PUComponent_sec": 0.0,
            "Regime2026_MassComponent_sec": 0.0,
            "Regime2026_TyreComponent_sec": 0.0,
            "Regime2026_PriorLapDelta_sec": 0.0,
        })

    if event_layout_slug is not None:
        data["CircuitLayoutSlug"] = event_layout_slug
        log.append(
            f"Event {year}_{gp_slug}: circuit reference "
            f"{circuit_reference_slug}; layout {event_layout_slug}"
        )

    return data, log

def _utc_iso(value):
    """
    Canonical UTC string:
    YYYY-MM-DDTHH:MM:SSZ
    """
    ts = pd.to_datetime(
        value,
        utc=True,
        errors="coerce",
    )

    if pd.isna(ts):
        return np.nan

    return ts.strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

def _legacy_build_event_timing_values(
    event_datetime_utc,
    cutoff_hours=3.0,
    race_window_minutes=120,
    cutoff_policy=None,
    race_window_source=None,
):
    """
    Bouwt de canonical event-timingvelden.

    Default:
    PredictionCutoffUTC = EventDateTimeUTC - 3 uur
    Race window         = EventDateTimeUTC t/m +120 min
    """

    event_ts = pd.to_datetime(
        event_datetime_utc,
        utc=True,
        errors="coerce",
    )

    if pd.isna(event_ts):
        raise ValueError(
            "EventDateTimeUTC is ongeldig."
        )

    cutoff_hours = float(
        cutoff_hours
    )

    race_window_minutes = int(
        race_window_minutes
    )

    if cutoff_hours <= 0:
        raise ValueError(
            "cutoff_hours moet > 0 zijn."
        )

    if race_window_minutes <= 0:
        raise ValueError(
            "RaceWindowDurationMinutes moet > 0 zijn."
        )

    cutoff_ts = (
        event_ts
        - pd.Timedelta(
            hours=cutoff_hours
        )
    )

    window_start = event_ts

    window_end = (
        event_ts
        + pd.Timedelta(
            minutes=race_window_minutes
        )
    )

    if cutoff_policy is None:
        cutoff_policy = (
            f"T-{cutoff_hours:g}h before EventDateTimeUTC"
        )

    if race_window_source is None:
        race_window_source = (
            "EventDateTimeUTC + configured race-window duration"
        )

    return {
        "EventDateTimeUTC":
            _utc_iso(event_ts),

        "PredictionCutoffUTC":
            _utc_iso(cutoff_ts),

        "PredictionCutoffPolicy":
            str(cutoff_policy),

        "R_window_start_utc":
            _utc_iso(window_start),

        "R_window_end_utc":
            _utc_iso(window_end),

        "RaceWindowDurationMinutes":
            int(race_window_minutes),

        "RaceWindowSource":
            str(race_window_source),
    }

# ============================================================
# Canonical operational pre-race write orchestrators
# Phase 7H.68a
# ============================================================

def _canonical_operational_validators():
    return {
        CONFIG.test_path:
            lambda df:
                validate_test_contract(
                    df,
                    strict_schema=True,
                ),

        CONFIG.overview_path:
            validate_overview_contract,
    }


def _append_prediction_ready_comment(
    year: int,
    gp_slug: str,
) -> None:

    df_overview = read_csv_checked(
        CONFIG.overview_path
    )

    mask = require_one_event(
        df_overview,
        int(year),
        str(gp_slug),
        dataset_name="df_overview",
    )

    if "Commentaar" not in df_overview.columns:
        df_overview["Commentaar"] = pd.NA

    existing = df_overview.loc[
        mask,
        "Commentaar",
    ].iloc[0]

    if pd.isna(existing):
        existing = ""

    readiness_comment = (
        "Prediction Readiness geslaagd; "
        "RaceStatus=prediction_ready. "
        "Nog geen modelrun uitgevoerd."
    )

    existing_text = str(existing).strip()

    if readiness_comment not in existing_text:

        df_overview.loc[
            mask,
            "Commentaar",
        ] = (
            existing_text
            + " "
            + readiness_comment
        ).strip()

        safe_commit_csv(
            df_overview,
            CONFIG.overview_path,
            validator=validate_overview_contract,
            backup_dir=CONFIG.backup_dir,
        )


def _legacy_save_event_timing(
    year: int,
    gp_slug: str,
    event_datetime_utc,
    *,
    cutoff_hours: float = 3.0,
    race_window_minutes: int = 120,
):

    year = int(year)
    gp_slug = str(gp_slug).strip()

    timing_values = build_event_timing_values(
        event_datetime_utc,
        cutoff_hours=float(cutoff_hours),
        race_window_minutes=int(
            race_window_minutes
        ),
    )

    df_test = read_csv_checked(
        CONFIG.test_path
    )

    df_overview = read_csv_checked(
        CONFIG.overview_path
    )

    new_test = _ensure_columns_local(
        df_test,
        list(timing_values.keys()),
    )

    new_overview = _ensure_columns_local(
        df_overview,
        list(timing_values.keys()),
    )

    mt = require_one_event(
        new_test,
        year,
        gp_slug,
        dataset_name="df_test",
    )

    mo = require_one_event(
        new_overview,
        year,
        gp_slug,
        dataset_name="df_overview",
    )

    for col, value in timing_values.items():
        new_test.loc[mt, col] = value
        new_overview.loc[mo, col] = value

    gate = validate_event_timing(
        new_test.loc[mt].iloc[0],
        new_overview.loc[mo].iloc[0],
    )

    if not gate.ok:
        raise ValueError(
            "Event timing validatie faalt: "
            + str(gate.detail)
        )

    safe_commit_many(
        {
            CONFIG.test_path: new_test,
            CONFIG.overview_path: new_overview,
        },
        validators=_canonical_operational_validators(),
        backup_dir=CONFIG.backup_dir,
    )

    final_test = read_csv_checked(
        CONFIG.test_path
    )

    final_overview = read_csv_checked(
        CONFIG.overview_path
    )

    mt2 = require_one_event(
        final_test,
        year,
        gp_slug,
        dataset_name="df_test",
    )

    mo2 = require_one_event(
        final_overview,
        year,
        gp_slug,
        dataset_name="df_overview",
    )

    final_gate = validate_event_timing(
        final_test.loc[mt2].iloc[0],
        final_overview.loc[mo2].iloc[0],
    )

    if not final_gate.ok:
        raise RuntimeError(
            "Event timing readback validatie faalt: "
            + str(final_gate.detail)
        )

    return {
        "ok": True,
        "year": year,
        "gp_slug": gp_slug,
        "timing_values": timing_values,
    }


@_service_storage_transaction
def prepare_circuit_regime_ready(
    year: int,
    gp_slug: str,
    *,
    is_sprint_weekend: bool,
):

    _assert_inputs_not_frozen(year, gp_slug)
    year = int(year)
    gp_slug = str(gp_slug).strip()

    df_test = read_csv_checked(
        CONFIG.test_path
    )

    df_overview = read_csv_checked(
        CONFIG.overview_path
    )

    df_training = read_csv_checked(
        CONFIG.training_path
    )

    circuit_info = read_csv_checked(
        CONFIG.circuit_info_path
    )

    features, log = build_structural_features(
        year,
        gp_slug,
        int(bool(is_sprint_weekend)),
        df_test,
        df_overview,
        df_training,
        circuit_info,
    )

    new_test = _ensure_columns_local(
        df_test,
        list(features.keys())
        + ["RaceStatus"],
    )

    new_overview = _ensure_columns_local(
        df_overview,
        list(features.keys())
        + [
            "RaceStatus",
            "Commentaar",
        ],
    )

    mt = require_one_event(
        new_test,
        year,
        gp_slug,
        dataset_name="df_test",
    )

    mo = require_one_event(
        new_overview,
        year,
        gp_slug,
        dataset_name="df_overview",
    )

    for col, value in features.items():
        new_test.loc[mt, col] = value
        new_overview.loc[mo, col] = value

    new_overview.loc[
        mo,
        "Commentaar",
    ] = (
        "Circuit/regimeverrijking voltooid; "
        "nog niet qualifying_ready."
    )

    validation_test = new_test.copy()
    validation_overview = new_overview.copy()

    validation_test.loc[
        mt,
        "RaceStatus",
    ] = "circuit_regime_ready"

    validation_overview.loc[
        mo,
        "RaceStatus",
    ] = "circuit_regime_ready"

    errors = validate_circuit_regime_rows(
        validation_test,
        validation_overview,
        year,
        gp_slug,
    )

    if errors:
        raise ValueError(
            "Circuit/regime validatie faalt: "
            + " | ".join(str(x) for x in errors)
        )

    safe_commit_many(
        {
            CONFIG.test_path: new_test,
            CONFIG.overview_path: new_overview,
        },
        validators=_canonical_operational_validators(),
        backup_dir=CONFIG.backup_dir,
    )

    transition = transition_event_status(
        year=year,
        gp_slug=gp_slug,
        expected_from="row_created",
        to_status="circuit_regime_ready",
    )

    final_test = read_csv_checked(
        CONFIG.test_path
    )

    final_overview = read_csv_checked(
        CONFIG.overview_path
    )

    final_errors = validate_circuit_regime_rows(
        final_test,
        final_overview,
        year,
        gp_slug,
    )

    if final_errors:
        raise RuntimeError(
            "Circuit/regime readback validatie faalt: "
            + " | ".join(
                str(x)
                for x in final_errors
            )
        )

    return {
        "ok": bool(transition.ok),
        "year": year,
        "gp_slug": gp_slug,
        "RaceStatus":
            "circuit_regime_ready",
        "feature_log": log,
    }


@_service_storage_transaction
def save_qualifying_ready(
    year: int,
    gp_slug: str,
    *,
    q_airtemp_mean: float,
    q_airtemp_max: float,
    q_rain_sum: float,
    q_humidity_mean: float,
    pole_time_sec: float,
    pole_compound: str,
    pole_c_index: int,
    sprint_fastest_lap_sec=None,
):

    _assert_inputs_not_frozen(year, gp_slug)
    year = int(year)
    gp_slug = str(gp_slug).strip()

    q_airtemp_mean = float(
        q_airtemp_mean
    )
    q_airtemp_max = float(
        q_airtemp_max
    )
    q_rain_sum = float(
        q_rain_sum
    )
    q_humidity_mean = float(
        q_humidity_mean
    )
    pole_time_sec = float(
        pole_time_sec
    )
    pole_c_index = int(
        pole_c_index
    )

    if q_airtemp_max < q_airtemp_mean:
        raise ValueError(
            "Q_airtemp_max moet >= Q_airtemp_mean zijn."
        )

    if q_rain_sum < 0:
        raise ValueError(
            "Q_rain_sum moet >= 0 zijn."
        )

    if not (
        0 <= q_humidity_mean <= 100
    ):
        raise ValueError(
            "Q_humidity_mean moet tussen 0 en 100 liggen."
        )

    if pole_time_sec <= 0:
        raise ValueError(
            "PoleTime_sec moet > 0 zijn."
        )

    if pole_c_index < 0:
        raise ValueError(
            "pole_c_index mag niet negatief zijn."
        )

    df_test = read_csv_checked(
        CONFIG.test_path
    )

    df_overview = read_csv_checked(
        CONFIG.overview_path
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

    sprint_flag = (
        pd.to_numeric(
            df_test.loc[
                mt,
                "IsSprintWeekend",
            ],
            errors="coerce",
        )
        .fillna(0)
        .iloc[0]
    )

    is_sprint = (
        int(sprint_flag) == 1
    )

    if is_sprint:
        if (
            sprint_fastest_lap_sec
            is None
            or float(
                sprint_fastest_lap_sec
            ) <= 0
        ):
            raise ValueError(
                "SprintFastestLap_sec is verplicht "
                "en moet > 0 zijn bij een sprintweekend."
            )

    cols = [
        "Q_airtemp_mean",
        "Q_airtemp_max",
        "Q_rain_sum",
        "Q_humidity_mean",
        "PoleTime_sec",
        "PoleMinusRecord_sec",
        "pole_compound",
        "pole_c_index",
        "RaceStatus",
    ]

    if is_sprint:
        cols.append(
            "SprintFastestLap_sec"
        )

    new_test = _ensure_columns_local(
        df_test,
        cols,
    )

    new_overview = _ensure_columns_local(
        df_overview,
        cols + ["Commentaar"],
    )

    mt = require_one_event(
        new_test,
        year,
        gp_slug,
        dataset_name="df_test",
    )

    mo = require_one_event(
        new_overview,
        year,
        gp_slug,
        dataset_name="df_overview",
    )

    record = pd.to_numeric(
        new_test.loc[
            mt,
            "RecordLap_sec",
        ],
        errors="coerce",
    ).iloc[0]

    if pd.isna(record):
        raise ValueError(
            "RecordLap_sec ontbreekt; "
            "qualifying-update gestopt."
        )

    values = {
        "Q_airtemp_mean":
            q_airtemp_mean,
        "Q_airtemp_max":
            q_airtemp_max,
        "Q_rain_sum":
            q_rain_sum,
        "Q_humidity_mean":
            q_humidity_mean,
        "PoleTime_sec":
            pole_time_sec,
        "pole_compound":
            str(
                pole_compound
            ).strip().lower(),
        "pole_c_index":
            pole_c_index,
        "PoleMinusRecord_sec":
            pole_time_sec
            - float(record),
    }

    if is_sprint:
        values[
            "SprintFastestLap_sec"
        ] = float(
            sprint_fastest_lap_sec
        )

    for col, value in values.items():
        new_test.loc[mt, col] = value
        new_overview.loc[mo, col] = value

    # QUALIFYING_COMMENT_PRESERVATION_V1
    qualifying_note = (
        "Qualifyinggegevens compleet; klaar voor qualifying_ready."
    )
    existing_comment = new_overview.loc[mo, "Commentaar"].iloc[0]
    existing_comment = (
        "" if pd.isna(existing_comment) else str(existing_comment)
    )
    if qualifying_note not in existing_comment:
        new_overview.loc[mo, "Commentaar"] = (
            existing_comment
            + (" | " if existing_comment else "")
            + qualifying_note
        )

    validation_test = new_test.copy()
    validation_overview = new_overview.copy()

    validation_test.loc[
        mt,
        "RaceStatus",
    ] = "qualifying_ready"

    validation_overview.loc[
        mo,
        "RaceStatus",
    ] = "qualifying_ready"

    errors = validate_qualifying_ready_rows(
        validation_test,
        validation_overview,
        year,
        gp_slug,
    )

    if errors:
        raise ValueError(
            "Qualifyingvalidatie faalt: "
            + " | ".join(
                str(x)
                for x in errors
            )
        )

    safe_commit_many(
        {
            CONFIG.test_path: new_test,
            CONFIG.overview_path: new_overview,
        },
        validators=_canonical_operational_validators(),
        backup_dir=CONFIG.backup_dir,
    )

    transition = transition_event_status(
        year=year,
        gp_slug=gp_slug,
        expected_from="circuit_regime_ready",
        to_status="qualifying_ready",
    )

    final_test = read_csv_checked(
        CONFIG.test_path
    )

    final_overview = read_csv_checked(
        CONFIG.overview_path
    )

    final_errors = (
        validate_qualifying_ready_rows(
            final_test,
            final_overview,
            year,
            gp_slug,
        )
    )

    if final_errors:
        raise RuntimeError(
            "Qualifying readback validatie faalt: "
            + " | ".join(
                str(x)
                for x in final_errors
            )
        )

    return {
        "ok": bool(transition.ok),
        "year": year,
        "gp_slug": gp_slug,
        "RaceStatus":
            "qualifying_ready",
    }


@_service_storage_transaction
def prepare_race_weather_ready(
    year: int,
    gp_slug: str,
    *,
    r_airtemp_mean: float,
    r_airtemp_max: float,
    r_rain_sum: float,
    r_humidity_mean: float,
):

    _assert_inputs_not_frozen(year, gp_slug)
    year = int(year)
    gp_slug = str(gp_slug).strip()

    r_airtemp_mean = float(
        r_airtemp_mean
    )
    r_airtemp_max = float(
        r_airtemp_max
    )
    r_rain_sum = float(
        r_rain_sum
    )
    r_humidity_mean = float(
        r_humidity_mean
    )

    if r_airtemp_max < r_airtemp_mean:
        raise ValueError(
            "R_airtemp_max moet >= R_airtemp_mean zijn."
        )

    if r_rain_sum < 0:
        raise ValueError(
            "R_rain_sum mag niet negatief zijn."
        )

    if not (
        0 <= r_humidity_mean <= 100
    ):
        raise ValueError(
            "R_humidity_mean moet tussen 0 en 100 liggen."
        )

    df_test = read_csv_checked(
        CONFIG.test_path
    )

    df_overview = read_csv_checked(
        CONFIG.overview_path
    )

    cols = [
        "R_airtemp_mean",
        "R_airtemp_max",
        "R_rain_sum",
        "R_humidity_mean",
        "RaceStatus",
    ]

    new_test = _ensure_columns_local(
        df_test,
        cols,
    )

    new_overview = _ensure_columns_local(
        df_overview,
        cols + ["Commentaar"],
    )

    mt = require_one_event(
        new_test,
        year,
        gp_slug,
        dataset_name="df_test",
    )

    mo = require_one_event(
        new_overview,
        year,
        gp_slug,
        dataset_name="df_overview",
    )

    values = {
        "R_airtemp_mean":
            r_airtemp_mean,
        "R_airtemp_max":
            r_airtemp_max,
        "R_rain_sum":
            r_rain_sum,
        "R_humidity_mean":
            r_humidity_mean,
    }

    for col, value in values.items():
        new_test.loc[mt, col] = value
        new_overview.loc[mo, col] = value

    q_rain = pd.to_numeric(
        new_test.loc[
            mt,
            "Q_rain_sum",
        ],
        errors="coerce",
    ).iloc[0]

    q_humidity = pd.to_numeric(
        new_test.loc[
            mt,
            "Q_humidity_mean",
        ],
        errors="coerce",
    ).iloc[0]

    wet_forecast_flag = int(
        r_rain_sum >= 1.0
    )

    condition_shift = False
    condition_notes = []

    if (
        pd.notna(q_rain)
        and float(q_rain) < 1.0
        and r_rain_sum >= 1.0
    ):
        condition_shift = True
        condition_notes.append(
            "Dry/wetter qualifying naar natte raceforecast."
        )

    if (
        pd.notna(q_humidity)
        and (
            r_humidity_mean
            - float(q_humidity)
            >= 15.0
        )
    ):
        condition_shift = True
        condition_notes.append(
            "Materiële stijging luchtvochtigheid versus qualifying."
        )

    if wet_forecast_flag:
        condition_notes.append(
            "WetForecastFlag=1; forecast rain is niet automatisch "
            "gelijk aan pace-relevante natte omstandigheden."
        )

    comment = (
        "Race weather compleet; "
        "RaceStatus=race_weather_ready."
    )

    if condition_notes:
        comment += (
            " "
            + " ".join(
                condition_notes
            )
        )

    new_overview.loc[
        mo,
        "Commentaar",
    ] = comment

    validation_test = new_test.copy()
    validation_overview = new_overview.copy()

    validation_test.loc[
        mt,
        "RaceStatus",
    ] = "race_weather_ready"

    validation_overview.loc[
        mo,
        "RaceStatus",
    ] = "race_weather_ready"

    errors = (
        validate_race_weather_ready_rows(
            validation_test,
            validation_overview,
            year,
            gp_slug,
        )
    )

    if errors:
        raise ValueError(
            "Race-weathervalidatie faalt: "
            + " | ".join(
                str(x)
                for x in errors
            )
        )

    safe_commit_many(
        {
            CONFIG.test_path: new_test,
            CONFIG.overview_path: new_overview,
        },
        validators=_canonical_operational_validators(),
        backup_dir=CONFIG.backup_dir,
    )

    transition = transition_event_status(
        year=year,
        gp_slug=gp_slug,
        expected_from="qualifying_ready",
        to_status="race_weather_ready",
    )

    final_test = read_csv_checked(
        CONFIG.test_path
    )

    final_overview = read_csv_checked(
        CONFIG.overview_path
    )

    final_errors = (
        validate_race_weather_ready_rows(
            final_test,
            final_overview,
            year,
            gp_slug,
        )
    )

    if final_errors:
        raise RuntimeError(
            "Race-weather readback validatie faalt: "
            + " | ".join(
                str(x)
                for x in final_errors
            )
        )

    return {
        "ok": bool(transition.ok),
        "year": year,
        "gp_slug": gp_slug,
        "RaceStatus":
            "race_weather_ready",
        "wet_forecast_flag":
            wet_forecast_flag,
        "condition_shift":
            condition_shift,
        "condition_notes":
            condition_notes,
    }


# EVENT_VERIFICATION_WRITER_V1
@_service_storage_transaction
def save_event_verification(year, gp_slug, *, evidence):
    """Persist reviewed event/layout/record evidence without changing pace or state."""
    _assert_inputs_not_frozen(year, gp_slug)
    import json
    from .event_circuits import resolve_event_circuit

    year = int(year)
    gp_slug = str(gp_slug).strip()

    required_evidence = {
        "CircuitReferenceSlug", "CircuitLayoutSlug",
        "Length_km", "Turns", "RecordLap_sec", "EventDateTimeUTC",
        "EventSource", "LayoutSource", "RecordLapSource",
    }
    missing = required_evidence - set(evidence)
    if missing:
        raise ValueError(f"Missing verification evidence: {sorted(missing)}")

    for key in ["EventSource", "LayoutSource", "RecordLapSource"]:
        if not str(evidence[key]).startswith("https://"):
            raise ValueError(f"Invalid source: {key}")

    dt = read_csv_checked(CONFIG.test_path)
    do = read_csv_checked(CONFIG.overview_path)
    ci = read_csv_checked(CONFIG.circuit_info_path)

    mt = require_one_event(dt, year, gp_slug, dataset_name="df_test")
    mo = require_one_event(do, year, gp_slug, dataset_name="df_overview")
    circuit, reference, layout = resolve_event_circuit(year, gp_slug, ci)

    if reference != evidence["CircuitReferenceSlug"]:
        raise ValueError("Circuit reference does not match reviewed evidence.")
    if layout is not None and layout != evidence["CircuitLayoutSlug"]:
        raise ValueError("Resolved layout does not match reviewed evidence.")

    for name, row in [
        ("df_test", dt.loc[mt].iloc[0]),
        ("df_overview", do.loc[mo].iloc[0]),
    ]:
        if str(row["RaceStatus"]) != "circuit_regime_ready":
            raise ValueError(f"{name}: verification requires circuit_regime_ready.")
        if str(row["Circuit"]) != str(circuit["Circuit"]):
            raise ValueError(f"{name}: circuit mismatch.")
        if str(row["CircuitLayoutSlug"]) != evidence["CircuitLayoutSlug"]:
            raise ValueError(f"{name}: layout mismatch.")

        for key in ["Length_km", "Turns", "RecordLap_sec"]:
            if not np.isclose(
                float(row[key]), float(evidence[key]), rtol=0, atol=0.001
            ):
                raise ValueError(f"{name}: {key} differs from reviewed evidence.")

        if pd.to_datetime(row["EventDateTimeUTC"], utc=True) != (
            pd.to_datetime(evidence["EventDateTimeUTC"], utc=True)
        ):
            raise ValueError(f"{name}: event timing mismatch.")

    for key in ["Length_km", "Turns", "RecordLap_sec"]:
        if not np.isclose(
            float(circuit[key]), float(evidence[key]), rtol=0, atol=0.001
        ):
            raise ValueError(f"Circuit master: {key} mismatch.")

    overview_row = do.loc[mo].iloc[0]
    for key in [
        "Pred_GBR", "Pred_XGB", "Pred_CAT",
        "PredictedFastestLap_sec", "ActualFastestLap_sec",
    ]:
        if key in overview_row.index and pd.notna(overview_row[key]):
            raise ValueError("Verification blocked after prediction/actual.")

    fields = [
        "EventIdentityStatus",
        "CircuitLayoutValidationStatus",
        "RecordLapValidationStatus",
    ]
    for frame in [dt, do]:
        missing_columns = set(fields) - set(frame.columns)
        if missing_columns:
            raise ValueError(f"Missing schema fields: {sorted(missing_columns)}")
    if "Commentaar" not in do.columns:
        raise ValueError("Overview Commentaar column missing.")

    nt, no = dt.copy(), do.copy()
    for frame, target in [(nt, mt), (no, mo)]:
        for key in fields:
            frame.loc[target, key] = "verified"

    audit_tag = f"event_verification:{year}_{gp_slug}"
    old_comment = overview_row.get("Commentaar")
    old_comment = "" if pd.isna(old_comment) else str(old_comment)

    if audit_tag not in old_comment:
        audit = {
            "tag": audit_tag,
            "VerifiedAtUTC": pd.Timestamp.now(tz="UTC").isoformat(),
            "scope": "event venue, GP layout and historical race lap record",
            "evidence": evidence,
        }
        note = json.dumps(audit, ensure_ascii=False, sort_keys=True)
        no.loc[mo, "Commentaar"] = (
            old_comment + (" | " if old_comment else "") + note
        )

    def check_gates(test, overview):
        tr = test.loc[mt].iloc[0]
        ov = overview.loc[mo].iloc[0]
        for gate in [
            validate_identity_layout(tr, ov),
            validate_recordlap_readiness(tr, ov),
            validate_event_timing(tr, ov),
        ]:
            if not gate.ok:
                raise ValueError(str(gate.detail))

    check_gates(nt, no)

    changed = not nt.equals(dt) or not no.equals(do)
    if changed:
        safe_commit_many(
            {CONFIG.test_path: nt, CONFIG.overview_path: no},
            validators=_canonical_operational_validators(),
            backup_dir=CONFIG.backup_dir,
        )

    ft = read_csv_checked(CONFIG.test_path)
    fo = read_csv_checked(CONFIG.overview_path)
    check_gates(ft, fo)

    pd.testing.assert_frame_equal(nt, ft, check_dtype=False)
    pd.testing.assert_frame_equal(no, fo, check_dtype=False)

    return {
        "ok": True,
        "race_event_id": f"{year}_{gp_slug}",
        "changed": changed,
        "RaceStatus": "circuit_regime_ready",
        **{key: "verified" for key in fields},
    }



# ACTUAL_RACE_START_V1_SANDBOX_DRAFT
NEW_TIMING_POLICY = "actual_race_start_v1"
NEW_EVENT_TIMING_FIELDS = (
    "EventDateTimeUTC", "PredictionTimingPolicyVersion",
    "AnnouncedRaceStartUTC", "RaceStartSource", "RaceStartVerifiedAtUTC",
    "R_window_start_utc", "R_window_end_utc",
    "RaceWindowDurationMinutes", "RaceWindowSource",
)

def _timing_policy(row):
    value = row.get("PredictionTimingPolicyVersion")
    return "" if value is None or pd.isna(value) else str(value).strip()

def _timing_utc(value, label):
    # Explicit timezone required: do not silently reinterpret a local time as UTC.
    if value is None or not isinstance(value, (str, pd.Timestamp)):
        raise ValueError(f"{label}: timestamp missing or invalid")
    try:
        timestamp = pd.Timestamp(value)
    except Exception as exc:
        raise ValueError(f"{label}: invalid timestamp") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise ValueError(f"{label}: explicit timezone required")
    return timestamp.tz_convert("UTC")

def _timing_start_metadata(row, *, reference_utc=None):
    if _timing_policy(row) != NEW_TIMING_POLICY:
        raise ValueError("explicit actual_race_start_v1 policy required")
    _timing_utc(row.get("EventDateTimeUTC"), "EventDateTimeUTC")
    start = _timing_utc(row.get("AnnouncedRaceStartUTC"), "AnnouncedRaceStartUTC")
    verified = _timing_utc(row.get("RaceStartVerifiedAtUTC"), "RaceStartVerifiedAtUTC")
    source = row.get("RaceStartSource")
    if source is None or pd.isna(source) or not str(source).strip():
        raise ValueError("RaceStartSource missing")
    if reference_utc is not None and verified > reference_utc:
        raise ValueError("start verification unavailable at prediction time")
    return start, verified

def validate_event_timing(test_row, overview_row):
    policy = _timing_policy(test_row)
    other_policy = _timing_policy(overview_row)
    if policy != other_policy:
        return GateResult("event_timing", False, "timing policy mismatch")
    if policy != NEW_TIMING_POLICY:
        # Never migrate or reclassify historical rows implicitly.
        if policy and policy not in {"legacy_cutoff_v1"}:
            return GateResult("event_timing", False, "unsupported explicit timing policy")
        return _legacy_validate_event_timing(test_row, overview_row)
    problems = _require_synced_fields(test_row, overview_row, NEW_EVENT_TIMING_FIELDS)
    if not problems:
        try:
            start, verified = _timing_start_metadata(test_row)
            begin = _timing_utc(test_row.get("R_window_start_utc"), "R_window_start_utc")
            end = _timing_utc(test_row.get("R_window_end_utc"), "R_window_end_utc")
            duration = float(test_row["RaceWindowDurationMinutes"])
            if not np.isfinite(duration) or duration <= 0:
                raise ValueError("RaceWindowDurationMinutes must be positive")
            if begin != start:
                raise ValueError("race window must follow announced race start")
            if end <= begin or end - begin != pd.Timedelta(minutes=duration):
                raise ValueError("race window duration mismatch")
        except (ValueError, TypeError, OverflowError) as exc:
            problems.append(str(exc))
    return GateResult("event_timing", not problems,
                      "; ".join(problems) if problems else "announced start metadata valid; final timing not yet proved")

def validate_feature_snapshot(test_row, overview_row):
    policy = _timing_policy(test_row)
    if policy != _timing_policy(overview_row):
        return GateResult("feature_snapshot", False, "timing policy mismatch")
    if policy != NEW_TIMING_POLICY:
        if policy and policy not in {"legacy_cutoff_v1"}:
            return GateResult("feature_snapshot", False, "unsupported explicit timing policy")
        return _legacy_validate_feature_snapshot(test_row, overview_row)
    problems = _require_synced_fields(test_row, overview_row, SNAPSHOT_FIELDS)
    if not problems:
        try:
            start, verified = _timing_start_metadata(test_row)
            after_q = _timing_utc(test_row.get("FeatureSnapshotAfterQualifyingUTC"), "FeatureSnapshotAfterQualifyingUTC")
            snapshot = _timing_utc(test_row.get("FeatureSnapshotTimestampUTC"), "FeatureSnapshotTimestampUTC")
            if after_q > snapshot:
                raise ValueError("after-qualifying timestamp after snapshot")
            if snapshot >= start:
                raise ValueError("snapshot must be strictly before announced start")
        except (ValueError, TypeError, OverflowError) as exc:
            problems.append(str(exc))
    return GateResult("feature_snapshot", not problems,
                      "; ".join(problems) if problems else "snapshot order valid; prediction timestamp still to be checked")

def validate_forward_prediction_timing(
    row, prediction_timestamp_utc, *, qualifying_completed_at_utc,
    latest_input_available_at_utc,
):
    """Pure proposed final-storage timing gate. No writes, clock changes or inference.

    The owning prediction writer must supply the real current storage timestamp.
    Availability/completion times must come from verified ingestion metadata.
    """
    problems = []
    try:
        prediction = _timing_utc(prediction_timestamp_utc, "PredictionTimestampUTC")
        start, verified = _timing_start_metadata(row, reference_utc=prediction)
        after_q = _timing_utc(row.get("FeatureSnapshotAfterQualifyingUTC"), "FeatureSnapshotAfterQualifyingUTC")
        snapshot = _timing_utc(row.get("FeatureSnapshotTimestampUTC"), "FeatureSnapshotTimestampUTC")
        qualifying = _timing_utc(qualifying_completed_at_utc, "qualifying completion proof")
        input_available = _timing_utc(latest_input_available_at_utc, "latest input availability proof")
        if not (qualifying <= after_q <= snapshot <= prediction):
            raise ValueError("qualifying/snapshot/prediction order invalid")
        if input_available > snapshot:
            raise ValueError("frozen input unavailable at snapshot time")
        if prediction >= start:
            raise ValueError("prediction must be strictly before announced race start")
        actual = row.get("ActualFastestLap_sec")
        if actual is not None and not pd.isna(actual) and str(actual).strip().lower() not in {"", "nan", "none", "null"}:
            raise ValueError("actual already available at prediction time")
        if str(row.get("RaceStatus", "")).strip() in {"evaluated", "completed", "finished"}:
            raise ValueError("event already finished or evaluated")
    except (ValueError, TypeError, OverflowError) as exc:
        problems.append(str(exc))
    return GateResult("forward_prediction_timing", not problems,
                      "; ".join(problems) if problems else "provisional timing PASS; actual race start evidence pending")


# ACTUAL_RACE_START_V1: intake and immutable input snapshot orchestration.
def _readiness_now_utc():
    return pd.Timestamp.now(tz="UTC")

def _forward_rows(year, gp_slug):
    dt, do = read_csv_checked(CONFIG.test_path), read_csv_checked(CONFIG.overview_path)
    mt = require_one_event(dt, year, gp_slug, dataset_name="df_test")
    mo = require_one_event(do, year, gp_slug, dataset_name="df_overview")
    return dt, do, mt, mo, dt.loc[mt].iloc[0], do.loc[mo].iloc[0]

def _assert_inputs_not_frozen(year, gp_slug):
    from .prediction import _assert_no_existing_final_prediction
    dt, do, mt, mo, rt, ro = _forward_rows(year, gp_slug)
    _assert_no_existing_final_prediction(year, gp_slug)
    for row in (rt, ro):
        if _present(row.get("FeatureSnapshotTimestampUTC")):
            raise ValueError("Inputs are frozen; forecast or start changes are audit-only")
        if str(row.get("RaceStatus", "")) in {"prediction_ready", "predicted", "evaluated", "finished", "completed"}:
            raise ValueError("Input mutation blocked at this lifecycle state")
    return dt, do, mt, mo, rt, ro

def _input_snapshot_hash(row, fields):
    # Same canonical CSV roundtrip normalization as prediction payload hashing.
    from .prediction import _timing_record_hash
    payload = {field: row.get(field) for field in fields}
    payload["TimingHashFieldsJSON"] = __import__("json").dumps(fields)
    return _timing_record_hash(payload)

def _snapshot_manifest(row):
    import json
    excluded = {"RaceStatus", "Commentaar", "CreatedAt", "UpdatedAt", "RegistryEligible",
                "FeatureSnapshotTimestampUTC", "FeatureSnapshotAfterQualifyingUTC",
                "FeatureSnapshotInputHash", "FeatureSnapshotInputFieldsJSON"}
    return sorted(str(c) for c in row.index if c not in excluded)

def validate_frozen_input_integrity(test_row, overview_row):
    import json
    problems = _require_synced_fields(test_row, overview_row, ("FeatureSnapshotInputHash", "FeatureSnapshotInputFieldsJSON"))
    if not problems:
        try:
            fields = json.loads(str(test_row["FeatureSnapshotInputFieldsJSON"]))
            if not isinstance(fields, list) or not fields or len(set(fields)) != len(fields):
                raise ValueError("Invalid snapshot field manifest")
            if any(field not in test_row for field in fields):
                raise ValueError("Frozen input column missing")
            expected = str(test_row["FeatureSnapshotInputHash"])
            if _input_snapshot_hash(test_row, fields) != expected:
                raise ValueError("Frozen feature values changed")
            # Overview may acquire output columns; compare only frozen input fields.
            if any(field not in overview_row for field in fields) or _input_snapshot_hash(overview_row, fields) != expected:
                raise ValueError("Frozen inputs differ between test and overview")
        except (ValueError, TypeError, KeyError) as exc:
            problems.append(str(exc))
    return GateResult("frozen_input_integrity", not problems,
                      "; ".join(problems) if problems else "frozen input hash valid")

def build_event_timing_values(event_datetime_utc, cutoff_hours=None, race_window_minutes=120,
                              cutoff_policy=None, race_window_source=None, *,
                              announced_race_start_utc=None, race_start_source=None,
                              race_start_verified_at_utc=None,
                              prediction_timing_policy_version="actual_race_start_v1"):
    if prediction_timing_policy_version == "legacy_cutoff_v1":
        return _legacy_build_event_timing_values(event_datetime_utc,
            cutoff_hours=3.0 if cutoff_hours is None else cutoff_hours,
            race_window_minutes=race_window_minutes, cutoff_policy=cutoff_policy,
            race_window_source=race_window_source)
    if prediction_timing_policy_version != NEW_TIMING_POLICY:
        raise ValueError("Unsupported timing policy")
    if cutoff_hours is not None or cutoff_policy is not None:
        raise ValueError("New timing policy does not accept cutoff controls")
    event = _timing_utc(event_datetime_utc, "EventDateTimeUTC")
    start = _timing_utc(announced_race_start_utc, "AnnouncedRaceStartUTC")
    verified = _timing_utc(race_start_verified_at_utc, "RaceStartVerifiedAtUTC")
    if not race_start_source or not str(race_start_source).strip():
        raise ValueError("Official race start source reference required")
    duration = float(race_window_minutes)
    if not np.isfinite(duration) or duration <= 0:
        raise ValueError("Race window duration must be positive")
    def iso(value): return value.isoformat().replace("+00:00", "Z")
    return {"EventDateTimeUTC": iso(event), "PredictionTimingPolicyVersion": NEW_TIMING_POLICY,
            "AnnouncedRaceStartUTC": iso(start), "RaceStartSource": str(race_start_source).strip(),
            "RaceStartVerifiedAtUTC": iso(verified), "R_window_start_utc": iso(start),
            "R_window_end_utc": iso(start + pd.Timedelta(minutes=duration)),
            "RaceWindowDurationMinutes": duration,
            "RaceWindowSource": str(race_window_source or "official announced race start + configured duration")}

def _commit_forward_inputs(year, gp_slug, dt, do, mt, mo, values, *, gate=None):
    new_test, new_overview = dt.copy(), do.copy()
    for field, value in values.items():
        if field not in new_test: new_test[field] = pd.Series([None] * len(new_test), index=new_test.index, dtype="object")
        if field not in new_overview: new_overview[field] = pd.Series([None] * len(new_overview), index=new_overview.index, dtype="object")
        if isinstance(value, str):
            new_test[field] = new_test[field].astype("object")
            new_overview[field] = new_overview[field].astype("object")
        new_test.loc[mt, field] = value
        new_overview.loc[mo, field] = value
    if gate:
        result = gate(new_test.loc[mt].iloc[0], new_overview.loc[mo].iloc[0])
        if not result.ok: raise ValueError(result.detail)
    def guard():
        current = _readiness_now_utc()
        start = _timing_utc(new_test.loc[mt].iloc[0].get("AnnouncedRaceStartUTC"), "AnnouncedRaceStartUTC")
        if current >= start: raise ValueError("Input commit must be strictly before announced start")
    safe_commit_many({CONFIG.test_path: new_test, CONFIG.overview_path: new_overview},
        validators=_canonical_operational_validators(), backup_dir=CONFIG.backup_dir,
        commit_guard=guard, after_commit=guard)
    return {"ok": True, **values}

@_service_storage_transaction
def save_event_timing(year, gp_slug, event_datetime_utc, *, cutoff_hours=None,
                      race_window_minutes=120, announced_race_start_utc=None,
                      race_start_source=None, race_start_verified_at_utc=None):
    dt, do, mt, mo, rt, ro = _assert_inputs_not_frozen(year, gp_slug)
    if _timing_policy(rt) not in {"", NEW_TIMING_POLICY}:
        raise ValueError("Historical timing policy cannot be migrated by race intake")
    if _present(rt.get("PredictionCutoffUTC")) and _timing_policy(rt) != NEW_TIMING_POLICY:
        raise ValueError("Existing historical cutoff timing is preserved; no implicit migration")
    current = _readiness_now_utc()
    values = build_event_timing_values(event_datetime_utc, cutoff_hours=cutoff_hours,
        race_window_minutes=race_window_minutes, announced_race_start_utc=announced_race_start_utc,
        race_start_source=race_start_source,
        race_start_verified_at_utc=current if race_start_verified_at_utc is None else race_start_verified_at_utc)
    if _timing_utc(values["RaceStartVerifiedAtUTC"], "RaceStartVerifiedAtUTC") > current:
        raise ValueError("Start verification cannot be in the future")
    if _timing_utc(values["AnnouncedRaceStartUTC"], "AnnouncedRaceStartUTC") <= current:
        raise ValueError("Announced race start must be in the future")
    return _commit_forward_inputs(year, gp_slug, dt, do, mt, mo, values, gate=validate_event_timing)

@_service_storage_transaction
def save_pre_race_input_proof(year, gp_slug, *, qualifying_completed_at_utc, qualifying_source):
    dt, do, mt, mo, rt, ro = _assert_inputs_not_frozen(year, gp_slug)
    if _timing_policy(rt) != NEW_TIMING_POLICY:
        raise ValueError("New timing policy must be saved before input proof")
    if str(rt.get("RaceStatus")) not in {"qualifying_ready", "race_weather_ready", "race_context_plausibility_ready"}:
        raise ValueError("Qualifying/input proof requires completed qualifying input stage")
    current = _readiness_now_utc()
    completion = _timing_utc(qualifying_completed_at_utc, "QualifyingCompletedAtUTC")
    if completion > current: raise ValueError("Qualifying has not finished at input intake time")
    if not qualifying_source or not str(qualifying_source).strip(): raise ValueError("Qualifying completion source required")
    values = {"QualifyingCompletedAtUTC": completion.isoformat().replace("+00:00", "Z"),
              "QualifyingCompletionSource": str(qualifying_source).strip(),
              "LatestInputAvailableAtUTC": current.isoformat().replace("+00:00", "Z"),
              "InputAvailabilityProofVersion": "observed_at_intake_v1"}
    return _commit_forward_inputs(year, gp_slug, dt, do, mt, mo, values)

@_service_storage_transaction
def freeze_feature_snapshot(year, gp_slug, now_utc=None):
    dt, do, mt, mo, rt, ro = _forward_rows(year, gp_slug)
    if _timing_policy(rt) != NEW_TIMING_POLICY:
        # Historical immutable snapshot may be read, never reconstructed or edited.
        if _present(rt.get("FeatureSnapshotTimestampUTC")):
            gate = _legacy_validate_feature_snapshot(rt, ro)
            if not gate.ok: raise ValueError(gate.detail)
            return {"ok": True, "idempotent": True, **{c: rt[c] for c in SNAPSHOT_FIELDS}, "PredictionCutoffUTC": rt.get("PredictionCutoffUTC")}
        raise ValueError("No new snapshot may be reconstructed under a historical timing policy")
    if now_utc is not None:
        raise ValueError("Explicit snapshot time override prohibited; use actual runtime clock")
    if _present(rt.get("FeatureSnapshotTimestampUTC")) or _present(ro.get("FeatureSnapshotTimestampUTC")):
        gate = validate_feature_snapshot(rt, ro)
        integrity = validate_frozen_input_integrity(rt, ro)
        if not gate.ok or not integrity.ok: raise ValueError(gate.detail + "; " + integrity.detail)
        return {"ok": True, "idempotent": True, **{c: rt[c] for c in SNAPSHOT_FIELDS},
                "AnnouncedRaceStartUTC": rt["AnnouncedRaceStartUTC"]}
    _assert_inputs_not_frozen(year, gp_slug)
    if str(rt.get("RaceStatus")) not in {"race_weather_ready", "race_context_plausibility_ready"}:
        raise ValueError("Snapshot requires race-weather/context readiness")
    for name, validator in (("qualifying", validate_qualifying), ("weather", validate_race_weather), ("timing", validate_event_timing)):
        result = validator(rt, ro)
        if not result.ok: raise ValueError(name + ": " + result.detail)
    if not _present(rt.get("QualifyingCompletionSource")) or str(rt.get("InputAvailabilityProofVersion")) != "observed_at_intake_v1":
        raise ValueError("Input intake and qualifying completion proof required")
    current = _readiness_now_utc()
    qualifying = _timing_utc(rt.get("QualifyingCompletedAtUTC"), "QualifyingCompletedAtUTC")
    available = _timing_utc(rt.get("LatestInputAvailableAtUTC"), "LatestInputAvailableAtUTC")
    if not qualifying <= available <= current: raise ValueError("Input proof/snapshot order invalid")
    fields = _snapshot_manifest(rt)
    rounded = pd.Series({field: (round(float(rt[field]), 3) if isinstance(rt[field], (float, np.floating)) and pd.notna(rt[field]) else rt[field]) for field in fields})
    values = {"FeatureSnapshotAfterQualifyingUTC": available.isoformat().replace("+00:00", "Z"),
              "FeatureSnapshotTimestampUTC": current.isoformat().replace("+00:00", "Z"),
              "FeatureSnapshotInputFieldsJSON": __import__("json").dumps(fields),
              "FeatureSnapshotInputHash": _input_snapshot_hash(rounded, fields)}
    return _commit_forward_inputs(year, gp_slug, dt, do, mt, mo, values, gate=validate_feature_snapshot)

@_service_storage_transaction
def prepare_race_context_plausibility_ready(year, gp_slug):
    """Validate completed race weather/context and commit the canonical transition."""
    dt, do, mt, mo, rt, ro = _assert_inputs_not_frozen(year, gp_slug)
    if str(rt['RaceStatus']) != 'race_weather_ready' or str(ro['RaceStatus']) != 'race_weather_ready':
        raise ValueError('Racecontext requires race_weather_ready in both datasets.')
    for name, validator in [('race_weather', validate_race_weather), ('race_context', validate_race_context_plausibility)]:
        gate = validator(rt, ro)
        if not gate.ok:
            raise ValueError(name + ': ' + gate.detail)
    transition = transition_event_status(year=int(year), gp_slug=str(gp_slug),
        expected_from='race_weather_ready', to_status='race_context_plausibility_ready')
    if not transition.ok:
        raise ValueError(str(transition))
    return {'ok': True, 'RaceStatus': 'race_context_plausibility_ready'}

# TRUSTED_CIRCUIT_MASTER_V1
def inspect_circuit_reference(year, gp_slug):
    """Read-only completeness/consistency check; no external-source certification."""
    import math
    import re
    from .event_circuits import resolve_event_circuit

    issues = []
    def issue(source, field, problem):
        issues.append({"Bron": source, "Veld": field, "Probleem": problem})

    ci = read_csv_checked(CONFIG.circuit_info_path)
    try:
        circuit, reference, layout = resolve_event_circuit(year, gp_slug, ci)
    except (ValueError, KeyError) as exc:
        issue("circuitreferentie", "gp_slug", str(exc))
        return {"ok": False, "issues": issues, "values": {}, "state": "", "completed": False}

    values = circuit.to_dict()
    values["CircuitLayoutSlug"] = layout
    positive = {"Length_km", "Turns", "RecordLap_sec", "avg_speed_kmh"}
    numeric = positive | {"Elevation_m", "Altitude_m", "Brake_intensity", "Latitude", "Longitude"}
    text = {"Circuit", "CircuitLayoutSlug", "Altitude_bucket", "Downforce_profile", "Track_type", "Direction"}
    for key in sorted(numeric | text):
        value = values.get(key)
        if not _present(value):
            issue("circuitreferentie", key, "ontbreekt; vul de centrale referentie aan")
            continue
        if key in numeric:
            try:
                number = float(value)
                valid = math.isfinite(number)
                if key in positive:
                    valid = valid and number > 0
                if key == "Turns":
                    valid = valid and number.is_integer()
                if key == "Latitude":
                    valid = valid and -90 <= number <= 90
                if key == "Longitude":
                    valid = valid and -180 <= number <= 180
                if not valid:
                    raise ValueError()
            except (ValueError, TypeError, OverflowError):
                issue("circuitreferentie", key, "ongeldige numerieke waarde")
        elif key == "CircuitLayoutSlug" and not re.fullmatch(r"[a-z0-9_]+", str(value)):
            issue("circuitreferentie", key, "gebruik kleine letters, cijfers en underscores")

    dt = read_csv_checked(CONFIG.test_path)
    do = read_csv_checked(CONFIG.overview_path)
    try:
        mt = require_one_event(dt, year, gp_slug, dataset_name="df_test")
        mo = require_one_event(do, year, gp_slug, dataset_name="df_overview")
    except ValueError as exc:
        issue("racegegevens", "Year/gp_slug", str(exc))
        return {"ok": False, "issues": issues, "values": values, "state": "", "completed": False}
    rt, ro = dt.loc[mt].iloc[0], do.loc[mo].iloc[0]
    state = str(rt.get("RaceStatus", ""))
    if state != str(ro.get("RaceStatus", "")):
        issue("racegegevens", "RaceStatus", "df_test en df_overview verschillen")
    if state == "row_created":
        issue("racegegevens", "RaceStatus", "voer eerst circuit- en regimeverrijking uit")
    elif state not in {"circuit_regime_ready", "qualifying_ready", "race_weather_ready", "race_context_plausibility_ready", "prediction_ready", "predicted", "evaluated"}:
        issue("racegegevens", "RaceStatus", "onbekende workflowstatus")

    for name, row in (("df_test", rt), ("df_overview", ro)):
        for key in sorted((numeric | text) - {"Latitude", "Longitude"}):
            if not _present(row.get(key)):
                issue(name, key, "ontbreekt in de racerij; circuitverrijking/herstel nodig")
            elif _present(values.get(key)):
                if key in numeric:
                    try:
                        equal = math.isclose(float(row[key]), float(values[key]), rel_tol=0, abs_tol=0.001)
                    except (ValueError, TypeError, OverflowError):
                        equal = False
                else:
                    equal = str(row[key]).strip() == str(values[key]).strip()
                if not equal:
                    issue(name, key, "wijkt af van de centrale circuitreferentie; expliciet herstel nodig")
        for key in ("EventIdentityStatus", "CircuitLayoutValidationStatus", "RecordLapValidationStatus"):
            status = row.get(key)
            if _present(status) and str(status) not in {"verified", "approved", "provisional_model_surrogate"}:
                issue(name, key, "bestaande reviewstatus blokkeert automatische acceptatie")
        if _present(row.get("RecordLapValidationStatus")) and str(row["RecordLapValidationStatus"]) == "provisional_model_surrogate":
            issue(name, "RecordLapValidationStatus", "voorlopige modelreferentie vereist afzonderlijke beoordeling")

    gate = validate_event_timing(rt, ro)
    if not gate.ok:
        issue("raceplanning", "starttijd / racewindow", str(gate.detail))
    completed = all(
        str(row.get(key)) == "verified"
        for row in (rt, ro)
        for key in ("EventIdentityStatus", "CircuitLayoutValidationStatus", "RecordLapValidationStatus")
    )
    return {"ok": not issues, "issues": issues, "values": values,
            "reference_slug": reference, "state": state, "completed": completed}


@_service_storage_transaction
def save_circuit_reference_validation(year, gp_slug):
    """Accept user-trusted master values; retain official event timing gates and audit method."""
    import json
    import hashlib
    report = inspect_circuit_reference(year, gp_slug)
    if not report["ok"]:
        raise ValueError(" | ".join(f"{x['Bron']}: {x['Veld']}: {x['Probleem']}" for x in report["issues"]))
    # Never append metadata to old, already-verified or progressed predictions.
    if report["completed"]:
        return {"ok": True, "changed": False, "method": "trusted_circuit_master_v1"}
    if report["state"] != "circuit_regime_ready":
        raise ValueError("Automatische circuitacceptatie vereist circuit_regime_ready; geen status terugzetten.")
    dt, do, mt, mo, rt, ro = _assert_inputs_not_frozen(year, gp_slug)
    for key in ("Pred_GBR", "Pred_XGB", "Pred_CAT", "PredictedFastestLap_sec", "ActualFastestLap_sec"):
        if _present(ro.get(key)):
            raise ValueError("Circuitacceptatie geblokkeerd na prediction/actual: " + key)
    if str(rt.get("PredictionTimingPolicyVersion")) != "actual_race_start_v1":
        raise ValueError("Nieuwe circuitacceptatie vereist actual_race_start_v1; historische policy blijft intact.")
    nt, no = dt.copy(), do.copy()
    for frame, mask in ((nt, mt), (no, mo)):
        for key in ("EventIdentityStatus", "CircuitLayoutValidationStatus", "RecordLapValidationStatus"):
            if key not in frame:
                raise ValueError(f"Schema mist {key}")
            frame[key] = frame[key].astype("object")
            frame.loc[mask, key] = "verified"

    if "Commentaar" not in no:
        raise ValueError("Overview Commentaar ontbreekt")
    audit = {"tag": f"event_verification:{int(year)}_{gp_slug}",
             "VerifiedAtUTC": pd.Timestamp.now(tz="UTC").isoformat(),
             "method": "trusted_circuit_master_v1",
             "scope": "central reference completeness and event consistency; no new external-source review",
             "reference_slug": report["reference_slug"],
             "master_sha256": hashlib.sha256(CONFIG.circuit_info_path.read_bytes()).hexdigest(),
             "values": {key: str(value) for key, value in report["values"].items()}}
    old = ro.get("Commentaar")
    old = str(old) if _present(old) else ""
    no["Commentaar"] = no["Commentaar"].astype("object")
    no.loc[mo, "Commentaar"] = old + (" | " if old else "") + json.dumps(audit, sort_keys=True, ensure_ascii=False)

    def check_gates(test, overview):
        tr, ov = test.loc[mt].iloc[0], overview.loc[mo].iloc[0]
        for gate in (validate_identity_layout(tr, ov), validate_recordlap_readiness(tr, ov), validate_event_timing(tr, ov)):
            if not gate.ok:
                raise ValueError(gate.detail)
    check_gates(nt, no)
    def readback():
        ft, fo = read_csv_checked(CONFIG.test_path), read_csv_checked(CONFIG.overview_path)
        check_gates(ft, fo)
        # Normalize expected frames using the canonical CSV reader's inference.
        from io import StringIO
        for expected, actual in ((nt, ft), (no, fo)):
            normalized = pd.read_csv(StringIO(expected.to_csv(index=False)))
            pd.testing.assert_frame_equal(normalized, actual, check_dtype=False, rtol=0, atol=1e-12)
    safe_commit_many({CONFIG.test_path: nt, CONFIG.overview_path: no},
                     validators=_canonical_operational_validators(),
                     backup_dir=CONFIG.backup_dir, after_commit=readback)
    return {"ok": True, "changed": True, "method": "trusted_circuit_master_v1"}
