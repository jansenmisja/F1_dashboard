
"""
Canonical lifecycle state machine for the F1 Fastest Lap pipeline.

This module owns RaceStatus.

No other module should directly mutate RaceStatus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from .config import (
    CONFIG,
    PRIMARY_KEY,
)

from .io import (
    read_csv_checked,
    safe_commit_many,
)

from .schemas import (
    require_one_event,
    validate_test_contract,
    validate_overview_contract,
)


# ============================================================
# Canonical lifecycle
# ============================================================

WORKFLOW_STATUS_ORDER = (
    "row_created",
    "circuit_regime_ready",
    "qualifying_ready",
    "race_weather_ready",
    "race_context_plausibility_ready",
    "prediction_ready",
    "predicted",
    "evaluated",
)

STATUS_INDEX = {
    status: idx
    for idx, status in enumerate(
        WORKFLOW_STATUS_ORDER
    )
}


# ============================================================
# Result objects
# ============================================================

@dataclass
class TransitionResult:
    ok: bool
    year: int
    gp_slug: str
    from_status: str
    to_status: str
    message: str


@dataclass
class EventState:
    year: int
    gp_slug: str

    present_in_test: bool
    present_in_overview: bool

    test_count: int
    overview_count: int

    test_status: Optional[str]
    overview_status: Optional[str]

    race_status: Optional[str]

    statuses_match: bool
    lifecycle_valid: bool

    blockers: List[str] = field(
        default_factory=list
    )

    details: Dict[str, object] = field(
        default_factory=dict
    )


# ============================================================
# Status helpers
# ============================================================

def validate_status_transition(
    current_status: str,
    requested_status: str,
) -> bool:

    if current_status not in STATUS_INDEX:
        return False

    if requested_status not in STATUS_INDEX:
        return False

    return (
        STATUS_INDEX[
            requested_status
        ]
        ==
        STATUS_INDEX[
            current_status
        ]
        + 1
    )


def expected_next_status(
    current_status: str,
) -> Optional[str]:

    if current_status not in STATUS_INDEX:
        return None

    idx = STATUS_INDEX[
        current_status
    ]

    next_idx = (
        idx + 1
    )

    if next_idx >= len(
        WORKFLOW_STATUS_ORDER
    ):
        return None

    return WORKFLOW_STATUS_ORDER[
        next_idx
    ]


# ============================================================
# Event state
# ============================================================

from .storage_transactions import transactional as _service_storage_transaction

@_service_storage_transaction
def get_event_state(
    year: int,
    gp_slug: str,
) -> EventState:

    df_test = read_csv_checked(
        CONFIG.test_path
    )

    df_overview = read_csv_checked(
        CONFIG.overview_path
    )

    test_mask = (
        pd.to_numeric(
            df_test["Year"],
            errors="coerce",
        ).eq(
            int(year)
        )
        &
        df_test["gp_slug"]
        .astype(str)
        .str.strip()
        .eq(
            str(gp_slug).strip()
        )
    )

    overview_mask = (
        pd.to_numeric(
            df_overview["Year"],
            errors="coerce",
        ).eq(
            int(year)
        )
        &
        df_overview["gp_slug"]
        .astype(str)
        .str.strip()
        .eq(
            str(gp_slug).strip()
        )
    )

    test_count = int(
        test_mask.sum()
    )

    overview_count = int(
        overview_mask.sum()
    )

    blockers = []

    test_status = None
    overview_status = None

    if test_count == 1:

        test_status = str(
            df_test.loc[
                test_mask,
                "RaceStatus",
            ].iloc[0]
        ).strip()

    elif test_count == 0:

        blockers.append(
            "Event ontbreekt in df_test."
        )

    else:

        blockers.append(
            f"df_test bevat {test_count} rows voor event."
        )

    if overview_count == 1:

        overview_status = str(
            df_overview.loc[
                overview_mask,
                "RaceStatus",
            ].iloc[0]
        ).strip()

    elif overview_count == 0:

        blockers.append(
            "Event ontbreekt in df_overview."
        )

    else:

        blockers.append(
            f"df_overview bevat {overview_count} rows voor event."
        )

    statuses_match = (
        test_count == 1
        and overview_count == 1
        and test_status == overview_status
    )

    if (
        test_count == 1
        and overview_count == 1
        and not statuses_match
    ):

        blockers.append(
            "RaceStatus mismatch tussen df_test en df_overview."
        )

    race_status = (
        test_status
        if statuses_match
        else None
    )

    lifecycle_valid = (
        race_status
        in STATUS_INDEX
    )

    if (
        race_status is not None
        and not lifecycle_valid
    ):

        blockers.append(
            f"Onbekende RaceStatus: {race_status}"
        )

    return EventState(
        year=int(year),
        gp_slug=str(gp_slug),

        present_in_test=(
            test_count == 1
        ),

        present_in_overview=(
            overview_count == 1
        ),

        test_count=test_count,
        overview_count=overview_count,

        test_status=test_status,
        overview_status=overview_status,

        race_status=race_status,

        statuses_match=statuses_match,
        lifecycle_valid=lifecycle_valid,

        blockers=blockers,

        details={
            "next_status":
                (
                    expected_next_status(
                        race_status
                    )
                    if lifecycle_valid
                    else None
                )
        },
    )


# ============================================================
# Official transition entrypoint
# ============================================================

@_service_storage_transaction
def transition_event_status(
    year: int,
    gp_slug: str,
    expected_from: str,
    to_status: str,
) -> TransitionResult:

    # --------------------------------------------------------
    # Validate requested transition
    # --------------------------------------------------------
    if expected_from not in STATUS_INDEX:

        raise ValueError(
            f"Invalid expected_from status: {expected_from}"
        )

    if to_status not in STATUS_INDEX:

        raise ValueError(
            f"Invalid to_status: {to_status}"
        )

    if not validate_status_transition(
        expected_from,
        to_status,
    ):

        raise ValueError(
            "Invalid lifecycle transition: "
            f"{expected_from} -> {to_status}"
        )

    # --------------------------------------------------------
    # Fresh read
    # --------------------------------------------------------
    df_test = read_csv_checked(
        CONFIG.test_path
    )

    df_overview = read_csv_checked(
        CONFIG.overview_path
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

    test_current = str(
        df_test.loc[
            test_mask,
            "RaceStatus",
        ].iloc[0]
    ).strip()

    overview_current = str(
        df_overview.loc[
            overview_mask,
            "RaceStatus",
        ].iloc[0]
    ).strip()

    if test_current != overview_current:

        raise ValueError(
            "RaceStatus mismatch before transition: "
            f"test={test_current}, "
            f"overview={overview_current}"
        )

    if test_current != expected_from:

        raise ValueError(
            "Unexpected current status. "
            f"Expected={expected_from}, "
            f"actual={test_current}"
        )

    # --------------------------------------------------------
    # Mutate copies only
    # --------------------------------------------------------
    new_test = (
        df_test.copy()
    )

    new_overview = (
        df_overview.copy()
    )

    new_test.loc[
        test_mask,
        "RaceStatus",
    ] = to_status

    new_overview.loc[
        overview_mask,
        "RaceStatus",
    ] = to_status

    # --------------------------------------------------------
    # Transactional synchronized commit
    # --------------------------------------------------------
    safe_commit_many(
        {
            CONFIG.test_path:
                new_test,

            CONFIG.overview_path:
                new_overview,
        },
        validators={
            CONFIG.test_path:
                lambda df:
                    validate_test_contract(
                        df,
                        strict_schema=True,
                    ),

            CONFIG.overview_path:
                validate_overview_contract,
        },
        backup_dir=(
            CONFIG.backup_dir
        ),
    )

    # --------------------------------------------------------
    # Final readback/state check
    # --------------------------------------------------------
    state = get_event_state(
        year,
        gp_slug,
    )

    if state.race_status != to_status:

        raise RuntimeError(
            "Lifecycle readback failed. "
            f"Expected={to_status}, "
            f"readback={state.race_status}"
        )

    return TransitionResult(
        ok=True,
        year=int(year),
        gp_slug=str(gp_slug),
        from_status=expected_from,
        to_status=to_status,
        message="Lifecycle transition successful",
    )
