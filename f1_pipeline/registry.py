
"""
Authoritative future-event registry.

Authoritative source:
    future_event_registry.csv

Compatibility mirror:
    future_event_registry_eligible.csv

Semantics:
- source_eligible:
    event is substantively eligible according to source logic

- registry_present:
    physical authoritative registry row exists

- RegistryEligible:
    source_eligible AND registry_present

The eligible mirror is always derived from the authoritative registry.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from .config import CONFIG
from .io import (
    read_csv_checked,
    safe_commit_csv,
    safe_commit_many,
)
from .schemas import require_one_event
from utils import helpers
from f1_pipeline.schemas import validate_row_created_contract


REGISTRY_REQUIRED_COLUMNS = (
    "Year",
    "gp_slug",
    "race_event_id",
    "source_eligible",
    "registry_present",
    "RegistryEligible",
    "EventStatus",
)

# Legacy compatibility for migrated registry validator.
REGISTRY_REQUIRED_COLS = REGISTRY_REQUIRED_COLUMNS



@dataclass
class RegistryState:
    year: int
    gp_slug: str
    race_event_id: str

    source_eligible: bool
    registry_present: bool
    registry_eligible: bool

    authoritative_count: int
    mirror_count: int


def build_race_event_id(
    year: int,
    gp_slug: str,
) -> str:

    return (
        f"{int(year)}_"
        f"{str(gp_slug).strip()}"
    )


def _bool_value(
    value,
) -> bool:

    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(int(value))

    return (
        str(value)
        .strip()
        .lower()
        in {
            "1",
            "true",
            "yes",
            "y",
            "pass",
            "eligible",
        }
    )


def _ensure_registry_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    if "Year" not in result.columns:
        result["Year"] = pd.Series(
            dtype="Int64"
        )

    if "gp_slug" not in result.columns:
        result["gp_slug"] = pd.Series(
            dtype="object"
        )

    if "race_event_id" not in result.columns:

        if not result.empty:

            result["race_event_id"] = (
                pd.to_numeric(
                    result["Year"],
                    errors="coerce",
                )
                .astype("Int64")
                .astype(str)
                + "_"
                + result["gp_slug"]
                .astype(str)
            )

        else:

            result["race_event_id"] = (
                pd.Series(
                    dtype="object"
                )
            )

    if "source_eligible" not in result.columns:

        if "RegistryEligible" in result.columns:

            result[
                "source_eligible"
            ] = (
                result[
                    "RegistryEligible"
                ]
                .map(_bool_value)
            )

        else:

            result[
                "source_eligible"
            ] = False

    if "registry_present" not in result.columns:
        result["registry_present"] = True

    if "RegistryEligible" not in result.columns:
        result["RegistryEligible"] = False

    if not result.empty:

        result[
            "source_eligible"
        ] = (
            result[
                "source_eligible"
            ]
            .map(_bool_value)
        )

        result[
            "registry_present"
        ] = True

        result[
            "RegistryEligible"
        ] = (
            result[
                "source_eligible"
            ]
            &
            result[
                "registry_present"
            ]
        )

    return result


def validate_registry(
    df: pd.DataFrame,
) -> bool:

    missing = [
        col
        for col in REGISTRY_REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing:
        return False

    if df.empty:
        return True

    duplicates = (
        df
        .duplicated(
            [
                "Year",
                "gp_slug",
            ],
            keep=False,
        )
        .any()
    )

    if duplicates:
        return False

    expected_id = (
        pd.to_numeric(
            df["Year"],
            errors="coerce",
        )
        .astype("Int64")
        .astype(str)
        + "_"
        + df["gp_slug"]
        .astype(str)
    )

    if not (
        df["race_event_id"]
        .astype(str)
        .eq(expected_id)
        .all()
    ):
        return False

    expected_eligible = (
        df["source_eligible"]
        .map(_bool_value)
        &
        df["registry_present"]
        .map(_bool_value)
    )

    actual_eligible = (
        df["RegistryEligible"]
        .map(_bool_value)
    )

    return bool(
        expected_eligible
        .eq(actual_eligible)
        .all()
    )


from .storage_transactions import transactional as _registry_storage_transaction

@_registry_storage_transaction
def bootstrap_authoritative_registry(
) -> pd.DataFrame:
    """
    Create authoritative registry once from the current
    compatibility mirror when the authoritative file is absent.

    Existing authoritative data is never overwritten here.
    """

    if CONFIG.registry_path.exists():

        df = read_csv_checked(
            CONFIG.registry_path
        )

        df = _ensure_registry_columns(
            df
        )

        if not validate_registry(df):
            raise ValueError(
                "Existing authoritative registry is invalid."
            )

        return df

    if (
        CONFIG
        .registry_eligible_mirror_path
        .exists()
    ):

        mirror = read_csv_checked(
            CONFIG.registry_eligible_mirror_path
        )

        authoritative = (
            _ensure_registry_columns(
                mirror
            )
        )

    else:

        authoritative = (
            _ensure_registry_columns(
                pd.DataFrame()
            )
        )

    safe_commit_csv(
        authoritative,
        CONFIG.registry_path,
        validator=validate_registry,
        backup_dir=CONFIG.backup_dir,
    )

    return read_csv_checked(
        CONFIG.registry_path
    )


@_registry_storage_transaction
def regenerate_eligible_mirror(
    registry: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:

    if registry is None:

        registry = (
            bootstrap_authoritative_registry()
        )

    registry = (
        _ensure_registry_columns(
            registry
        )
    )

    eligible = (
        registry.loc[
            registry[
                "RegistryEligible"
            ]
            .map(_bool_value)
        ]
        .copy()
    )

    safe_commit_csv(
        eligible,
        CONFIG.registry_eligible_mirror_path,
        validator=validate_registry,
        backup_dir=CONFIG.backup_dir,
    )

    return read_csv_checked(
        CONFIG.registry_eligible_mirror_path
    )


def _source_eligibility_from_existing_sources(
    year: int,
    gp_slug: str,
    registry: pd.DataFrame,
) -> bool:
    """
    Transitional source-eligibility adapter.

    During phase 3B we preserve current eligibility semantics
    from the existing registry/mirror.

    Phase 4 readiness will become the canonical producer of
    source_eligible.
    """

    year = int(year)
    gp_slug = str(gp_slug).strip()

    if not registry.empty:

        mask = (
            pd.to_numeric(
                registry["Year"],
                errors="coerce",
            ).eq(year)
            &
            registry["gp_slug"]
            .astype(str)
            .str.strip()
            .eq(gp_slug)
        )

        if int(mask.sum()) == 1:

            return _bool_value(
                registry.loc[
                    mask,
                    "source_eligible",
                ].iloc[0]
            )

    mirror_path = (
        CONFIG.registry_eligible_mirror_path
    )

    if mirror_path.exists():

        mirror = read_csv_checked(
            mirror_path
        )

        if (
            "Year" in mirror.columns
            and "gp_slug" in mirror.columns
        ):

            mask = (
                pd.to_numeric(
                    mirror["Year"],
                    errors="coerce",
                ).eq(year)
                &
                mirror["gp_slug"]
                .astype(str)
                .str.strip()
                .eq(gp_slug)
            )

            if int(mask.sum()) == 1:
                return True

    return False


@_registry_storage_transaction
def sync_event_registry(
    year: int,
    gp_slug: str,
    *,
    source_eligible: Optional[bool] = None,
) -> RegistryState:
    """
    Idempotently ensure an authoritative row exists.

    If source_eligible is supplied by readiness, that value is
    authoritative.

    Otherwise existing registry/mirror eligibility is preserved.
    """

    year = int(year)
    gp_slug = str(gp_slug).strip()

    # Event must exist exactly once in operational datasets.
    df_test = read_csv_checked(
        CONFIG.test_path
    )

    df_overview = read_csv_checked(
        CONFIG.overview_path
    )

    require_one_event(
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

    registry = (
        bootstrap_authoritative_registry()
    )

    registry = (
        _ensure_registry_columns(
            registry
        )
    )

    if source_eligible is None:

        source_eligible = (
            _source_eligibility_from_existing_sources(
                year,
                gp_slug,
                registry,
            )
        )

    source_eligible = bool(
        source_eligible
    )

    race_event_id = (
        build_race_event_id(
            year,
            gp_slug,
        )
    )

    mask = (
        pd.to_numeric(
            registry["Year"],
            errors="coerce",
        ).eq(year)
        &
        registry["gp_slug"]
        .astype(str)
        .str.strip()
        .eq(gp_slug)
    )

    count = int(
        mask.sum()
    )

    if count > 1:

        raise ValueError(
            "Authoritative registry contains duplicate event rows."
        )

    # Prefer existing registry schema.
    if count == 1:

        idx = registry.index[
            mask
        ][0]

        registry.loc[
            idx,
            "race_event_id",
        ] = race_event_id

        registry.loc[
            idx,
            "source_eligible",
        ] = source_eligible

        registry.loc[
            idx,
            "registry_present",
        ] = True

        registry.loc[
            idx,
            "RegistryEligible",
        ] = source_eligible

    else:

        # Build row from overview context so useful compatibility
        # fields are retained where names match registry schema.
        overview_row = (
            df_overview.loc[
                overview_mask
            ]
            .iloc[0]
        )

        row = {
            col: pd.NA
            for col in registry.columns
        }

        for col in registry.columns:

            if col in overview_row.index:
                row[col] = overview_row[col]

        row["Year"] = year
        row["gp_slug"] = gp_slug
        row["race_event_id"] = race_event_id
        row["source_eligible"] = source_eligible
        row["registry_present"] = True
        row["RegistryEligible"] = source_eligible

        registry = pd.concat(
            [
                registry,
                pd.DataFrame([row]),
            ],
            ignore_index=True,
        )

    overview_row = df_overview.loc[overview_mask].iloc[0]
    if str(overview_row.get("PredictionTimingPolicyVersion", "")).strip() == "actual_race_start_v1":
        from .readiness import validate_event_timing
        test_row = df_test.loc[require_one_event(df_test, year, gp_slug, dataset_name="df_test")].iloc[0]
        timing = validate_event_timing(test_row, overview_row)
        if not timing.ok: raise ValueError(timing.detail)
        context_fields = ("Circuit", "CircuitLayoutSlug", "RecordLap_sec", "RecordLapValidationStatus",
                          "EventIdentityStatus", "RaceStatus", "EventDateTimeUTC", "PredictionTimingPolicyVersion",
                          "AnnouncedRaceStartUTC", "RaceStartSource", "RaceStartVerifiedAtUTC")
        target_mask = registry["race_event_id"].astype(str).eq(race_event_id)
        for field in context_fields:
            if field not in registry: registry[field] = pd.Series([None] * len(registry), index=registry.index, dtype="object")
            registry.loc[target_mask, field] = overview_row.get(field)

    registry = (
        _ensure_registry_columns(
            registry
        )
    )

    if not validate_registry(
        registry
    ):
        raise ValueError(
            "Registry validation failed before commit."
        )

    # Build mirror directly from authoritative source.
    mirror = (
        registry.loc[
            registry[
                "RegistryEligible"
            ]
            .map(_bool_value)
        ]
        .copy()
    )

    # Registry + compatibility mirror are one logical transaction.
    safe_commit_many(
        {
            CONFIG.registry_path:
                registry,

            CONFIG.registry_eligible_mirror_path:
                mirror,
        },
        validators={
            CONFIG.registry_path:
                validate_registry,

            CONFIG.registry_eligible_mirror_path:
                validate_registry,
        },
        backup_dir=CONFIG.backup_dir,
    )

    # Final readback
    authoritative = (
        read_csv_checked(
            CONFIG.registry_path
        )
    )

    mirror_readback = (
        read_csv_checked(
            CONFIG.registry_eligible_mirror_path
        )
    )

    auth_mask = (
        pd.to_numeric(
            authoritative["Year"],
            errors="coerce",
        ).eq(year)
        &
        authoritative["gp_slug"]
        .astype(str)
        .str.strip()
        .eq(gp_slug)
    )

    mirror_mask = (
        pd.to_numeric(
            mirror_readback["Year"],
            errors="coerce",
        ).eq(year)
        &
        mirror_readback["gp_slug"]
        .astype(str)
        .str.strip()
        .eq(gp_slug)
    )

    auth_count = int(
        auth_mask.sum()
    )

    mirror_count = int(
        mirror_mask.sum()
    )

    if auth_count != 1:

        raise RuntimeError(
            "Registry sync readback failed."
        )

    auth_row = (
        authoritative.loc[
            auth_mask
        ]
        .iloc[0]
    )

    return RegistryState(
        year=year,
        gp_slug=gp_slug,
        race_event_id=race_event_id,

        source_eligible=_bool_value(
            auth_row[
                "source_eligible"
            ]
        ),

        registry_present=True,

        registry_eligible=_bool_value(
            auth_row[
                "RegistryEligible"
            ]
        ),

        authoritative_count=auth_count,
        mirror_count=mirror_count,
    )


@_registry_storage_transaction
def get_registry_state(
    year: int,
    gp_slug: str,
) -> RegistryState:

    year = int(year)
    gp_slug = str(gp_slug).strip()

    registry = (
        bootstrap_authoritative_registry()
    )

    registry = (
        _ensure_registry_columns(
            registry
        )
    )

    mask = (
        pd.to_numeric(
            registry["Year"],
            errors="coerce",
        ).eq(year)
        &
        registry["gp_slug"]
        .astype(str)
        .str.strip()
        .eq(gp_slug)
    )

    count = int(
        mask.sum()
    )

    source_eligible = False
    eligible = False

    if count == 1:

        row = registry.loc[
            mask
        ].iloc[0]

        source_eligible = (
            _bool_value(
                row[
                    "source_eligible"
                ]
            )
        )

        eligible = (
            _bool_value(
                row[
                    "RegistryEligible"
                ]
            )
        )

    mirror_count = 0

    if (
        CONFIG
        .registry_eligible_mirror_path
        .exists()
    ):

        mirror = read_csv_checked(
            CONFIG.registry_eligible_mirror_path
        )

        if (
            "Year" in mirror.columns
            and "gp_slug" in mirror.columns
        ):

            mirror_mask = (
                pd.to_numeric(
                    mirror["Year"],
                    errors="coerce",
                ).eq(year)
                &
                mirror["gp_slug"]
                .astype(str)
                .str.strip()
                .eq(gp_slug)
            )

            mirror_count = int(
                mirror_mask.sum()
            )

    return RegistryState(
        year=year,
        gp_slug=gp_slug,
        race_event_id=build_race_event_id(
            year,
            gp_slug,
        ),

        source_eligible=source_eligible,
        registry_present=(count == 1),
        registry_eligible=eligible,

        authoritative_count=count,
        mirror_count=mirror_count,
    )

# ============================================================
# PHASE 5C.5a — LEGACY REGISTRY ROW VALIDATOR
# ============================================================
def _registry_text_ok(value) -> bool:
    if pd.isna(value):
        return False

    value = str(value).strip()

    return (
        bool(value)
        and value.lower()
        not in {
            "nan",
            "none",
            "null",
        }
    )


def validate_registry_row(
    registry_df: pd.DataFrame,
    year: int,
    gp_slug: str,
    require_eligible: bool = False,
) -> list:

    errors = []

    if registry_df is None:
        return [
            "Registry dataframe ontbreekt."
        ]

    missing_cols = [
        col
        for col in REGISTRY_REQUIRED_COLS
        if col not in registry_df.columns
    ]

    if missing_cols:
        return [
            "Registry mist verplichte kolommen: "
            + str(missing_cols)
        ]

    race_event_id = (
        build_race_event_id(
            year,
            gp_slug,
        )
    )

    mask = (
        registry_df[
            "race_event_id"
        ]
        .astype(str)
        .eq(race_event_id)
    )

    count = int(
        mask.sum()
    )

    if count != 1:
        return [
            f"Registry target count={count}; verwacht exact 1."
        ]

    row = (
        registry_df
        .loc[mask]
        .iloc[0]
    )

    required_fields = [
        "race_event_id",
        "Year",
        "gp_slug",
        "Circuit",
        "CircuitLayoutSlug",
        "EventDateTimeUTC",
        "PredictionCutoffUTC",
        "PredictionCutoffPolicy",
        "RecordLap_sec",
        "RecordLapValidationStatus",
        "EventIdentityStatus",
        "RaceStatus",
        "RegistryEligible",
    ]
    policy = str(row.get("PredictionTimingPolicyVersion", "")).strip()
    if policy == "actual_race_start_v1":
        required_fields = [c for c in required_fields if c not in {"PredictionCutoffUTC", "PredictionCutoffPolicy"}]
        required_fields += ["PredictionTimingPolicyVersion", "AnnouncedRaceStartUTC", "RaceStartSource", "RaceStartVerifiedAtUTC"]
    for col in required_fields:

        value = row.get(col)

        if col == "RecordLap_sec":

            numeric = pd.to_numeric(
                value,
                errors="coerce",
            )

            if (
                pd.isna(numeric)
                or float(numeric) <= 0
            ):
                errors.append(
                    "Registry RecordLap_sec ongeldig."
                )

        elif col == "RegistryEligible":

            eligible = pd.to_numeric(
                value,
                errors="coerce",
            )

            if (
                pd.isna(eligible)
                or int(eligible) not in {
                    0,
                    1,
                }
            ):
                errors.append(
                    "RegistryEligible moet 0 of 1 zijn."
                )

        elif not _registry_text_ok(
            value
        ):
            errors.append(
                f"Registry {col} ontbreekt."
            )

    if require_eligible:

        eligible = pd.to_numeric(
            row.get(
                "RegistryEligible"
            ),
            errors="coerce",
        )

        if (
            pd.isna(eligible)
            or int(eligible) != 1
        ):
            errors.append(
                "Registry target is niet eligible."
            )

    return list(
        dict.fromkeys(
            errors
        )
    )

def registry_state_message(
    state: RegistryState,
) -> str:
    """
    Human-readable registry status based only on canonical RegistryState.

    This intentionally does not recreate the legacy
    RegistryEligibilityReason field.
    """

    if state.registry_eligible:
        return "Registry eligible."

    if not state.source_eligible:
        return (
            "Event is niet registry-eligible omdat de "
            "brondata nog niet aan de eligibilityvoorwaarden voldoen."
        )

    if not state.registry_present:
        return (
            "Event ontbreekt in de authoritative registry."
        )

    return (
        "Event staat in de registry maar is niet registry-eligible."
    )


@_registry_storage_transaction
def prepare_event(
    year: int,
    gp_slug: str,
    *,
    is_sprint_weekend: bool = False,
):
    """
    Canonical creation of a new operational event row.

    Creates exactly one row in df_test and df_overview with
    lifecycle status row_created.

    Existing events are never overwritten.
    """

    year = int(
        year
    )

    requested_slug = str(
        gp_slug
    ).strip()

    if not requested_slug:

        raise ValueError(
            "gp_slug mag niet leeg zijn."
        )

    # --------------------------------------------------------
    # Fresh canonical reads
    # --------------------------------------------------------
    df_test = read_csv_checked(
        CONFIG.test_path
    )

    df_overview = read_csv_checked(
        CONFIG.overview_path
    )

    circuit_info = read_csv_checked(
        CONFIG.circuit_info_path
    )

    # --------------------------------------------------------
    # Reject pre-existing target before any mutation
    # --------------------------------------------------------
    test_existing = (
        pd.to_numeric(
            df_test["Year"],
            errors="coerce",
        ).eq(
            year
        )
        &
        df_test[
            "gp_slug"
        ]
        .astype(
            str
        )
        .eq(
            requested_slug
        )
    )

    if bool(
        test_existing.any()
    ):

        raise ValueError(
            "Race bestaat al in df_test."
        )

    overview_existing = (
        pd.to_numeric(
            df_overview["Year"],
            errors="coerce",
        ).eq(
            year
        )
        &
        df_overview[
            "gp_slug"
        ]
        .astype(
            str
        )
        .eq(
            requested_slug
        )
    )

    if bool(
        overview_existing.any()
    ):

        raise ValueError(
            "Race bestaat al in df_overview."
        )

    # --------------------------------------------------------
    # Preserve existing canonical slug resolver semantics
    # --------------------------------------------------------
    canonical = (
        helpers.resolve_gp_slug(
            requested_slug,
            circuit_info,
        )
    )

    canonical = str(
        canonical
    ).strip()

    if (
        canonical
        not in set(
            circuit_info[
                "gp_slug"
            ]
            .astype(
                str
            )
        )
    ):

        raise ValueError(
            "Canonieke gp_slug bestaat niet "
            "in circuit_info.csv."
        )

    # Re-check duplicates on canonical key because an alias may
    # resolve to an existing canonical event.
    canonical_test_existing = (
        pd.to_numeric(
            df_test["Year"],
            errors="coerce",
        ).eq(
            year
        )
        &
        df_test[
            "gp_slug"
        ]
        .astype(
            str
        )
        .eq(
            canonical
        )
    )

    if bool(
        canonical_test_existing.any()
    ):

        raise ValueError(
            "Race bestaat al in df_test."
        )

    canonical_overview_existing = (
        pd.to_numeric(
            df_overview["Year"],
            errors="coerce",
        ).eq(
            year
        )
        &
        df_overview[
            "gp_slug"
        ]
        .astype(
            str
        )
        .eq(
            canonical
        )
    )

    if bool(
        canonical_overview_existing.any()
    ):

        raise ValueError(
            "Race bestaat al in df_overview."
        )

    # --------------------------------------------------------
    # Resolve circuit row
    # --------------------------------------------------------
    # EVENT_CIRCUIT_SELECTION_V1
    from .event_circuits import resolve_event_circuit
    circuit_row, circuit_reference_slug, event_layout_slug = (
        resolve_event_circuit(year, canonical, circuit_info)
    )

    circuit_name = (
        circuit_row.get(
            "Circuit"
        )
    )

    if (
        pd.isna(
            circuit_name
        )
        or not str(
            circuit_name
        ).strip()
    ):

        raise ValueError(
            "Circuit ontbreekt in circuit_info."
        )

    circuit_name = str(
        circuit_name
    ).strip()

    # --------------------------------------------------------
    # Preserve legacy race-number semantics
    # --------------------------------------------------------
    race_col = (
        "Race #"
        if "Race #"
        in df_overview.columns
        else "RaceNumber"
    )

    if (
        race_col
        in df_overview.columns
        and pd.to_numeric(
            df_overview[
                race_col
            ],
            errors="coerce",
        )
        .notna()
        .any()
    ):

        next_race = (
            int(
                pd.to_numeric(
                    df_overview[
                        race_col
                    ],
                    errors="coerce",
                ).max()
            )
            + 1
        )

    else:

        next_race = 1

    # --------------------------------------------------------
    # Build row-created payloads
    # --------------------------------------------------------
    base_test = {
        "Year":
            year,

        "gp_slug":
            canonical,

        "Circuit":
            circuit_name,

        "IsSprintWeekend":
            int(
                bool(
                    is_sprint_weekend
                )
            ),

        "RaceStatus":
            "row_created",

        "EventStatus":
            "scheduled",
    }

    base_overview = {
        race_col:
            next_race,

        "Year":
            year,

        "gp_slug":
            canonical,

        "Circuit":
            circuit_name,

        "IsSprintWeekend":
            int(
                bool(
                    is_sprint_weekend
                )
            ),

        "RaceStatus":
            "row_created",

        "EventStatus":
            "scheduled",

        "Commentaar":
            (
                "Nieuwe GP aangemaakt via Streamlit; "
                "vervolgstap circuit/regimeverrijking."
            ),
    }

    # --------------------------------------------------------
    # Add rows while preserving current schema
    # --------------------------------------------------------
    if event_layout_slug is not None:
        base_test["CircuitLayoutSlug"] = event_layout_slug
        base_overview["CircuitLayoutSlug"] = event_layout_slug

    new_test = (
        df_test.copy()
    )

    for col in base_test:

        if col not in new_test.columns:
            new_test[col] = pd.NA

    new_test = pd.concat(
        [
            new_test,
            pd.DataFrame(
                [
                    {
                        col:
                            base_test.get(
                                col,
                                pd.NA,
                            )
                        for col
                        in new_test.columns
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    new_overview = (
        df_overview.copy()
    )

    for col in base_overview:

        if col not in new_overview.columns:
            new_overview[col] = pd.NA

    new_overview = pd.concat(
        [
            new_overview,
            pd.DataFrame(
                [
                    {
                        col:
                            base_overview.get(
                                col,
                                pd.NA,
                            )
                        for col
                        in new_overview.columns
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Canonical row-created contract
    # --------------------------------------------------------
    contract = (
        validate_row_created_contract(
            new_test,
            new_overview,
            year,
            canonical,
        )
    )

    if not contract.ok:

        raise ValueError(
            "row_created contract faalt: "
            + " | ".join(
                contract.errors
            )
        )

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
        backup_dir=(
            CONFIG.backup_dir
        ),
    )

    # --------------------------------------------------------
    # Final readback
    # --------------------------------------------------------
    final_test = read_csv_checked(
        CONFIG.test_path
    )

    final_overview = read_csv_checked(
        CONFIG.overview_path
    )

    final_contract = (
        validate_row_created_contract(
            final_test,
            final_overview,
            year,
            canonical,
        )
    )

    if not final_contract.ok:

        raise RuntimeError(
            "row_created readback contract faalt: "
            + " | ".join(
                final_contract.errors
            )
        )

    test_mask = require_one_event(
        final_test,
        year,
        canonical,
        dataset_name="df_test",
    )

    overview_mask = require_one_event(
        final_overview,
        year,
        canonical,
        dataset_name="df_overview",
    )

    test_status = str(
        final_test.loc[
            test_mask,
            "RaceStatus",
        ].iloc[0]
    ).strip()

    overview_status = str(
        final_overview.loc[
            overview_mask,
            "RaceStatus",
        ].iloc[0]
    ).strip()

    if (
        test_status != "row_created"
        or overview_status
        != "row_created"
    ):

        raise RuntimeError(
            "prepare_event readback RaceStatus "
            "is niet row_created."
        )

    return {
        "ok":
            True,

        "Year":
            year,

        "gp_slug":
            canonical,

        "Circuit":
            circuit_name,

        "IsSprintWeekend":
            int(
                bool(
                    is_sprint_weekend
                )
            ),

        "RaceStatus":
            "row_created",

        "EventStatus":
            "scheduled",

        "race_column":
            race_col,

        "race_number":
            next_race,
    }

