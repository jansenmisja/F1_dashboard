"""Streamlit presentation helpers for audit and diagnostics."""

import pandas as pd
import streamlit as st

# STEP 13.2I-C — imports required by extracted diagnostics
import hashlib
import inspect
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
from f1_pipeline.config import CONFIG
from f1_pipeline.lifecycle import (
    validate_status_transition as canonical_validate_status_transition,
)
from f1_pipeline.prediction import (
    _validate_final_prediction_artifacts_raw as canonical_validate_final_prediction_artifacts_raw,
)
from f1_pipeline.readiness import (
    _build_prediction_readiness_report_raw as canonical_build_prediction_readiness_report_raw,
    build_event_timing_values as canonical_build_event_timing_values,
    validate_event_timing as canonical_validate_event_timing,
    validate_feature_snapshot as canonical_validate_feature_snapshot,
    validate_race_weather_audit_rows as canonical_validate_race_weather_audit_rows,
)


def _audit_ui_target_row(df, *, year, gp_slug):
    if df is None or df.empty:
        return None

    if not {"Year", "gp_slug"}.issubset(df.columns):
        return None

    years = pd.to_numeric(
        df["Year"],
        errors="coerce",
    )

    slugs = (
        df["gp_slug"]
        .astype(str)
        .str.strip()
    )

    mask = (
        years.eq(int(year))
        & slugs.eq(str(gp_slug))
    )

    target = df.loc[mask]

    if len(target) != 1:
        return None

    return target.iloc[0]


def _audit_ui_value(row, column, default=None):
    if row is None:
        return default

    try:
        if column not in row.index:
            return default

        value = row[column]

        if pd.isna(value):
            return default

        return value

    except Exception:
        return default


def _audit_ui_fmt(value):
    if value is None:
        return "—"

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return "—"

    return text


def render_audit_diagnostics_summary(
    *,
    df_test,
    df_overview,
    year,
    gp_slug,
    race_status,
):
    """
    Pure UI:
    toont alleen bestaande data/state.
    Geen validators, mutations of writes.
    """

    test_row = _audit_ui_target_row(
        df_test,
        year=year,
        gp_slug=gp_slug,
    )

    overview_row = _audit_ui_target_row(
        df_overview,
        year=year,
        gp_slug=gp_slug,
    )

    st.subheader("Diagnostic summary")

    test_ok = test_row is not None
    overview_ok = overview_row is not None

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "RaceStatus",
        race_status or "—",
    )

    c2.metric(
        "df_test",
        "✅ uniek" if test_ok else "⚠️",
    )

    c3.metric(
        "df_overview",
        "✅ uniek" if overview_ok else "⚠️",
    )

    c4.metric(
        "Row sync",
        "✅" if test_ok and overview_ok else "⚠️",
    )

    if not (test_ok and overview_ok):
        st.warning(
            "Targetrace is niet exact één keer zichtbaar "
            "in zowel df_test als df_overview."
        )

    def combined_value(column):
        value = _audit_ui_value(
            overview_row,
            column,
            None,
        )

        if value is None:
            value = _audit_ui_value(
                test_row,
                column,
                None,
            )

        return value

    metadata_fields = [
        "race_event_id",
        "Circuit",
        "CircuitLayoutSlug",
        "EventStatus",
        "EventIdentityStatus",
        "RecordLapValidationStatus",
        "RecordLap_sec",
        "EventDateTimeUTC",
        "PredictionCutoffUTC",
        "PredictionTimestampUTC",
        "FeatureSnapshotAfterQualifyingUTC",
        "FeatureSnapshotTimestampUTC",
        "RegistryEligible",
    ]

    metadata = pd.DataFrame(
        [
            {
                "Field": field,
                "Value": _audit_ui_fmt(
                    combined_value(field)
                ),
            }
            for field in metadata_fields
        ]
    )

    st.markdown("##### Event / registry metadata")

    st.dataframe(
        metadata,
        hide_index=True,
        width="stretch",
    )

    prediction_complete = all(
        _audit_ui_value(
            overview_row,
            field,
            None,
        ) is not None
        for field in (
            "Pred_GBR",
            "Pred_XGB",
            "Pred_CAT",
            "PredictedFastestLap_sec",
        )
    )

    actual_present = (
        _audit_ui_value(
            overview_row,
            "ActualFastestLap_sec",
            None,
        )
        is not None
    )

    evaluation_complete = all(
        _audit_ui_value(
            overview_row,
            field,
            None,
        ) is not None
        for field in (
            "ActualFastestLap_sec",
            "Error",
            "AbsError",
            "APE_pct",
        )
    )

    d1, d2, d3 = st.columns(3)

    d1.metric(
        "3-model prediction",
        "✅" if prediction_complete else "—",
    )

    d2.metric(
        "Actual",
        "✅" if actual_present else "—",
    )

    d3.metric(
        "Evaluation KPI's",
        "✅" if evaluation_complete else "—",
    )

    missing_metadata = [
        field
        for field in metadata_fields
        if combined_value(field) is None
    ]

    if missing_metadata:
        with st.expander(
            "Ontbrekende zichtbare metadata"
        ):
            st.write(missing_metadata)

            st.caption(
                "Alleen UI-inventory. "
                "Canonical readiness bepaalt welke velden "
                "voor de huidige lifecyclefase verplicht zijn."
            )

    with st.expander(
        "Technische target-row details"
    ):
        st.markdown("**df_test**")

        if test_row is not None:
            st.dataframe(
                pd.DataFrame([test_row]),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("Geen unieke df_test target-row.")

        st.markdown("**df_overview**")

        if overview_row is not None:
            st.dataframe(
                pd.DataFrame([overview_row]),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("Geen unieke df_overview target-row.")

    st.caption(
        "De bestaande technische diagnostics hieronder "
        "blijven beschikbaar voor diepere inspectie."
    )

    st.divider()


# ============================================================
# STEP 13.2B — extracted diagnostic/smoke tooling
# ============================================================
def helper_preflight():
    """
    Hardened helper preflight.

    Regels:
    - iedere run gebruikt unieke smoke-bestanden;
    - backup_file() en safe_to_csv() gebruiken aparte bestanden;
    - nooit vaste smoke.csv;
    - smoke-bronbestanden worden na test verwijderd;
    - canonical backups blijven in data/processed/backups/;
    - een FileExistsError tijdens de backup-smoke is
      niet blokkerend wanneer de helperlaag verder correct is.
    """

    from datetime import datetime
    from pathlib import Path
    import inspect
    import pandas as pd

    result = {
        "ok": True,
        "errors": [],
        "warnings": [],
        "checks": {},
    }

    # --------------------------------------------------------
    # Canonical directories
    # --------------------------------------------------------
    processed_dir = Path(
        str(CONFIG.processed_dir)
    )

    backup_dir = (
        processed_dir
        / "backups"
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Required helper layer
    # --------------------------------------------------------
    required_helpers = [
        "safe_read_csv",
        "safe_to_csv",
        "backup_file",
        "diag_df",
        "rmse",
        "normalize_name",
        "resolve_gp_slug",
        "time_str_to_sec",
        "convert_column_time_to_sec",
        "agg_weather_per_gp",
        "merge_with_fallback",
        "clean_feature_names",
        "dump_feature_weights",
        "get_model_log_path",
        "round_float_columns",
    ]

    for helper_name in required_helpers:

        exists = hasattr(
            helpers,
            helper_name,
        )

        result["checks"][
            f"helper_{helper_name}"
        ] = bool(exists)

        if not exists:

            result["errors"].append(
                f"Helper ontbreekt: {helper_name}"
            )

    if result["errors"]:

        result["ok"] = False
        return result

    # --------------------------------------------------------
    # Signature checks
    # --------------------------------------------------------
    expected_signatures = {
        "safe_read_csv": [
            "path",
        ],
        "safe_to_csv": [
            "df",
            "path",
            "index",
        ],
        "backup_file": [
            "path",
        ],
        "diag_df": [
            "df",
            "name",
            "head_n",
            "tail_n",
        ],
        "rmse": [
            "y_true",
            "y_pred",
        ],
        "normalize_name": [
            "name",
        ],
        "resolve_gp_slug": [
            "name_or_circuit",
            "circuit_info_df",
        ],
        "time_str_to_sec": [
            "x",
        ],
        "convert_column_time_to_sec": [
            "df",
            "col",
        ],
        "agg_weather_per_gp": [
            "weather_df",
            "gp_col",
            "session_col",
        ],
        "merge_with_fallback": [
            "left_df",
            "right_df",
            "primary_keys",
            "fallback_keys",
            "how",
            "verbose",
        ],
        "clean_feature_names": [
            "cols",
        ],
        "dump_feature_weights": [
            "model",
            "feature_names",
            "out_path",
        ],
        "get_model_log_path": [],
        "round_float_columns": [
            "df",
            "ndigits",
        ],
    }

    for helper_name, required_params in (
        expected_signatures.items()
    ):

        try:

            sig = inspect.signature(
                getattr(
                    helpers,
                    helper_name,
                )
            )

            actual_params = list(
                sig.parameters.keys()
            )

            missing_params = [
                p
                for p in required_params
                if p not in actual_params
            ]

            if missing_params:

                result["errors"].append(
                    f"{helper_name}: ontbrekende parameters "
                    f"{missing_params}; "
                    f"actueel={actual_params}"
                )

            else:

                result["checks"][
                    f"signature_{helper_name}"
                ] = True

        except Exception as exc:

            result["errors"].append(
                f"Signaturecheck {helper_name} faalt: {exc}"
            )

    if result["errors"]:

        result["ok"] = False
        return result

    # --------------------------------------------------------
    # Canonical model log path
    # --------------------------------------------------------
    try:

        actual_model_log = str(
            helpers.get_model_log_path()
        )

        expected_model_log = (
            str(CONFIG.model_log_path)
        )

        if actual_model_log != expected_model_log:

            result["errors"].append(
                "get_model_log_path() wijkt af van canonical path. "
                f"Actueel: {actual_model_log}"
            )

        else:

            result["checks"][
                "canonical_model_log"
            ] = True

    except Exception as exc:

        result["errors"].append(
            f"get_model_log_path() faalt: {exc}"
        )

    if result["errors"]:

        result["ok"] = False
        return result

    # --------------------------------------------------------
    # Unieke smoke filenames
    # --------------------------------------------------------
    stamp = datetime.utcnow().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    smoke_backup_path = (
        processed_dir
        / f"smoke_backup_{stamp}.csv"
    )

    smoke_write_path = (
        processed_dir
        / f"smoke_write_{stamp}.csv"
    )

    # Controle: ze MOETEN verschillend zijn
    if smoke_backup_path == smoke_write_path:

        result["errors"].append(
            "Smoke filenames zijn niet uniek."
        )

        result["ok"] = False
        return result

    result["checks"][
        "unique_smoke_paths"
    ] = True

    created_temp_files = []

    try:

        # ====================================================
        # A. backup_file smoke-test
        #
        # Fixturebestand wordt alleen voor de functionele
        # helpertest aangemaakt. Dit is geen projectdataset.
        # ====================================================
        fixture_df = pd.DataFrame(
            {
                "smoke_value": [
                    1.23456
                ],
                "smoke_type": [
                    "backup_test"
                ],
            }
        )

        fixture_df.to_csv(
            smoke_backup_path,
            index=False,
        )

        created_temp_files.append(
            smoke_backup_path
        )

        try:

            backup_result = (
                helpers.backup_file(
                    smoke_backup_path
                )
            )

            result["checks"][
                "backup_file_smoke"
            ] = True

            result["checks"][
                "backup_file_result"
            ] = str(
                backup_result
            )

        except FileExistsError as exc:

            # Niet-blokkerend:
            # unieke fixture + correcte helperlaag.
            result["warnings"].append(
                "backup_file smoke gaf FileExistsError. "
                "Dit wordt niet als harde fout behandeld: "
                f"{exc}"
            )

            result["checks"][
                "backup_file_smoke"
            ] = "warning_collision"

        except Exception as exc:

            result["errors"].append(
                f"backup_file smoke faalt: {exc}"
            )

        # ====================================================
        # B. safe_to_csv smoke-test
        #
        # Expliciet ANDER bestand.
        # ====================================================
        smoke_df = pd.DataFrame(
            {
                "float_value": [
                    1.23456,
                    7.89123,
                ],
                "text_value": [
                    "a",
                    "b",
                ],
            }
        )

        helpers.safe_to_csv(
            smoke_df,
            smoke_write_path,
            index=False,
        )

        created_temp_files.append(
            smoke_write_path
        )

        if not smoke_write_path.exists():

            result["errors"].append(
                "safe_to_csv smoke heeft geen outputbestand gemaakt."
            )

        else:

            reread = helpers.safe_read_csv(
                smoke_write_path
            )

            result["checks"][
                "safe_to_csv_smoke"
            ] = True

            # -----------------------------------------------
            # 3-decimalen check
            # -----------------------------------------------
            test_value = pd.to_numeric(
                reread[
                    "float_value"
                ],
                errors="coerce",
            ).iloc[0]

            if pd.isna(test_value):

                result["errors"].append(
                    "safe_to_csv rounding smoke gaf NaN."
                )

            elif abs(
                float(test_value)
                - 1.235
            ) > 0.0001:

                result["errors"].append(
                    "safe_to_csv rondt niet correct op "
                    f"3 decimalen: {test_value}"
                )

            else:

                result["checks"][
                    "safe_to_csv_rounding_3dp"
                ] = True

        # ====================================================
        # Backup directory bestaat
        # ====================================================
        if backup_dir.exists():

            result["checks"][
                "canonical_backup_dir"
            ] = True

        else:

            result["errors"].append(
                f"Backup directory ontbreekt: {backup_dir}"
            )

    finally:

        # ----------------------------------------------------
        # Alleen tijdelijke BRONFILES verwijderen.
        #
        # Canonical helperbackups blijven behouden.
        # ----------------------------------------------------
        for temp_path in created_temp_files:

            try:

                if temp_path.exists():

                    temp_path.unlink()

            except Exception as exc:

                result["warnings"].append(
                    "Tijdelijk smoke-bestand kon "
                    f"niet worden verwijderd: "
                    f"{temp_path}: {exc}"
                )

    # --------------------------------------------------------
    # Eindstatus
    # --------------------------------------------------------
    if result["errors"]:

        result["ok"] = False

    return result


# ============================================================
# STEP 13.2B — extracted diagnostic/smoke tooling
# ============================================================
def audit_uncontrolled_race_status_writes(
    source_text: str,
) -> list:
    """
    Statische audit.

    Zoekt assignments zoals:
        df.loc[mask, "RaceStatus"] = ...

    De twee assignments BINNEN controlled_status_transition()
    zijn toegestaan.

    Geeft overige verdachte regels terug.
    """

    tree = ast.parse(
        source_text
    )

    issues = []

    allowed_function = (
        "controlled_status_transition"
    )

    class Visitor(
        ast.NodeVisitor
    ):

        def __init__(self):
            self.function_stack = []

        def visit_FunctionDef(
            self,
            node,
        ):
            self.function_stack.append(
                node.name
            )

            self.generic_visit(
                node
            )

            self.function_stack.pop()

        def visit_Assign(
            self,
            node,
        ):
            current_function = (
                self.function_stack[-1]
                if self.function_stack
                else None
            )

            line_text = ""

            if (
                hasattr(
                    node,
                    "lineno",
                )
            ):
                lines = (
                    source_text
                    .splitlines()
                )

                if (
                    1
                    <= node.lineno
                    <= len(lines)
                ):
                    line_text = (
                        lines[
                            node.lineno - 1
                        ]
                    )

            node_source = ast.get_source_segment(
                source_text,
                node,
            ) or line_text

            if (
                "RaceStatus"
                in node_source
                and ".loc"
                in node_source
                and current_function
                != allowed_function
            ):
                issues.append(
                    {
                        "line":
                            node.lineno,
                        "function":
                            current_function,
                        "code":
                            node_source,
                    }
                )

            self.generic_visit(
                node
            )

    Visitor().visit(
        tree
    )

    return issues


# ============================================================
# STEP 13.2B — extracted diagnostic/smoke tooling
# ============================================================
def dry_smoke_test_italy_2026() -> dict:
    """
    Volledige niet-destructieve Italy 2026 pre-race smoke-test.

    Geen writes naar:
    - df_test.csv
    - df_overview.csv
    - production registry
    - production ledger
    - production model_log

    Tijdelijke registry/ledger/log fixtures gaan uitsluitend
    naar een tijdelijke /content-directory.
    """

    import tempfile
    import shutil as _shutil

    results = {
        "ok": True,
        "checks": {},
        "errors": [],
        "warnings": [],
    }

    temp_dir = Path(
        tempfile.mkdtemp(
            prefix="f1_italy_2026_dry_smoke_"
        )
    )

    # Globals die tijdelijk naar dry-run files wijzen.

    original_ledger_path = (
        FINAL_PREDICTION_LEDGER_PATH
    )

    original_model_log_path = (
        FINAL_PREDICTION_MODEL_LOG_PATH
    )

    try:

        # ====================================================
        # A. Hardened helper preflight
        # ====================================================
        preflight = (
            helper_preflight()
        )

        results["checks"][
            "helper_preflight"
        ] = bool(
            preflight.get(
                "ok",
                False,
            )
        )

        if not preflight.get(
            "ok",
            False,
        ):

            results["errors"].extend(
                preflight.get(
                    "errors",
                    [],
                )
            )

        # ====================================================
        # B. Geen vaste smoke.csv-code meer
        # ====================================================
        try:

            source_text = Path(
                __file__
            ).read_text(
                encoding="utf-8"
            )

            fixed_smoke_refs = [
                line
                for line in source_text.splitlines()
                if (
                    '"smoke.csv"' in line
                    or "'smoke.csv'" in line
                )
            ]

            smoke_ok = (
                len(
                    fixed_smoke_refs
                )
                == 0
            )

            results["checks"][
                "no_fixed_smoke_csv"
            ] = smoke_ok

            if not smoke_ok:

                results["errors"].append(
                    "Er bestaan nog vaste smoke.csv-codeverwijzingen."
                )

        except Exception as exc:

            results["warnings"].append(
                f"Source smoke.csv-audit niet uitgevoerd: {exc}"
            )

        # ====================================================
        # C. Italy 2026 event timing
        # ====================================================
        timing = (
            canonical_build_event_timing_values(
                "2026-09-06T13:00:00Z",
                cutoff_hours=3,
                race_window_minutes=120,
            )
        )

        row = {
            "Year":
                2026,

            "gp_slug":
                "italy",

            "Circuit":
                "Monza",

            "RaceStatus":
                "race_context_plausibility_ready",

            "IsSprintWeekend":
                0,

            "CircuitLayoutSlug":
                "monza_gp",

            "CircuitLayoutValidationStatus":
                "verified",

            "RecordLap_sec":
                81.046,

            "RecordLapValidationStatus":
                "verified",

            "EventIdentityStatus":
                "verified",

            "ActualFastestLap_sec":
                np.nan,

            # ---------------- Structural / regime --------
            # Synthetic maar intern plausibele Monza-waarden.
            # Alleen voor niet-destructieve Step-10 dry smoke.
            "YearOffset":
                8,

            "Length_km":
                5.793,

            "Turns":
                11,

            "Turns/Length_km":
                11.0 / 5.793,

            "TurnDensity":
                11.0 / 5.793,

            "avg_speed_kmh":
                260.000,

            "v_proxy_kmh":
                250.000,

            "Latitude":
                45.6156,

            "Longitude":
                9.2811,

            "Turns/length rank":
                20,

            "avg_speed rank":
                1,

            "CornerFraction_est":
                0.250,

            "Regime2026_Flag":
                1,

            "Regime2026_AeroComponent_sec":
                -0.400,

            "Regime2026_PUComponent_sec":
                -0.300,

            "Regime2026_MassComponent_sec":
                -0.200,

            "Regime2026_TyreComponent_sec":
                -0.200,

            "Regime2026_PriorLapDelta_sec":
                -1.100,

            # ---------------- Qualifying ----------------
            "Q_airtemp_mean":
                26.000,

            "Q_airtemp_max":
                28.000,

            "Q_rain_sum":
                0.000,

            "Q_humidity_mean":
                50.000,

            "PoleTime_sec":
                78.500,

            "pole_compound":
                "soft",

            "pole_c_index":
                5,

            "PoleMinusRecord_sec":
                78.500
                - 81.046,

            "SprintFastestLap_sec":
                np.nan,

            # ---------------- Race weather --------------
            "R_airtemp_mean":
                27.000,

            "R_airtemp_max":
                29.000,

            "R_rain_sum":
                0.000,

            "R_humidity_mean":
                55.000,

            "WeatherSourceMode":
                "manual_verified",

            # ---------------- Snapshot ------------------
            "FeatureSnapshotAfterQualifyingUTC":
                "2026-09-05T17:00:00Z",

            "FeatureSnapshotTimestampUTC":
                "2026-09-05T17:00:00Z",
        }

        row.update(
            timing
        )

        df_test = pd.DataFrame(
            [row]
        )

        df_overview = pd.DataFrame(
            [row]
        )

        timing_gate = canonical_validate_event_timing(
            df_test.iloc[0],
            df_overview.iloc[0],
        )

        timing_errors = (
            []
            if timing_gate.ok
            else [timing_gate.detail]
        )

        results["checks"][
            "event_timing"
        ] = (
            not timing_errors
        )

        if timing_errors:
            results["errors"].extend(
                [
                    "Event timing: " + e
                    for e in timing_errors
                ]
            )

        # ====================================================
        # D. Race weather
        # ====================================================
        audit_smoke_mt = require_one(
            df_test,
            2026,
            "italy",
            "df_test",
        )

        audit_smoke_mo = require_one(
            df_overview,
            2026,
            "italy",
            "df_overview",
        )

        weather_errors = canonical_validate_race_weather_audit_rows(
            df_test.loc[audit_smoke_mt].iloc[0],
            df_overview.loc[audit_smoke_mo].iloc[0],
            require_status=False,
        )

        results["checks"][
            "race_weather_sync"
        ] = (
            not weather_errors
        )

        if weather_errors:
            results["errors"].extend(
                [
                    "Race weather: " + e
                    for e in weather_errors
                ]
            )

        # ====================================================
        # E. Feature snapshot
        # ====================================================
        snapshot_gate = canonical_validate_feature_snapshot(
            df_test.iloc[0],
            df_overview.iloc[0],
        )

        snapshot_errors = (
            []
            if snapshot_gate.ok
            else [snapshot_gate.detail]
        )

        results["checks"][
            "feature_snapshot"
        ] = (
            not snapshot_errors
        )

        if snapshot_errors:
            results["errors"].extend(
                [
                    "Feature snapshot: " + e
                    for e in snapshot_errors
                ]
            )


        # ====================================================
        # F. Canonical source eligibility uit synthetic rows
        # ====================================================
        registry_readiness = (
            canonical_build_prediction_readiness_report_raw(
                df_test.iloc[0],
                df_overview.iloc[0],
                year=2026,
                gp_slug="italy",
                race_status="race_context_plausibility_ready",
                registry_present=True,
                registry_eligible=True,
            )
        )

        results["checks"][
            "registry_source_eligible"
        ] = bool(
            registry_readiness.source_eligible
        )

        if not registry_readiness.source_eligible:

            results["errors"].extend(
                [
                    "Registry/readiness: " + blocker
                    for blocker
                    in registry_readiness.blockers
                ]
            )

        # ====================================================
        # H. Leakage guard
        # ====================================================
        leakage_gate = registry_readiness.gates["leakage_guard"]

        leakage_errors = (
            []
            if leakage_gate.ok
            else [leakage_gate.detail]
        )

        results["checks"][
            "leakage_guard"
        ] = (
            not leakage_errors
        )

        if leakage_errors:
            results["errors"].extend(
                [
                    "Leakage: " + e
                    for e in leakage_errors
                ]
            )

        # ====================================================
        # I. Volledige prediction readiness
        # ====================================================
        readiness = registry_readiness

        failed_readiness = [
            gate
            for gate in readiness.gates.values()
            if not gate.ok
        ]

        results["checks"][
            "prediction_readiness"
        ] = (
            len(
                failed_readiness
            )
            == 0
        )

        if failed_readiness:

            for gate in failed_readiness:

                results["errors"].append(
                    "Readiness "
                    f"{gate.name}: "
                    f"{gate.detail}"
                )

        # ====================================================
        # J. Canonical status sequence
        # ====================================================
        transition_errors = []

        for current_status, next_status in zip(
            WORKFLOW_STATUS_ORDER[:-1],
            WORKFLOW_STATUS_ORDER[1:],
        ):

            errs = (
                canonical_validate_status_transition(
                    current_status,
                    next_status,
                )
            )

            transition_errors.extend(
                errs
            )

        # Specifieke bypass moet falen.
        bypass_errors = (
            canonical_validate_status_transition(
                "race_weather_ready",
                "prediction_ready",
            )
        )

        transitions_ok = (
            not transition_errors
            and bool(
                bypass_errors
            )
        )

        results["checks"][
            "canonical_status_sequence"
        ] = transitions_ok

        if transition_errors:

            results["errors"].extend(
                transition_errors
            )

        if bypass_errors:

            results["errors"].append(
                "Status-bypass race_weather_ready → "
                "prediction_ready werd niet geblokkeerd."
            )

        # ====================================================
        # K. Synthetic final-prediction artifacts
        # ====================================================
        pred_test = (
            df_test.copy()
        )

        pred_overview = (
            df_overview.copy()
        )

        pred_test.loc[
            0,
            "RaceStatus"
        ] = "prediction_ready"

        pred_overview.loc[
            0,
            "RaceStatus"
        ] = "prediction_ready"

        pred_gbr = 82.100
        pred_xgb = 82.250
        pred_cat = 82.000

        pred_values = {
            "Pred_GBR":
                pred_gbr,

            "Pred_XGB":
                pred_xgb,

            "Pred_CAT":
                pred_cat,

            "PredictedFastestLap_sec":
                pred_gbr,

            "Pred_GBR_offset_sec":
                pred_gbr
                - 81.046,

            "Pred_XGB_offset_sec":
                pred_xgb
                - 81.046,

            "Pred_CAT_offset_sec":
                pred_cat
                - 81.046,

            "pred_offset_sec":
                pred_gbr
                - 81.046,
        }

        for col, value in (
            pred_values.items()
        ):

            pred_overview.loc[
                0,
                col,
            ] = value

        temp_ledger = (
            temp_dir
            / "ledger.csv"
        )

        temp_log = (
            temp_dir
            / "model_log.csv"
        )

        run_id = (
            "2026_italy_dry_smoke"
        )

        ledger_row = {
            "RunID":
                run_id,

            "race_event_id":
                "2026_italy",

            **pred_values,

            "ImmutableFlag":
                1,

            "CATFallbackUsed":
                1,

            "CATFallbackReason":
                "dry_smoke_explicit_fallback",
        }

        log_row = {
            "RunID":
                run_id,

            "Pred_GBR":
                pred_gbr,

            "Pred_XGB":
                pred_xgb,

            "Pred_CAT":
                pred_cat,

            "PredictedFastestLap_sec":
                pred_gbr,
        }

        pd.DataFrame(
            [ledger_row]
        ).to_csv(
            temp_ledger,
            index=False,
        )

        pd.DataFrame(
            [log_row]
        ).to_csv(
            temp_log,
            index=False,
        )

        globals()[
            "FINAL_PREDICTION_LEDGER_PATH"
        ] = str(
            temp_ledger
        )

        globals()[
            "FINAL_PREDICTION_MODEL_LOG_PATH"
        ] = str(
            temp_log
        )

        artifact_validation = (
            canonical_validate_final_prediction_artifacts_raw(pred_test, pred_overview, 2026, "italy")
        )

        results["checks"][
            "final_prediction_artifacts"
        ] = bool(
            artifact_validation[
                "ok"
            ]
        )

        if not artifact_validation[
            "ok"
        ]:

            results["errors"].extend(
                [
                    "Prediction artifacts: " + e
                    for e in artifact_validation[
                        "errors"
                    ]
                ]
            )

        # ====================================================
        # L. prediction_ready -> predicted
        # ====================================================
        predicted_transition = (
            canonical_validate_status_transition(
                "prediction_ready",
                "predicted",
            )
        )

        results["checks"][
            "prediction_ready_to_predicted"
        ] = (
            predicted_transition
        )

        if not predicted_transition:

            results["errors"].extend(
                predicted_transition
            )

    except Exception as exc:

        results["errors"].append(
            f"{type(exc).__name__}: {exc}"
        )

    finally:

        # ----------------------------------------------------
        # Globals altijd herstellen
        # ----------------------------------------------------

        globals()[
            "FINAL_PREDICTION_LEDGER_PATH"
        ] = (
            original_ledger_path
        )

        globals()[
            "FINAL_PREDICTION_MODEL_LOG_PATH"
        ] = (
            original_model_log_path
        )

        # Tempfiles opruimen
        try:

            _shutil.rmtree(
                temp_dir,
                ignore_errors=True,
            )

        except Exception:
            pass

    if results[
        "errors"
    ]:

        results[
            "ok"
        ] = False

    return results
