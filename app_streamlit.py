
from f1_pipeline.readiness import validate_event_timing as canonical_validate_event_timing
from f1_pipeline.readiness import validate_feature_snapshot as canonical_validate_feature_snapshot

from f1_pipeline.readiness import build_prediction_readiness_report as canonical_build_prediction_readiness_report

from f1_pipeline.readiness import _build_prediction_readiness_report_raw as canonical_build_prediction_readiness_report_raw

from f1_pipeline.lifecycle import validate_status_transition as canonical_validate_status_transition

from f1_pipeline.prediction import (
    validate_final_prediction_artifacts as canonical_validate_final_prediction_artifacts,
    _validate_final_prediction_artifacts_raw as canonical_validate_final_prediction_artifacts_raw,
)


from f1_pipeline.lifecycle import transition_event_status

from f1_pipeline.registry import (
    sync_event_registry,
    registry_state_message,
)
from f1_pipeline.config import CONFIG


from f1_pipeline.readiness import prepare_prediction_ready

# ============================================================
# Canonical prediction API — Phase 7B.2
# ============================================================
from f1_pipeline.prediction import (
    run_final_prediction,
    promote_prediction_to_predicted,
)


import sys
import inspect
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from f1_pipeline.readiness import validate_race_weather_audit_rows as canonical_validate_race_weather_audit_rows
from f1_pipeline.schemas import validate_event_uniqueness as canonical_validate_event_uniqueness
from f1_pipeline.schemas import validate_primary_key_uniqueness as canonical_validate_primary_key_uniqueness
from f1_pipeline.schemas import validate_row_created_contract as canonical_validate_row_created_contract
from f1_pipeline.readiness import validate_circuit_regime_rows as canonical_validate_circuit_regime_rows
from f1_pipeline.readiness import validate_qualifying_ready_rows as canonical_validate_qualifying_ready_rows
from f1_pipeline.readiness import validate_race_weather_ready_rows as canonical_validate_race_weather_ready_rows
from f1_pipeline.readiness import validate_prediction_ready_rows as canonical_validate_prediction_ready_rows
from f1_pipeline.evaluation import recalc_recordlap_dependents as canonical_recalc_recordlap_dependents
from f1_pipeline.qualifying_completion import freeze_with_automatic_qualifying as canonical_freeze_feature_snapshot
from f1_pipeline.readiness import save_race_weather_audit as canonical_save_race_weather_audit
from f1_pipeline.readiness import (
    inspect_circuit_reference as canonical_inspect_circuit_reference,
    save_circuit_reference_validation as canonical_save_circuit_reference_validation,
)
from f1_pipeline.readiness import prepare_race_context_plausibility_ready as canonical_prepare_race_context_plausibility_ready
from f1_pipeline.registry import prepare_event
from f1_pipeline.readiness import (
    build_event_timing_values as canonical_build_event_timing_values,
    build_structural_features as canonical_build_structural_features,
    circuit_row_by_slug as canonical_circuit_row_by_slug,
)
from f1_pipeline.readiness import (
    save_event_timing as canonical_save_event_timing,
    prepare_circuit_regime_ready as canonical_prepare_circuit_regime_ready,
    save_qualifying_ready as canonical_save_qualifying_ready,
    prepare_prediction_ready as canonical_prepare_prediction_ready,
)
from f1_pipeline.evaluation import (
    correct_recordlap as canonical_correct_recordlap,
    save_suggested_model_audit as canonical_save_suggested_model_audit,
)

# ============================================================
# Canonieke projectpaden
# ============================================================
PROJECT_ROOT = str(CONFIG.root)
UTILS_PATH = f"{PROJECT_ROOT}/utils"

DF_TRAINING_PATH = f"{PROJECT_ROOT}/data/processed/df_training.csv"
DF_TEST_PATH = f"{PROJECT_ROOT}/data/processed/df_test.csv"
DF_OVERVIEW_PATH = f"{PROJECT_ROOT}/data/processed/df_overview.csv"
CIRCUIT_INFO_PATH = f"{PROJECT_ROOT}/data/raw/circuit_info.csv"
BACKUP_DIR = f"{PROJECT_ROOT}/data/processed/backups"

if UTILS_PATH not in sys.path:
    sys.path.insert(0, UTILS_PATH)

import helpers

# STEP 9.2 IMPORT — canonical post-race evaluation
from f1_pipeline.evaluation import evaluate_race as canonical_evaluate_race

# STEP 10.6a IMPORT — readiness presentation module
from ui.readiness import (
    render_readiness_action_panel,
    render_workflow_status_header,
)

# STEP 10.6b IMPORT — prediction presentation module
from ui.prediction import render_prediction_dashboard

# STEP 10.6c IMPORT — evaluation presentation module
from ui.evaluation import render_post_race_evaluation_dashboard

# STEP 10.6d IMPORT — diagnostics presentation module
from ui.diagnostics import render_audit_diagnostics_summary

# STEP 11.3A IMPORT — dashboard metrics
from analytics.dashboard_metrics import (
    dashboard_rolling_p90,
    dashboard_rolling_median,
    dashboard_agg_ape_by_group,
    dashboard_agg_ape_by_sprint,
    build_walk_forward_kpi_comparison,
    strategy_kpis,
)

# STEP 11.5A IMPORT — model selection analytics
from analytics.model_selection import (
    MIN_N_SEGMENT,
    MIN_IMPROVEMENT_APE,
    _context_segment_value,
    select_suggested_model,
    walk_forward_model_selection,
)

# STEP 11.6B IMPORT — dashboard presentation
from ui.dashboard import (
    render_dashboard_context_mean_ape,
    render_dashboard_detail_statistics,
    render_dashboard_diagnostic_tables,
    render_dashboard_dynamic_advantage,
    render_dashboard_dynamic_delta_summary,
    render_dashboard_kpi_comparison,
    render_dashboard_model_selection_policy,
    render_dashboard_per_race_table,
    render_dashboard_segment_summary,
    render_dashboard_step4_detail,
)

from ui.diagnostics import (
    audit_uncontrolled_race_status_writes,
    dry_smoke_test_italy_2026,
)

from ui.dashboard import (
    render_dashboard_main_charts,
    render_dashboard_context_bar_charts,
    render_dashboard_model_comparison,
)

from analytics.model_selection import build_context_model_analysis




# ============================================================
# Config
# ============================================================



# Maiden circuit audit semantics:
# OfficialF1RecordAvailable == 0 means RecordLap_sec is a
# provisional model surrogate and not an official F1 record.
OFFICIAL_F1_RECORD_AVAILABLE_MAIDEN_DEFAULT = 0

# === MADRID_MAIDEN_CIRCUIT_POLICY_V1 ===

MAIDEN_RECORD_STATUS = "provisional_model_surrogate"

ALLOWED_RECORDLAP_STATUSES_FOR_ROW_CREATION = {
    "verified",
    MAIDEN_RECORD_STATUS,
}

ALLOWED_RECORDLAP_STATUSES_FOR_CIRCUIT_REGIME_READY = {
    "verified",
    MAIDEN_RECORD_STATUS,
}

def is_maiden_circuit_record_status(status):
    return str(status).strip().lower() == MAIDEN_RECORD_STATUS


def get_default_event_status():
    return "scheduled"







st.set_page_config(page_title="F1 GP Workflow", layout="wide")
st.title("F1 Fastest Lap — GP Workflow")

KEY_COLS = ["Year", "gp_slug"]

STRUCTURAL_COLS = [
    "YearOffset", "Elevation_m", "Length_km", "Turns", "Altitude_m",
    "Altitude_bucket", "Downforce_profile", "Track_type", "Direction",
    "Brake_intensity", "Turns/Length_km", "TurnDensity",
    "avg_speed_kmh", "v_proxy_kmh", "RecordLap_sec",
    "SprintFastestLap_sec", "IsSprintWeekend",
    "CornerFraction_est", "Regime2026_Flag",
    "Regime2026_AeroComponent_sec", "Regime2026_PUComponent_sec",
    "Regime2026_MassComponent_sec", "Regime2026_TyreComponent_sec",
    "Regime2026_PriorLapDelta_sec",
    "Turns/length rank", "avg_speed rank",
    "Latitude", "Longitude",
]

QUALIFYING_COLS = [
    "Q_airtemp_mean", "Q_airtemp_max", "Q_rain_sum",
    "Q_humidity_mean", "PoleTime_sec", "PoleMinusRecord_sec",
    "pole_compound", "pole_c_index",
]

RACE_WEATHER_COLS = [
    "R_airtemp_mean",
    "R_airtemp_max",
    "R_rain_sum",
    "R_humidity_mean",
]

PROTECTED_COLS = {
    "Pred_GBR", "Pred_XGB", "Pred_CAT", "PredictedFastestLap_sec",
    "ActualFastestLap_sec", "Error", "AbsError", "APE_pct",
    "Error_GBR", "AbsError_GBR", "APE_pct_GBR",
    "Error_XGB", "AbsError_XGB", "APE_pct_XGB",
    "Error_CAT", "AbsError_CAT", "APE_pct_CAT",
    "pred_offset_sec", "target_offset_sec", "Error_offset",
    "AbsError_offset", "APE_pct_offset",
    "Pred_GBR_offset_sec", "Pred_XGB_offset_sec", "Pred_CAT_offset_sec",
    "Cumulative_MAE", "Cumulative_RMSE", "Cumulative_R2",
}

STATUS_ORDER = {
    None: 0,
    "": 0,
    "row_created": 1,
    "circuit_regime_ready": 2,
    "qualifying_ready": 3,
    "race_weather_ready": 4,
    "race_context_plausibility_ready": 5,
    "prediction_ready": 6,
    "predicted": 7,
    "evaluated": 8,
}

# ============================================================
# Helper preflight
# ============================================================
def helper_preflight() -> Dict:
    required = {
        "safe_read_csv": ["path"],
        "safe_to_csv": ["df", "path", "index"],
        "backup_file": ["path"],
        "round_float_columns": ["df", "ndigits"],
        "diag_df": ["df", "name", "head_n", "tail_n"],
        "resolve_gp_slug": ["name_or_circuit", "circuit_info_df"],
    }

    errors = []
    for name, expected in required.items():
        fn = getattr(helpers, name, None)
        if fn is None or not callable(fn):
            errors.append(f"Helper ontbreekt of is niet callable: {name}")
            continue
        actual = list(inspect.signature(fn).parameters.keys())
        if actual != expected:
            errors.append(
                f"Signature fout voor {name}: verwacht {expected}, gevonden {actual}"
            )

    if Path(BACKUP_DIR).resolve() != Path(
        str(CONFIG.backup_dir)
    ).resolve():
        errors.append("Backupmap wijkt af van de canonieke backupmap.")

    return {"ok": len(errors) == 0, "errors": errors}


# ============================================================
# Algemene dataframefuncties
# ============================================================
def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = np.nan
    return out


def get_target_mask(df: pd.DataFrame, year: int, gp_slug: str) -> pd.Series:
    return (
        pd.to_numeric(df["Year"], errors="coerce").eq(int(year))
        & df["gp_slug"].astype(str).eq(str(gp_slug))
    )


def require_one(df: pd.DataFrame, year: int, gp_slug: str, name: str) -> pd.Series:
    mask = get_target_mask(df, year, gp_slug)
    count = int(mask.sum())
    if count != 1:
        raise ValueError(
            f"{name}: target {year}/{gp_slug} moet exact één keer voorkomen; gevonden {count}."
        )
    return mask


def duplicate_count(df: pd.DataFrame) -> int:
    return int(df.duplicated(subset=KEY_COLS).sum())


def append_bottom(df: pd.DataFrame, row: Dict) -> pd.DataFrame:
    out = ensure_columns(df, list(row.keys()))
    return pd.concat([out, pd.DataFrame([row])], ignore_index=True)




def show_diag(df: pd.DataFrame, name: str) -> None:
    diag = helpers.diag_df(df, name=name, head_n=5, tail_n=5)
    st.write(f"**{name} shape:** {diag['shape']}")
    with st.expander(f"{name} — columns"):
        st.write(diag["columns"])
    with st.expander(f"{name} — missings top 20"):
        st.dataframe(diag["missings_top"].head(20))
    with st.expander(f"{name} — head/tail"):
        st.dataframe(diag["head"])
        st.dataframe(diag["tail"])


def show_before_after(
    before_test: pd.DataFrame,
    after_test: pd.DataFrame,
    before_overview: pd.DataFrame,
    after_overview: pd.DataFrame,
    year: int,
    gp_slug: str,
) -> None:
    for title, df in [
        ("df_test vóór", before_test),
        ("df_test na", after_test),
        ("df_overview vóór", before_overview),
        ("df_overview na", after_overview),
    ]:
        mask = get_target_mask(df, year, gp_slug) if set(KEY_COLS).issubset(df.columns) else pd.Series(False, index=df.index)
        st.write(f"**{title}**")
        st.dataframe(df.loc[mask] if mask.any() else pd.DataFrame({"melding": ["Targetrij niet aanwezig"]}))

    show_diag(after_test, "df_test")
    show_diag(after_overview, "df_overview")


# ============================================================
# Mapping, circuitdata en regime
# ============================================================
















# ============================================================
# Statusvalidatie
# ============================================================













def infer_status(
    df_test: pd.DataFrame,
    df_overview: pd.DataFrame,
    year: int,
    gp_slug: str,
) -> str:
    """
    Canonical UI-status.

    De opgeslagen RaceStatus in df_test en df_overview is leidend,
    mits:
    - target exact 1x bestaat;
    - beide datasets dezelfde status hebben;
    - status voorkomt in WORKFLOW_STATUS_ORDER.

    Geen terugval meer naar oude afgeleide statuslogica.
    """

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

    if (
        int(mt.sum()) != 1
        or int(mo.sum()) != 1
    ):
        return ""

    status_test = str(
        df_test.loc[
            mt,
            "RaceStatus"
        ].iloc[0]
    ).strip()

    status_overview = str(
        df_overview.loc[
            mo,
            "RaceStatus"
        ].iloc[0]
    ).strip()

    if status_test != status_overview:
        return ""

    if status_test not in WORKFLOW_STATUS_ORDER:
        return ""

    return status_test


# ============================================================
# RecordLap-correctie
# ============================================================



# ============================================================
# Dashboard helpers
# ============================================================








def render_dashboard():


    import matplotlib.pyplot as plt

    st.header(
        "Dashboard — Model Performance"
    )

    # --------------------------------------------------------
    # Altijd opnieuw laden uit autoritatieve overview
    # --------------------------------------------------------
    df = helpers.safe_read_csv(
        DF_OVERVIEW_PATH
    ).copy()

    # STEP 19.2 — context volgt de geselecteerde overviewrij
    try:
        from f1_external_context_panel import render_external_context_panel
        render_external_context_panel(
            df_overview=df,
            year=selected_year,
            gp_slug=selected_slug,
        )
    except Exception:
        st.warning("Extern contextpaneel niet beschikbaar.")


    # --------------------------------------------------------
    # Basischecks
    # --------------------------------------------------------
    for col in [
        "Race #",
        "Year",
        "Circuit",
    ]:
        if col not in df.columns:
            st.error(
                f"Kolom ontbreekt in "
                f"df_overview: {col}"
            )
            return

    # --------------------------------------------------------
    # Numeric conversie
    # --------------------------------------------------------
    numeric_cols = [
        "PredictedFastestLap_sec",
        "ActualFastestLap_sec",
        "Error",
        "AbsError",
        "APE_pct",
        "pred_offset_sec",
        "target_offset_sec",
        "Error_offset",
        "AbsError_offset",
        "APE_pct_offset",
        "Cumulative_MAE",
        "Cumulative_RMSE",
        "Cumulative_R2",
        "IsSprintWeekend",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # --------------------------------------------------------
    # Sortering
    # --------------------------------------------------------
    df = (
        df
        .sort_values(
            ["Year", "Race #"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Dashboard filters
    # --------------------------------------------------------
    years = (
        pd.to_numeric(
            df["Year"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    years = sorted(years)

    f1, f2, f3 = st.columns(3)

    with f1:
        season = st.selectbox(
            "Season",
            options=["All"] + years,
            index=0,
            key="dashboard_season",
        )

    with f2:
        rolling_window = st.slider(
            "Rolling window",
            min_value=2,
            max_value=12,
            value=5,
            step=1,
            key="dashboard_rolling_window",
        )

    with f3:
        max_rows = st.slider(
            "Max rows in table",
            min_value=10,
            max_value=200,
            value=50,
            step=5,
            key="dashboard_max_rows",
        )

    d = df.copy()

    if season != "All":
        d = d[
            pd.to_numeric(
                d["Year"],
                errors="coerce",
            ).eq(int(season))
        ].copy()

    # --------------------------------------------------------
    # Alleen races met prediction + actual
    # --------------------------------------------------------
    d_kpi = d.dropna(
        subset=[
            "PredictedFastestLap_sec",
            "ActualFastestLap_sec",
        ]
    ).copy()

    if d_kpi.empty:
        st.warning(
            "Geen rijen met zowel "
            "PredictedFastestLap_sec als "
            "ActualFastestLap_sec in deze selectie."
        )
        return

    # --------------------------------------------------------
    # X-label
    # --------------------------------------------------------
    d_kpi["Circuit_Year"] = (
        d_kpi["Circuit"].astype(str)
        + " ("
        + d_kpi["Year"]
        .astype(int)
        .astype(str)
        + ")"
    )

    # --------------------------------------------------------
    # Rolling metrics
    # --------------------------------------------------------
    d_kpi["APE_roll_median"] = (
        dashboard_rolling_median(
            d_kpi["APE_pct"],
            rolling_window,
        )
    )

    d_kpi["APE_roll_p90"] = (
        dashboard_rolling_p90(
            d_kpi["APE_pct"],
            rolling_window,
        )
    )

    # --------------------------------------------------------
    # Diagnostische aggregaties
    # --------------------------------------------------------
    by_track = dashboard_agg_ape_by_group(
        d_kpi,
        "Track_type",
    )

    by_alt = dashboard_agg_ape_by_group(
        d_kpi,
        "Altitude_bucket",
    )

    by_sprint = dashboard_agg_ape_by_sprint(
        d_kpi
    )

    # ========================================================
    # KPI CARDS
    # ========================================================
    ape_mean = float(
        np.nanmean(d_kpi["APE_pct"])
    )

    ape_median = float(
        np.nanmedian(d_kpi["APE_pct"])
    )

    ape_p90 = float(
        np.nanpercentile(
            d_kpi["APE_pct"],
            90,
        )
    )

    abs_mean = float(
        np.nanmean(
            d_kpi["AbsError"]
        )
    )

    bias_mean = float(
        np.nanmean(
            d_kpi["Error"]
        )
    )

    k1, k2, k3, k4, k5, k6 = (
        st.columns(6)
    )

    k1.metric(
        "Races",
        len(d_kpi),
    )

    k2.metric(
        "Mean APE",
        f"{ape_mean:.3f}%",
    )

    k3.metric(
        "Median APE",
        f"{ape_median:.3f}%",
    )

    k4.metric(
        "P90 APE",
        f"{ape_p90:.3f}%",
    )

    k5.metric(
        "Mean AbsError",
        f"{abs_mean:.3f} sec",
    )

    k6.metric(
        "Mean Error",
        f"{bias_mean:.3f} sec",
    )

    st.caption(
        "Mean Error < 0 = voorspelling te snel; "
        "Mean Error > 0 = voorspelling te langzaam."
    )

    # ========================================================
    # STEP 13.2E — main performance charts extracted
    render_dashboard_main_charts(
        d_kpi=d_kpi,
        rolling_window=rolling_window,
    )

    # STEP 13.2E — context bar charts extracted
    render_dashboard_context_bar_charts(
        by_track=by_track,
        by_alt=by_alt,
        by_sprint=by_sprint,
    )

    # STEP 13.2E — model comparison extracted
    render_dashboard_model_comparison(
        df=df,
    )

    # PER-RACE TABLE
    # ========================================================
    # STEP 11.6D UI — per-race table extracted
    render_dashboard_per_race_table(
        d_kpi=d_kpi,
        max_rows=max_rows,
    )

    st.subheader(
        "Model Performance by Race Context"
    )

    st.caption(
        "Audit-only analyse. Deze sectie verandert de "
        "operationele modelkeuze nog niet."
    )

    policy_col1, policy_col2 = st.columns(2)

    with policy_col1:
        context_min_n = st.number_input(
            "Minimum races per segment",
            min_value=2,
            max_value=50,
            value=int(MIN_N_SEGMENT),
            step=1,
            key="context_min_n_segment",
        )

    with policy_col2:
        context_min_improvement = st.number_input(
            "Minimum improvement vs GBR (percentage point)",
            min_value=0.0,
            max_value=5.0,
            value=float(MIN_IMPROVEMENT_APE),
            step=0.05,
            format="%.2f",
            key="context_min_improvement_ape",
        )

    st.caption(
        "Een alternatief model wordt alleen voorgesteld als "
        "het segment voldoende races bevat én de mean APE "
        "minimaal deze marge beter is dan GBR."
    )

    context_options = []

    for candidate in [
        "Track_type",
        "Altitude_bucket",
        "IsSprintWeekend",
        "Downforce_profile",
    ]:
        if candidate in df.columns:
            context_options.append(candidate)

    if not context_options:
        st.warning(
            "Geen geschikte contextkolommen beschikbaar "
            "voor segmentanalyse."
        )

    else:
        selected_context = st.selectbox(
            "Group by",
            options=context_options,
            key="dashboard_context_group",
        )

        # ----------------------------------------------------
        # Alleen races met actual + model-specifieke APE
        # ----------------------------------------------------
        required_context_cols = [
            "ActualFastestLap_sec",
            "APE_pct_GBR",
            "APE_pct_XGB",
            "APE_pct_CAT",
        ]

        missing_context_cols = [
            c
            for c in required_context_cols
            if c not in df.columns
        ]

        if missing_context_cols:
            st.warning(
                "Contextanalyse niet beschikbaar. "
                f"Ontbrekende kolommen: {missing_context_cols}"
            )

        else:
            context_df = df.copy()

            # Numeriek maken
            for col in [
                "APE_pct_GBR",
                "APE_pct_XGB",
                "APE_pct_CAT",
                "IsSprintWeekend",
            ]:
                if col in context_df.columns:
                    context_df[col] = pd.to_numeric(
                        context_df[col],
                        errors="coerce",
                    )

            # Alleen rijen met actual
            context_df = context_df[
                context_df[
                    "ActualFastestLap_sec"
                ].notna()
            ].copy()

            # Context opschonen
            if selected_context == "IsSprintWeekend":
                context_df["ContextSegment"] = np.where(
                    pd.to_numeric(
                        context_df["IsSprintWeekend"],
                        errors="coerce",
                    )
                    .fillna(0)
                    .astype(int)
                    .eq(1),
                    "Sprint weekend",
                    "Non-sprint weekend",
                )
            else:
                context_df["ContextSegment"] = (
                    context_df[
                        selected_context
                    ].astype(str)
                )

                context_df.loc[
                    context_df[selected_context].isna(),
                    "ContextSegment",
                ] = np.nan

            # ------------------------------------------------
            # Long format: model + APE
            # ------------------------------------------------
            long_parts = []

            model_map = {
                "GBR": "APE_pct_GBR",
                "XGB": "APE_pct_XGB",
                "CAT": "APE_pct_CAT",
            }

            for model_name, ape_col in model_map.items():

                temp = context_df[
                    [
                        "ContextSegment",
                        ape_col,
                    ]
                ].copy()

                temp = temp.rename(
                    columns={
                        ape_col: "APE_pct"
                    }
                )

                temp["Model"] = model_name

                temp = temp.dropna(
                    subset=[
                        "ContextSegment",
                        "APE_pct",
                    ]
                )

                long_parts.append(temp)

            if long_parts:
                context_long = pd.concat(
                    long_parts,
                    ignore_index=True,
                )
            else:
                context_long = pd.DataFrame()

            if context_long.empty:
                st.info(
                    "Geen bruikbare model-APE data "
                    "voor deze context."
                )

            else:
                # --------------------------------------------
                # STEP 13.2H — context model analytics extracted
                context_analysis = build_context_model_analysis(
                    context_df=context_df,
                    context_long=context_long,
                    context_min_improvement=context_min_improvement,
                    context_min_n=context_min_n,
                )
                context_stats = context_analysis["context_stats"]
                summary = context_analysis["summary"]

                # TABEL
                # --------------------------------------------
                # STEP 11.6R UI — segment summary extracted
                render_dashboard_segment_summary(
                    summary=summary,
                )

                # --------------------------------------------
                # Policy summary
                # --------------------------------------------
                # STEP 11.6P UI — model selection policy extracted
                render_dashboard_model_selection_policy(
                    summary=summary,
                )


                # --------------------------------------------
                # BAR CHART
                # --------------------------------------------
                # STEP 11.6H UI — context mean APE extracted
                render_dashboard_context_mean_ape(
                    context_stats=context_stats,
                    selected_context=selected_context,
                )
                # STEP 11.6T UI — detail statistics extracted
                render_dashboard_detail_statistics(
                    context_stats=context_stats,
                )



                # ============================================
                # STEP 4 — BASELINE VS DYNAMIC KPI COMPARISON
                # ============================================
                st.markdown(
                    "#### Baseline vs Dynamic KPI comparison"
                )

                st.caption(
                    "Leakage-vrije auditvergelijking. "
                    "GBR blijft operationeel de baseline."
                )

                try:
                    # Walk-forward opnieuw berekenen.
                    # Hierdoor is Stap 4 niet afhankelijk van
                    # lokale variabelen uit de Stap-3 UI.
                    step4_wf = walk_forward_model_selection(
                        source_df=df,
                        context_col=selected_context,
                        min_n_segment=int(
                            context_min_n
                        ),
                        min_improvement_ape=float(
                            context_min_improvement
                        ),
                    )

                    step4_comp = (
                        build_walk_forward_kpi_comparison(
                            wf_results=step4_wf,
                            source_df=df,
                        )
                    )

                except Exception as exc:

                    st.error(
                        "Stap 4 kon niet worden uitgevoerd: "
                        f"{exc}"
                    )

                    step4_comp = pd.DataFrame()

                if not step4_comp.empty:

                    # ----------------------------------------
                    # Eigen season filter
                    # ----------------------------------------
                    step4_years = sorted(
                        pd.to_numeric(
                            step4_comp["Year"],
                            errors="coerce",
                        )
                        .dropna()
                        .astype(int)
                        .unique()
                        .tolist()
                    )

                    step4_season = st.selectbox(
                        "KPI comparison season",
                        options=["All"] + step4_years,
                        key="step4_kpi_season",
                    )

                    step4_eval = (
                        step4_comp.copy()
                    )

                    if step4_season != "All":
                        step4_eval = step4_eval[
                            pd.to_numeric(
                                step4_eval["Year"],
                                errors="coerce",
                            ).eq(
                                int(step4_season)
                            )
                        ].copy()

                    step4_eval = step4_eval[
                        step4_eval[
                            "KPI_Evaluable"
                        ].eq(True)
                    ].copy()

                    if step4_eval.empty:

                        st.warning(
                            "Geen evalueerbare races "
                            "voor deze selectie."
                        )

                    else:
                        # ====================================
                        # KPI'S
                        # ====================================
                        gbr_kpi = strategy_kpis(
                            step4_eval,
                            "GBR_Error_sec",
                            "GBR_AbsError_sec",
                            "GBR_APE_pct_calc",
                            "Always GBR",
                        )

                        dyn_kpi = strategy_kpis(
                            step4_eval,
                            "Dynamic_Error_sec",
                            "Dynamic_AbsError_sec",
                            "Dynamic_APE_pct_calc",
                            "Dynamic selection",
                        )

                        kpi_table = pd.DataFrame(
                            [gbr_kpi, dyn_kpi]
                        )

                        numeric_cols = [
                            "MAE_sec",
                            "RMSE_sec",
                            "Mean_APE_pct",
                            "Median_APE_pct",
                            "P90_APE_pct",
                        ]

                        for col in numeric_cols:
                            kpi_table[col] = (
                                pd.to_numeric(
                                    kpi_table[col],
                                    errors="coerce",
                                ).round(3)
                            )

                        st.dataframe(
                            kpi_table,
                            width="stretch",
                            hide_index=True,
                        )

                        # ====================================
                        # IMPROVEMENT CARDS
                        # ====================================
                        mae_gain = (
                            gbr_kpi["MAE_sec"]
                            - dyn_kpi["MAE_sec"]
                        )

                        rmse_gain = (
                            gbr_kpi["RMSE_sec"]
                            - dyn_kpi["RMSE_sec"]
                        )

                        mean_ape_gain = (
                            gbr_kpi["Mean_APE_pct"]
                            - dyn_kpi["Mean_APE_pct"]
                        )

                        median_gain = (
                            gbr_kpi["Median_APE_pct"]
                            - dyn_kpi["Median_APE_pct"]
                        )

                        p90_gain = (
                            gbr_kpi["P90_APE_pct"]
                            - dyn_kpi["P90_APE_pct"]
                        )

                        c1, c2, c3, c4, c5 = (
                            st.columns(5)
                        )

                        c1.metric(
                            "MAE improvement",
                            f"{mae_gain:+.3f}s",
                        )

                        c2.metric(
                            "RMSE improvement",
                            f"{rmse_gain:+.3f}s",
                        )

                        c3.metric(
                            "Mean APE improvement",
                            f"{mean_ape_gain:+.3f} pp",
                        )

                        c4.metric(
                            "Median APE improvement",
                            f"{median_gain:+.3f} pp",
                        )

                        c5.metric(
                            "P90 APE improvement",
                            f"{p90_gain:+.3f} pp",
                        )

                        # STEP 11.6V UI — dynamic delta summary extracted
                        render_dashboard_dynamic_delta_summary(
                            step4_eval=step4_eval,
                        )

                        # ====================================
                        # MODEL USE
                        # ====================================
                        st.markdown(
                            "##### Dynamic model usage"
                        )

                        model_usage = (
                            step4_eval[
                                "SelectedModel"
                            ]
                            .value_counts()
                            .reindex(
                                ["GBR", "XGB", "CAT"],
                                fill_value=0,
                            )
                            .rename_axis("Model")
                            .reset_index(
                                name="Races"
                            )
                        )

                        model_usage["Pct"] = (
                            100
                            * model_usage["Races"]
                            / len(step4_eval)
                        ).round(1)

                        st.dataframe(
                            model_usage,
                            width="stretch",
                            hide_index=True,
                        )

                        # ====================================
                        # KPI BAR CHART
                        # ====================================
                        # STEP 11.6N UI — KPI comparison extracted
                        render_dashboard_kpi_comparison(
                            dyn_kpi=dyn_kpi,
                            gbr_kpi=gbr_kpi,
                        )

                        # ====================================
                        # RACE-BY-RACE DELTA
                        # ====================================
                        # STEP 11.6J UI — dynamic advantage extracted
                        render_dashboard_dynamic_advantage(
                            step4_eval=step4_eval,
                        )
                        # STEP 11.6L UI — step4 detail extracted
                        render_dashboard_step4_detail(
                            step4_eval=step4_eval,
                        )

                        # ====================================
                        # CONCLUSIE
                        # ====================================
                        if (
                            mae_gain > 0
                            and rmse_gain > 0
                            and mean_ape_gain > 0
                        ):
                            st.success(
                                "Dynamic selection is in deze "
                                "backtest beter op MAE, RMSE "
                                "en mean APE. Audit-only: "
                                "GBR blijft operationeel."
                            )

                        else:
                            st.warning(
                                "Dynamic selection is nog niet "
                                "overtuigend beter op alle "
                                "hoofd-KPI's. GBR blijft baseline."
                            )



    # ========================================================
    # DIAGNOSTIC TABLES
    # ========================================================
    # STEP 11.6B UI — diagnostic tables extracted
    render_dashboard_diagnostic_tables(
        by_alt=by_alt,
        by_sprint=by_sprint,
        by_track=by_track,
    )




# ============================================================
# Audit-only model selection policy
# ============================================================






# ============================================================
# Walk-forward model selection backtest
# ============================================================







# ============================================================
# Step 4 — Baseline vs Dynamic KPI helpers
# ============================================================







# ============================================================
# Step 5 — Audit-only SuggestedModel for future race
# ============================================================
SUGGESTED_MODEL_COLS = [
    "SuggestedModel",
    "SuggestedModelReason",
    "SuggestedSegment",
    "SuggestedSegment_n",
    "Suggested_APE",
    "Suggested_vs_GBR_APE",
]


def build_future_model_suggestion(
    df_overview: pd.DataFrame,
    year: int,
    gp_slug: str,
    min_n_segment: int = MIN_N_SEGMENT,
    min_improvement_ape: float = MIN_IMPROVEMENT_APE,
) -> dict:
    """
    Bepaalt audit-only modeladvies voor één toekomstige race.

    Leakagebescherming:
    uitsluitend races vóór de targetrace worden gebruikt.

    Contexten:
    - Track_type
    - Altitude_bucket
    - IsSprintWeekend
    - Downforce_profile

    Operational prediction wordt NIET aangepast.
    """

    target_mask = get_target_mask(
        df_overview,
        year,
        gp_slug,
    )

    if int(target_mask.sum()) != 1:
        raise ValueError(
            "Target moet exact 1 keer voorkomen in df_overview."
        )

    target = (
        df_overview
        .loc[target_mask]
        .iloc[0]
    )

    target_year = pd.to_numeric(
        target.get("Year"),
        errors="coerce",
    )

    target_race_number = pd.to_numeric(
        target.get("Race #"),
        errors="coerce",
    )

    if pd.isna(target_year):
        raise ValueError(
            "Year ontbreekt voor targetrace."
        )

    if pd.isna(target_race_number):
        raise ValueError(
            "Race # ontbreekt voor targetrace."
        )

    # --------------------------------------------------------
    # Alleen eerdere races
    # --------------------------------------------------------
    history = df_overview.copy()

    history["Year"] = pd.to_numeric(
        history["Year"],
        errors="coerce",
    )

    history["Race #"] = pd.to_numeric(
        history["Race #"],
        errors="coerce",
    )

    history = history[
        (
            history["Year"] < target_year
        )
        |
        (
            history["Year"].eq(target_year)
            & history["Race #"].lt(target_race_number)
        )
    ].copy()

    # Alleen races waarvoor model-APE beschikbaar is
    for col in [
        "APE_pct_GBR",
        "APE_pct_XGB",
        "APE_pct_CAT",
    ]:
        if col not in history.columns:
            raise ValueError(
                f"Ontbrekende kolom: {col}"
            )

        history[col] = pd.to_numeric(
            history[col],
            errors="coerce",
        )

    context_cols = [
        c
        for c in [
            "Track_type",
            "Altitude_bucket",
            "IsSprintWeekend",
            "Downforce_profile",
        ]
        if c in df_overview.columns
    ]

    audit_rows = []

    for context_col in context_cols:

        target_segment = _context_segment_value(
            target,
            context_col,
        )

        if pd.isna(target_segment):
            continue

        hist = history.copy()

        hist["__segment__"] = hist.apply(
            lambda row: _context_segment_value(
                row,
                context_col,
            ),
            axis=1,
        )

        segment_history = hist[
            hist["__segment__"].eq(
                target_segment
            )
        ].copy()

        # n_races = races met ten minste GBR-baseline beschikbaar
        valid_gbr = segment_history[
            "APE_pct_GBR"
        ].dropna()

        n_races = int(
            len(valid_gbr)
        )

        means = {}

        model_ns = {}

        for model, col in {
            "GBR": "APE_pct_GBR",
            "XGB": "APE_pct_XGB",
            "CAT": "APE_pct_CAT",
        }.items():

            vals = pd.to_numeric(
                segment_history[col],
                errors="coerce",
            ).dropna()

            model_ns[model] = int(
                len(vals)
            )

            means[model] = (
                float(vals.mean())
                if len(vals)
                else np.nan
            )

        policy_input = pd.Series({
            "n_races": n_races,
            "GBR_mean_APE": means["GBR"],
            "XGB_mean_APE": means["XGB"],
            "CAT_mean_APE": means["CAT"],
        })

        policy = select_suggested_model(
            policy_input,
            min_n_segment=int(
                min_n_segment
            ),
            min_improvement_ape=float(
                min_improvement_ape
            ),
        )

        audit_rows.append({
            "Context": context_col,
            "Segment": target_segment,
            "n": n_races,
            "n_GBR": model_ns["GBR"],
            "n_XGB": model_ns["XGB"],
            "n_CAT": model_ns["CAT"],
            "GBR_APE": means["GBR"],
            "XGB_APE": means["XGB"],
            "CAT_APE": means["CAT"],
            "SuggestedModel": policy[
                "SuggestedModel"
            ],
            "PolicyPassed": policy[
                "PolicyPassed"
            ],
            "Improvement_vs_GBR": policy[
                "Suggested_vs_GBR_APE"
            ],
            "Suggested_APE": policy[
                "Suggested_APE"
            ],
            "Reason": policy[
                "SuggestedModelReason"
            ],
        })

    audit_df = pd.DataFrame(
        audit_rows
    )

    # --------------------------------------------------------
    # Geen bruikbare context -> GBR fallback
    # --------------------------------------------------------
    if audit_df.empty:
        return {
            "SuggestedModel": "GBR",
            "SuggestedModelReason": (
                "Geen bruikbare contextsegmenten; GBR fallback."
            ),
            "SuggestedSegment": "fallback",
            "SuggestedSegment_n": 0,
            "Suggested_APE": np.nan,
            "Suggested_vs_GBR_APE": 0.0,
            "PolicyPassed": False,
            "audit_df": audit_df,
        }

    # --------------------------------------------------------
    # Alleen contextregels die policy passeren
    # --------------------------------------------------------
    passed = audit_df[
        audit_df["PolicyPassed"].eq(True)
    ].copy()

    if passed.empty:

        # Beste beschikbare GBR-context als auditreferentie
        fallback = (
            audit_df
            .sort_values(
                ["n", "GBR_APE"],
                ascending=[False, True],
                na_position="last",
            )
            .iloc[0]
        )

        return {
            "SuggestedModel": "GBR",
            "SuggestedModelReason": (
                "Geen alternatief model voldoet aan "
                f"n>={min_n_segment} en improvement>="
                f"{min_improvement_ape:.3f} pp. GBR fallback."
            ),
            "SuggestedSegment": (
                f"{fallback['Context']}="
                f"{fallback['Segment']}"
            ),
            "SuggestedSegment_n": int(
                fallback["n"]
            ),
            "Suggested_APE": (
                float(fallback["GBR_APE"])
                if pd.notna(fallback["GBR_APE"])
                else np.nan
            ),
            "Suggested_vs_GBR_APE": 0.0,
            "PolicyPassed": False,
            "audit_df": audit_df,
        }

    # --------------------------------------------------------
    # Meest overtuigende geldige context:
    # grootste verbetering versus GBR.
    # Bij gelijkspel: grootste n.
    # --------------------------------------------------------
    passed = passed.sort_values(
        [
            "Improvement_vs_GBR",
            "n",
        ],
        ascending=[
            False,
            False,
        ],
    )

    best = passed.iloc[0]

    model = str(
        best["SuggestedModel"]
    )

    segment_text = (
        f"{best['Context']}="
        f"{best['Segment']}"
    )

    reason = (
        f"{model} auditadvies op basis van {segment_text}; "
        f"n={int(best['n'])}; "
        f"{model} mean APE={float(best['Suggested_APE']):.3f}%; "
        f"verbetering vs GBR="
        f"{float(best['Improvement_vs_GBR']):.3f} pp. "
        "Audit-only; operationele prediction blijft GBR."
    )

    return {
        "SuggestedModel": model,
        "SuggestedModelReason": reason,
        "SuggestedSegment": segment_text,
        "SuggestedSegment_n": int(
            best["n"]
        ),
        "Suggested_APE": float(
            best["Suggested_APE"]
        ),
        "Suggested_vs_GBR_APE": float(
            best["Improvement_vs_GBR"]
        ),
        "PolicyPassed": True,
        "audit_df": audit_df,
    }





# ============================================================
# Step 6 — Operational model activation
# ============================================================
MIN_OPERATIONAL_BACKTEST_RACES = 12
MIN_OPERATIONAL_MEAN_APE_GAIN = 0.10

OPERATIONAL_MODEL_COLS = [
    "OperationalModel",
    "OperationalModelReason",
    "OperationalSelection_n",
    "OperationalSelection_APE",
]









# ============================================================
# Step 2 — Event timing
# ============================================================
EVENT_TIMING_COLS = [
    "PredictionTimingPolicyVersion", "AnnouncedRaceStartUTC",
    "RaceStartSource", "RaceStartVerifiedAtUTC",
    "EventDateTimeUTC",
    "PredictionCutoffUTC",
    "PredictionCutoffPolicy",
    "R_window_start_utc",
    "R_window_end_utc",
    "RaceWindowDurationMinutes",
    "RaceWindowSource",
]











# ============================================================
# Event timing hard gate for qualifying_ready
# ============================================================








# ============================================================
# Step 3 — Canonical Race Weather Audit Sync
# ============================================================









# ============================================================
# Step 4 — Feature snapshot freeze
# ============================================================









# ============================================================
# Step 5 — Future Event Registry
# ============================================================
FUTURE_EVENT_REGISTRY_PATH = (
    str(CONFIG.registry_path)
)

FUTURE_EVENT_REGISTRY_ELIGIBLE_PATH = (
    str(CONFIG.registry_eligible_mirror_path)
)

REGISTRY_REQUIRED_COLS = [
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

# Extra auditveld zodat "layout verified" niet stilzwijgend
# wordt afgeleid uit alleen een gevulde slug.
REGISTRY_AUDIT_COLS = [
    "CircuitLayoutValidationStatus",
    "RegistryEligibilityReason",
    "RegistryUpdatedUTC",
]























# ============================================================
# Step 6 — Prediction Readiness Gate
# ============================================================

WORKFLOW_STATUS_ORDER = [
    "row_created",
    "circuit_regime_ready",
    "qualifying_ready",
    "race_weather_ready",
    "race_context_plausibility_ready",
    "prediction_ready",
    "predicted",
    "evaluated",
]


def _readiness_value_present(value) -> bool:

    if value is None:
        return False

    if pd.isna(value):
        return False

    if isinstance(value, str):
        return (
            value.strip().lower()
            not in {
                "",
                "nan",
                "none",
                "null",
            }
        )

    return True


def _readiness_verified(value) -> bool:

    if not _readiness_value_present(
        value
    ):
        return False

    return (
        str(value)
        .strip()
        .lower()
        == "verified"
    )


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








def prediction_readiness_ok(
    gates: list,
) -> bool:

    return (
        bool(gates)
        and all(
            bool(
                gate.get(
                    "ok",
                    False,
                )
            )
            for gate in gates
        )
    )







# ============================================================
# Step 7 — Final Prediction Run
# ============================================================

FINAL_PREDICTION_TRAINING_PATH = (
    str(CONFIG.training_path)
)

FINAL_PREDICTION_LEDGER_PATH = (
    str(CONFIG.prediction_ledger_path)
)

FINAL_PREDICTION_MODEL_LOG_PATH = (
    str(CONFIG.model_log_path)
)

FINAL_PREDICTION_COLS = [
    "Pred_GBR",
    "Pred_XGB",
    "Pred_CAT",
    "PredictedFastestLap_sec",
    "Pred_GBR_offset_sec",
    "Pred_XGB_offset_sec",
    "Pred_CAT_offset_sec",
    "pred_offset_sec",
]

PREDICTION_LEDGER_REQUIRED_COLS = [
    "RunID",
    "PredictionTimestampUTC",
    "race_event_id",
    "Year",
    "gp_slug",
    "Circuit",
    "CircuitLayoutSlug",
    "EventDateTimeUTC",
    "PredictionCutoffUTC",
    "FeatureSnapshotTimestampUTC",
    "FeatureSnapshotAfterQualifyingUTC",
    "ModelVersion_GBR",
    "ModelVersion_XGB",
    "ModelVersion_CAT",
    "CATFallbackUsed",
    "CATFallbackReason",
    "FeatureSetHash",
    "Pred_GBR",
    "Pred_XGB",
    "Pred_CAT",
    "PredictedFastestLap_sec",
    "Pred_GBR_offset_sec",
    "Pred_XGB_offset_sec",
    "Pred_CAT_offset_sec",
    "pred_offset_sec",
    "RecordLap_sec",
    "PostPredictPlausibilityStatus",
    "PredictionRowHash",
    "ImmutableFlag",
    "EvaluationStatus",
]






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



















# ============================================================
# Step 8 — Promote prediction_ready -> predicted
# ============================================================








# ============================================================
# Step 9 — Canonical RaceStatus controller
# ============================================================








# ============================================================
# Step 10 — UX consolidation + Italy 2026 dry smoke
# ============================================================

WORKFLOW_NEXT_ACTION = {
    "row_created": {
        "next_status": "circuit_regime_ready",
        "action": "Verrijk circuit & regime",
    },

    "circuit_regime_ready": {
        "next_status": "qualifying_ready",
        "action": "Vul event timing en qualifying",
    },

    "qualifying_ready": {
        "next_status": "race_weather_ready",
        "action": "Run Race Weather",
    },

    "race_weather_ready": {
        "next_status": "race_context_plausibility_ready",
        "action": "Run race-context plausibility",
    },

    "race_context_plausibility_ready": {
        "next_status": "prediction_ready",
        "action": "Freeze snapshot + maak registry eligible",
    },

    "prediction_ready": {
        "next_status": "predicted",
        "action": "Run final prediction",
    },

    "predicted": {
        "next_status": "evaluated",
        "action": "Wacht op actual / voer evaluatie uit",
    },

    "evaluated": {
        "next_status": None,
        "action": "Workflow voltooid",
    },
}


def build_workflow_ux_state(
    df_test: pd.DataFrame,
    df_overview: pd.DataFrame,
    year: int,
    gp_slug: str,
) -> dict:

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

    if (
        test_count != 1
        or overview_count != 1
    ):

        return {
            "ok": False,
            "RaceStatus": None,
            "progress": 0.0,
            "next_status": None,
            "action":
                "Herstel duplicate/missing target",
            "detail":
                f"df_test={test_count}, "
                f"df_overview={overview_count}",
        }

    status_test = str(
        df_test
        .loc[mt]
        .iloc[0]
        .get(
            "RaceStatus",
            "",
        )
    ).strip()

    status_overview = str(
        df_overview
        .loc[mo]
        .iloc[0]
        .get(
            "RaceStatus",
            "",
        )
    ).strip()

    if status_test != status_overview:

        return {
            "ok": False,
            "RaceStatus": None,
            "progress": 0.0,
            "next_status": None,
            "action":
                "Herstel RaceStatus-synchronisatie",
            "detail":
                f"df_test={status_test!r}, "
                f"df_overview={status_overview!r}",
        }

    if status_test not in WORKFLOW_STATUS_ORDER:

        return {
            "ok": False,
            "RaceStatus": status_test,
            "progress": 0.0,
            "next_status": None,
            "action":
                "Herstel onbekende RaceStatus",
            "detail":
                status_test,
        }

    idx = WORKFLOW_STATUS_ORDER.index(
        status_test
    )

    progress = (
        idx
        / (
            len(
                WORKFLOW_STATUS_ORDER
            )
            - 1
        )
    )

    next_info = (
        WORKFLOW_NEXT_ACTION[
            status_test
        ]
    )

    return {
        "ok": True,

        "RaceStatus":
            status_test,

        "progress":
            float(
                progress
            ),

        "next_status":
            next_info[
                "next_status"
            ],

        "action":
            next_info[
                "action"
            ],

        "detail":
            "OK",
    }





# ============================================================
# UI data load
# ============================================================
preflight = helper_preflight()
if not preflight["ok"]:
    st.error("Helper-preflight faalt. Datasetwijzigingen zijn geblokkeerd.")
    st.write(preflight["errors"])
    st.stop()

df_test = helpers.safe_read_csv(DF_TEST_PATH)
df_overview = helpers.safe_read_csv(DF_OVERVIEW_PATH)
df_training = helpers.safe_read_csv(DF_TRAINING_PATH)
circuit_info = helpers.safe_read_csv(CIRCUIT_INFO_PATH)

gp_options = sorted(circuit_info["gp_slug"].dropna().astype(str).unique())


# STEP 8.4 UI HELPER — readiness/action panel




# STEP 8.5 UI HELPER — prediction dashboard
















# STEP 8.6 UI HELPER — post-race evaluation dashboard














# STEP 8.7 UI HELPER — audit diagnostics summary








# STEP 8.2 UI HELPER — workflow status header






with st.sidebar:
    st.header("Race selecteren")
    selected_year = st.number_input("Year", min_value=2020, max_value=2035, value=2026, step=1)
    selected_gp = st.selectbox("GP", gp_options)
    selected_slug = helpers.resolve_gp_slug(selected_gp, circuit_info)

current_exists = (
    get_target_mask(df_test, selected_year, selected_slug).sum() == 1
    and get_target_mask(df_overview, selected_year, selected_slug).sum() == 1
)
current_status = (
    infer_status(df_test, df_overview, selected_year, selected_slug)
    if current_exists else ""
)


# STEP 8.2 UI CALL — workflow status header
render_workflow_status_header(
    year=selected_year,
    gp_slug=selected_slug,
    race_status=current_status,
)

# STEP 8.4 UI CALL — readiness/action panel
render_readiness_action_panel(
    race_status=current_status,
)

# STEP 8.3 NAVIGATION — task-oriented labels
tabs = st.tabs([
    "Event setup",
    "Pre-race · Circuit",
    "Pre-race · Qualifying",
    "Pre-race · Race weather",
    "Event setup · RecordLap",
    "Prediction · Readiness & run",
    "Audit / diagnostics",
    "Post-race / Model KPI",
])

# ============================================================
# Tab 1 — row_created
# ============================================================
with tabs[0]:
    st.header("Stap 1 — Nieuwe GP aanmaken")
    # EVENT_CIRCUIT_UI_V1
    from f1_pipeline.event_circuits import (
        resolve_event_circuit as canonical_resolve_event_circuit,
    )
    ci, selected_circuit_reference, selected_event_layout = (
        canonical_resolve_event_circuit(
            int(selected_year), selected_slug, circuit_info
        )
    )
    if selected_event_layout is not None:
        st.caption(
            "Circuit: Sepang (Maleisië). "
            "Deze GP behoudt de eventnaam Bahrein."
        )
    st.dataframe(pd.DataFrame([ci]))

    is_sprint_create = st.checkbox("IsSprintWeekend", value=False, key="create_sprint")

    if st.button("Nieuwe GP aanmaken", type="primary"):
        before_test = df_test.copy()
        before_overview = df_overview.copy()

        try:
            create_result = prepare_event(
                year=selected_year,
                gp_slug=selected_slug,
                is_sprint_weekend=is_sprint_create,
            )
        except Exception as exc:
            st.error(
                "Nieuwe GP aanmaken geblokkeerd."
            )
            st.code(str(exc))
            st.stop()

        canonical = create_result["gp_slug"]
        reread_test = helpers.safe_read_csv(DF_TEST_PATH)
        reread_overview = helpers.safe_read_csv(DF_OVERVIEW_PATH)

        st.success("RaceStatus = row_created")
        show_before_after(
            before_test,
            reread_test,
            before_overview,
            reread_overview,
            selected_year,
            canonical,
        )
        st.info(
            "Voortgangslog: nieuwe rij canonical aangemaakt "
            "via registry.prepare_event; transactionele "
            "write en row_created-validatie geslaagd."
        )

        st.session_state["selected_year"] = int(selected_year)
        st.session_state["selected_gp_slug"] = canonical
        st.session_state["last_completed_step"] = "row_created"
        st.rerun()

# ============================================================


# ============================================================
# Tab 1 extra — Event timing
# ============================================================
with tabs[0]:

    st.divider()

    st.subheader(
        "Event timing"
    )

    st.caption(
        "Verplicht vóór qualifying_ready. "
        "Leg de officieel aangekondigde start en de bron vast; het racewindow volgt die start."
    )

    timing_test = helpers.safe_read_csv(
        DF_TEST_PATH
    )

    timing_overview = helpers.safe_read_csv(
        DF_OVERVIEW_PATH
    )

    timing_test_count = int(
        get_target_mask(
            timing_test,
            selected_year,
            selected_slug,
        ).sum()
    )

    timing_overview_count = int(
        get_target_mask(
            timing_overview,
            selected_year,
            selected_slug,
        ).sum()
    )

    if (
        timing_test_count != 1
        or timing_overview_count != 1
    ):

        st.info(
            "Maak/selecteer eerst de nieuwe GP. "
            "Event timing kan pas worden opgeslagen "
            "als de target exact één keer bestaat."
        )

    else:

        mt_timing = require_one(
            timing_test,
            selected_year,
            selected_slug,
            "df_test",
        )

        timing_row = (
            timing_test
            .loc[mt_timing]
            .iloc[0]
        )

        existing_event = pd.to_datetime(
            timing_row.get(
                "EventDateTimeUTC"
            ),
            utc=True,
            errors="coerce",
        )

        if pd.isna(existing_event):

            # Geen gok naar GP-datum opslaan:
            # gebruiker bevestigt eerst officiële racestart.
            default_date = pd.Timestamp.now(
                tz="UTC"
            ).date()

            default_time = pd.Timestamp(
                "14:00:00"
            ).time()

        else:

            default_date = (
                existing_event.date()
            )

            default_time = (
                existing_event.time()
                .replace(
                    tzinfo=None
                )
            )

        with st.form(
            "event_timing_form"
        ):

            event_date = st.date_input(
                "Race date (UTC)",
                value=default_date,
            )

            event_time = st.time_input(
                "Race start time (UTC)",
                value=default_time,
            )

            race_start_source = st.text_input(
                "Officiële bron/verwijzing voor de aangekondigde racestart",
                value="",
            )

            race_window_minutes = st.number_input(
                "Race window duration (minuten)",
                min_value=30,
                max_value=360,
                value=120,
                step=15,
            )

            timing_submit = (
                st.form_submit_button(
                    "Vul event timing"
                )
            )

        if timing_submit:
            before_test = helpers.safe_read_csv(DF_TEST_PATH)
            before_overview = helpers.safe_read_csv(DF_OVERVIEW_PATH)
            event_datetime_text = (
                f"{event_date.isoformat()} "
                f"{event_time.strftime('%H:%M:%S')}Z"
            )
            try:
                timing_result = canonical_save_event_timing(
                    selected_year,
                    selected_slug,
                    event_datetime_text,
                    announced_race_start_utc=event_datetime_text,
                    race_start_source=race_start_source,
                    race_window_minutes=int(race_window_minutes),
                )
            except Exception as exc:
                st.error("Event timing opslaan geblokkeerd.")
                st.code(str(exc))
                st.stop()
            reread_test = helpers.safe_read_csv(DF_TEST_PATH)
            reread_overview = helpers.safe_read_csv(DF_OVERVIEW_PATH)
            st.success("Event timing volledig en synchroon opgeslagen.")
            show_before_after(
                before_test,
                reread_test,
                before_overview,
                reread_overview,
                selected_year,
                selected_slug,
            )
            st.dataframe(
                reread_test.loc[
                    get_target_mask(reread_test, selected_year, selected_slug),
                    EVENT_TIMING_COLS,
                ],
                width="stretch",
            )
            st.rerun()



# Tab 2 — circuit_regime_ready
# ============================================================
with tabs[1]:
    st.header("Stap 2 — Circuit- en regimeverrijking")

    # Altijd opnieuw lezen om stale in-memory data na Stap 1 te voorkomen.
    df_test_step2 = helpers.safe_read_csv(DF_TEST_PATH)
    df_overview_step2 = helpers.safe_read_csv(DF_OVERVIEW_PATH)
    df_training_step2 = helpers.safe_read_csv(DF_TRAINING_PATH)
    circuit_info_step2 = helpers.safe_read_csv(CIRCUIT_INFO_PATH)

    step2_test_count = int(
        get_target_mask(
            df_test_step2,
            selected_year,
            selected_slug,
        ).sum()
    )

    step2_overview_count = int(
        get_target_mask(
            df_overview_step2,
            selected_year,
            selected_slug,
        ).sum()
    )

    step2_exists = (
        step2_test_count == 1
        and step2_overview_count == 1
    )

    if step2_exists:
        step2_status = infer_status(
            df_test_step2,
            df_overview_step2,
            selected_year,
            selected_slug,
        )
    else:
        step2_status = ""

    st.caption(
        f"Herkenning target: df_test={step2_test_count}, "
        f"df_overview={step2_overview_count}, status={step2_status or 'onbekend'}"
    )

    if not step2_exists:
        st.error(
            "De race wordt niet uniek herkend. "
            f"df_test count={step2_test_count}, "
            f"df_overview count={step2_overview_count}."
        )
    elif STATUS_ORDER.get(step2_status, 0) < 1:
        st.error("Race voldoet niet aan row_created.")
    else:
        is_sprint_enrich = st.checkbox(
            "IsSprintWeekend",
            value=bool(
                int(pd.to_numeric(
                    df_test.loc[get_target_mask(df_test, selected_year, selected_slug), "IsSprintWeekend"],
                    errors="coerce"
                ).fillna(0).iloc[0])
            ),
            key="enrich_sprint"
        )

        if st.button("Verrijk circuit- en regimefeatures", type="primary"):
            before_test = df_test_step2.copy()
            before_overview = df_overview_step2.copy()
            try:
                enrich_result = canonical_prepare_circuit_regime_ready(
                    selected_year,
                    selected_slug,
                    is_sprint_weekend=bool(is_sprint_enrich),
                )
            except Exception as exc:
                st.error("Circuit/regimeverrijking geblokkeerd.")
                st.code(str(exc))
                st.stop()
            reread_test = helpers.safe_read_csv(DF_TEST_PATH)
            reread_overview = helpers.safe_read_csv(DF_OVERVIEW_PATH)
            st.success("RaceStatus = circuit_regime_ready")
            show_before_after(
                before_test,
                reread_test,
                before_overview,
                reread_overview,
                selected_year,
                selected_slug,
            )
            if enrich_result.get("feature_log"):
                st.write(enrich_result["feature_log"])
            st.rerun()


    # TRUSTED_CIRCUIT_MASTER_UI_V1: UI only; canonical service owns validation/writes.
    st.subheader("Controle centrale circuitgegevens")
    st.caption(
        "Circuitgegevens komen uit de centrale referentie. "
        "Bestaande eventuitzonderingen worden automatisch toegepast."
    )
    try:
        circuit_report = canonical_inspect_circuit_reference(selected_year, selected_slug)
        if circuit_report["issues"]:
            st.warning("Circuitvoorbereiding is nog niet compleet. Los onderstaande punten op.")
            st.dataframe(pd.DataFrame(circuit_report["issues"]), hide_index=True)
        elif circuit_report["completed"]:
            st.success("Circuitgegevens compleet en gecontroleerd tegen de centrale referentie.")
        else:
            circuit_result = canonical_save_circuit_reference_validation(selected_year, selected_slug)
            if circuit_result.get("changed"):
                st.rerun()
            st.success("Circuitgegevens compleet en gecontroleerd tegen de centrale referentie.")
    except Exception as exc:
        st.error("Automatische circuitcontrole geblokkeerd.")
        st.code(str(exc))

# ============================================================
# Tab 3 — qualifying_ready
# ============================================================
with tabs[2]:
    st.header("Stap 3 — Qualifying-update")

    if not current_exists:
        st.error("Maak de race eerst aan.")
    elif STATUS_ORDER.get(current_status, 0) < 2:
        st.error("Circuit/regimeverrijking moet eerst volledig slagen.")
    else:
        # IsSprintWeekend ophalen uit de geselecteerde targetrij
        mt_current = require_one(
            df_test,
            selected_year,
            selected_slug,
            "df_test",
        )

        sprint_flag = pd.to_numeric(
            df_test.loc[mt_current, "IsSprintWeekend"],
            errors="coerce",
        ).fillna(0).iloc[0]

        is_sprint_weekend = int(sprint_flag) == 1

        with st.form("qualifying_form"):
            q_mean = st.number_input("Q_airtemp_mean", value=20.0)
            q_max = st.number_input("Q_airtemp_max", value=22.0)
            q_rain = st.number_input("Q_rain_sum", value=0.0, min_value=0.0)
            q_hum = st.number_input(
                "Q_humidity_mean",
                value=60.0,
                min_value=0.0,
                max_value=100.0,
            )

            if is_sprint_weekend:
                st.info(
                    "Sprintweekend gedetecteerd. "
                    "SprintFastestLap_sec is verplicht."
                )

                sprint_fastest = st.number_input(
                    "SprintFastestLap_sec",
                    value=80.0,
                    min_value=0.001,
                )
            else:
                sprint_fastest = None

            pole = st.number_input(
                "PoleTime_sec",
                value=80.0,
                min_value=0.001,
            )

            compound = st.text_input(
                "pole_compound",
                value="soft",
            )

            c_index = st.number_input(
                "pole_c_index",
                value=0,
                min_value=0,
                step=1,
            )

            submitted = st.form_submit_button(
                "Qualifyinggegevens opslaan"
            )

        if submitted:
            before_test = df_test.copy()
            before_overview = df_overview.copy()
            try:
                qualifying_result = canonical_save_qualifying_ready(
                    selected_year,
                    selected_slug,
                    q_airtemp_mean=float(q_mean),
                    q_airtemp_max=float(q_max),
                    q_rain_sum=float(q_rain),
                    q_humidity_mean=float(q_hum),
                    pole_time_sec=float(pole),
                    pole_compound=compound,
                    pole_c_index=int(c_index),
                    sprint_fastest_lap_sec=sprint_fastest,
                )
            except Exception as exc:
                st.error("Qualifying-update geblokkeerd.")
                st.code(str(exc))
                st.stop()
            reread_test = helpers.safe_read_csv(DF_TEST_PATH)
            reread_overview = helpers.safe_read_csv(DF_OVERVIEW_PATH)
            st.success("RaceStatus = qualifying_ready")
            show_before_after(
                before_test,
                reread_test,
                before_overview,
                reread_overview,
                selected_year,
                selected_slug,
            )
            st.rerun()

# ============================================================
# Tab 4 — Race Weather
# ============================================================
# SINGLE_RACE_WEATHER_FORM_V1



# ============================================================
# Tab 4 — RecordLap-correctie
# ============================================================
with tabs[4]:
    st.header("RecordLap_sec corrigeren")

    if not current_exists:
        st.error("Selecteer eerst een bestaande race.")
    else:
        mt = require_one(df_test, selected_year, selected_slug, "df_test")
        current_record = pd.to_numeric(
            df_test.loc[mt, "RecordLap_sec"], errors="coerce"
        ).iloc[0]
        if pd.isna(current_record) or float(current_record) <= 0:
            st.warning(
                "RecordLap_sec ontbreekt of is ongeldig. "
                "Vul hieronder een geldige waarde groter dan 0 in."
            )
            record_default = 0.001
        else:
            record_default = float(current_record)

        new_record = st.number_input(
            "Nieuwe RecordLap_sec",
            value=record_default,
            min_value=0.001,
            format="%.3f",
        )

        if st.button("RecordLap synchroon corrigeren"):
            before_test = df_test.copy()
            before_overview = df_overview.copy()
            try:
                recordlap_result = canonical_correct_recordlap(
                    selected_year,
                    selected_slug,
                    float(new_record),
                )
            except Exception as exc:
                st.error("RecordLap-correctie geblokkeerd.")
                st.code(str(exc))
                st.stop()
            reread_test = helpers.safe_read_csv(DF_TEST_PATH)
            reread_overview = helpers.safe_read_csv(DF_OVERVIEW_PATH)
            st.success("RecordLap_sec synchroon gecorrigeerd.")
            show_before_after(
                before_test,
                reread_test,
                before_overview,
                reread_overview,
                selected_year,
                selected_slug,
            )
            st.rerun()

# ============================================================
# Tab 6 — Prediction Readiness
# Workflowstap 5
# ============================================================
with tabs[5]:
    st.header("Stap 5 — Prediction Readiness Check")

    # --------------------------------------------------------
    # Altijd opnieuw lezen om stale Streamlit-state te voorkomen
    # --------------------------------------------------------
    df_test_ready = helpers.safe_read_csv(
        DF_TEST_PATH
    )

    df_overview_ready = helpers.safe_read_csv(
        DF_OVERVIEW_PATH
    )

    ready_test_count = int(
        get_target_mask(
            df_test_ready,
            selected_year,
            selected_slug,
        ).sum()
    )

    ready_overview_count = int(
        get_target_mask(
            df_overview_ready,
            selected_year,
            selected_slug,
        ).sum()
    )

    st.write(
        "Targetherkenning:",
        {
            "df_test": ready_test_count,
            "df_overview": ready_overview_count,
        },
    )

    if (
        ready_test_count != 1
        or ready_overview_count != 1
    ):
        st.error(
            "Harde stop: targetrace moet exact één keer "
            "voorkomen in df_test en df_overview."
        )

    else:
        readiness_current_status = infer_status(
            df_test_ready,
            df_overview_ready,
            selected_year,
            selected_slug,
        )

        st.write(
            "Huidige status:",
            readiness_current_status or "onbekend",
        )

        # ----------------------------------------------------
        # Vorige stap moet race_weather_ready zijn
        # ----------------------------------------------------
        if STATUS_ORDER.get(
            readiness_current_status,
            0,
        ) < STATUS_ORDER["race_weather_ready"]:

            st.error(
                "Prediction Readiness is geblokkeerd. "
                "De race moet eerst race_weather_ready zijn."
            )

        else:
            # ------------------------------------------------
            # Checks uitvoeren zonder write
            # ------------------------------------------------
            readiness_errors = canonical_validate_prediction_ready_rows(df_test_ready, df_overview_ready, selected_year, selected_slug)

            mt_ready = require_one(
                df_test_ready,
                selected_year,
                selected_slug,
                "df_test",
            )

            mo_ready = require_one(
                df_overview_ready,
                selected_year,
                selected_slug,
                "df_overview",
            )

            row_test_ready = (
                df_test_ready
                .loc[mt_ready]
                .iloc[0]
            )

            row_overview_ready = (
                df_overview_ready
                .loc[mo_ready]
                .iloc[0]
            )

            # ------------------------------------------------
            # Visuele checklist
            # ------------------------------------------------
            checklist = []

            def add_check(label, ok, detail=""):
                checklist.append({
                    "Check": label,
                    "Status": "PASS" if ok else "FAIL",
                    "Detail": detail,
                })

            add_check(
                "Unieke targetrij df_test",
                ready_test_count == 1,
                f"count={ready_test_count}",
            )

            add_check(
                "Unieke targetrij df_overview",
                ready_overview_count == 1,
                f"count={ready_overview_count}",
            )

            add_check(
                "Geen duplicates df_test",
                duplicate_count(df_test_ready) == 0,
                f"duplicates={duplicate_count(df_test_ready)}",
            )

            add_check(
                "Geen duplicates df_overview",
                duplicate_count(df_overview_ready) == 0,
                f"duplicates={duplicate_count(df_overview_ready)}",
            )

            add_check(
                "Circuit & regime compleet",
                len(
                    canonical_validate_circuit_regime_rows(df_test_ready, df_overview_ready, selected_year, selected_slug)
                ) == 0,
            )

            add_check(
                "Qualifying compleet",
                len(
                    canonical_validate_qualifying_ready_rows(df_test_ready, df_overview_ready, selected_year, selected_slug)
                ) == 0,
            )

            add_check(
                "Race weather compleet",
                len(
                    canonical_validate_race_weather_ready_rows(df_test_ready, df_overview_ready, selected_year, selected_slug)
                ) == 0,
            )

            sprint_flag_ready = pd.to_numeric(
                row_test_ready.get(
                    "IsSprintWeekend"
                ),
                errors="coerce",
            )

            if (
                pd.notna(sprint_flag_ready)
                and int(sprint_flag_ready) == 1
            ):
                sprint_value = pd.to_numeric(
                    row_test_ready.get(
                        "SprintFastestLap_sec"
                    ),
                    errors="coerce",
                )

                sprint_ok = (
                    pd.notna(sprint_value)
                    and sprint_value > 0
                )

                add_check(
                    "SprintFastestLap_sec",
                    sprint_ok,
                    "Verplicht omdat IsSprintWeekend=1",
                )

            else:
                add_check(
                    "SprintFastestLap_sec",
                    True,
                    "Niet verplicht omdat IsSprintWeekend=0",
                )

            add_check(
                "Volledige Prediction Readiness",
                len(readiness_errors) == 0,
                (
                    "Alle checks geslaagd"
                    if not readiness_errors
                    else f"{len(readiness_errors)} fout(en)"
                ),
            )

            checklist_df = pd.DataFrame(
                checklist
            )

            st.subheader(
                "Prediction readiness checklist"
            )

            st.dataframe(
                checklist_df,
                width="stretch",
                hide_index=True,
            )

            # ------------------------------------------------
            # Target snapshot
            # ------------------------------------------------
            readiness_display_cols = [
                "Year",
                "gp_slug",
                "Circuit",
                "YearOffset",
                "RecordLap_sec",
                "IsSprintWeekend",
                "SprintFastestLap_sec",
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
                "RaceStatus",
            ]

            readiness_display_cols = [
                c
                for c in readiness_display_cols
                if c in df_test_ready.columns
            ]

            st.subheader(
                "Prediction input snapshot"
            )

            st.dataframe(
                df_test_ready.loc[
                    mt_ready,
                    readiness_display_cols,
                ],
                width="stretch",
            )

            # ------------------------------------------------
            # Wet-weather waarschuwing
            # ------------------------------------------------
            r_rain_ready = pd.to_numeric(
                row_test_ready.get(
                    "R_rain_sum"
                ),
                errors="coerce",
            )

            if (
                pd.notna(r_rain_ready)
                and r_rain_ready >= 1.0
            ):
                st.warning(
                    "R_rain_sum >= 1.0. "
                    "Prediction-ready betekent alleen dat de "
                    "modelinput technisch compleet is. "
                    "Forecast rain is niet automatisch gelijk aan "
                    "pace-relevante natte omstandigheden. "
                    "Bij de toekomstige modelrun is een "
                    "plausibility challenge en audit-only "
                    "wet-scenario vereist."
                )

            # ------------------------------------------------
            # Fouten tonen
            # ------------------------------------------------
            if readiness_errors:

                st.error(
                    "Race is nog NIET prediction_ready."
                )

                for err in readiness_errors:
                    st.write("❌", err)

            else:
                st.success(
                    "Alle Prediction Readiness checks zijn geslaagd."
                )

                # STEP 10.3 UI CLEANUP — prediction readiness deduplicated
                st.caption(
                    "Prediction readiness wordt via de centrale "
                    "readiness-action verderop bevestigd."
                )



# ============================================================


# ============================================================
# Tab 4 — Canonical Race Weather Audit Sync
# ============================================================
with tabs[3]:

    st.divider()

    st.subheader(
        "Stap 4 — Race weather invoeren"
    )

    st.caption(
        "Slaat R-weather synchroon op in df_test en df_overview. "
        "RaceStatus wordt pas na succesvolle post-write validatie "
        "race_weather_ready."
    )

    weather_test = helpers.safe_read_csv(
        DF_TEST_PATH
    )

    weather_overview = helpers.safe_read_csv(
        DF_OVERVIEW_PATH
    )

    weather_test_count = int(
        get_target_mask(
            weather_test,
            selected_year,
            selected_slug,
        ).sum()
    )

    weather_overview_count = int(
        get_target_mask(
            weather_overview,
            selected_year,
            selected_slug,
        ).sum()
    )

    if (
        weather_test_count != 1
        or weather_overview_count != 1
    ):

        st.error(
            "Target moet exact één keer bestaan in "
            "df_test én df_overview."
        )

    else:

        weather_mask = require_one(
            weather_test,
            selected_year,
            selected_slug,
            "df_test",
        )

        weather_row = (
            weather_test
            .loc[weather_mask]
            .iloc[0]
        )

        weather_status = str(
            weather_row.get(
                "RaceStatus",
                "",
            )
        ).strip()

        st.write(
            f"Huidige status: `{weather_status}`"
        )

        if weather_status != "qualifying_ready":

            st.info(
                "Dit invoerformulier is beschikbaar bij qualifying_ready. "
                "Bij een latere status zijn de weergegevens al opgeslagen; "
                "ga verder met de volgende stap."
            )

        else:

            def _weather_default(
                column,
                default,
            ):
                value = pd.to_numeric(
                    weather_row.get(
                        column
                    ),
                    errors="coerce",
                )

                if pd.isna(value):
                    return float(default)

                return float(value)


            with st.form(
                "canonical_race_weather_form"
            ):

                rw_air_mean = (
                    st.number_input(
                        "R air temperature mean (°C)",
                        value=_weather_default(
                            "R_airtemp_mean",
                            25.0,
                        ),
                        step=0.1,
                    )
                )

                rw_air_max = (
                    st.number_input(
                        "R air temperature max (°C)",
                        value=_weather_default(
                            "R_airtemp_max",
                            28.0,
                        ),
                        step=0.1,
                    )
                )

                rw_rain_sum = (
                    st.number_input(
                        "R rain sum",
                        min_value=0.0,
                        value=max(
                            0.0,
                            _weather_default(
                                "R_rain_sum",
                                0.0,
                            ),
                        ),
                        step=0.1,
                    )
                )

                rw_humidity = (
                    st.number_input(
                        "R humidity mean (%)",
                        min_value=0.0,
                        max_value=100.0,
                        value=min(
                            100.0,
                            max(
                                0.0,
                                _weather_default(
                                    "R_humidity_mean",
                                    60.0,
                                ),
                            ),
                        ),
                        step=0.5,
                    )
                )

                existing_source = str(
                    weather_row.get(
                        "WeatherSourceMode",
                        "",
                    )
                ).strip()

                rw_source = st.selectbox(
                    "WeatherSourceMode (verplicht)",
                    ["", "manual_verified", "api_forecast", "api_observed", "derived_forecast"],
                    index=(
                        ["", "manual_verified", "api_forecast", "api_observed", "derived_forecast"].index(existing_source)
                        if existing_source in {
                            "manual_verified", "api_forecast",
                            "api_observed", "derived_forecast"
                        }
                        else 0
                    ),
                    format_func=lambda value: {
                        "": "Kies de gebruikte weerbron…",
                        "manual_verified": "Handmatig gecontroleerde gegevens",
                        "api_forecast": "API-forecast zonder interpolatie",
                        "api_observed": "API-waarnemingen",
                        "derived_forecast": "Afgeleide/geïnterpoleerde forecast",
                    }[value],
                )

                weather_submit = (
                    st.form_submit_button(
                        "Race weather opslaan"
                    )
                )

            if weather_submit:
                if not rw_source:
                    st.error(
                        "Kies eerst de gebruikte weerbron. "
                        "Er is niets opgeslagen."
                    )
                    st.stop()

                try:

                    weather_result = (
                        canonical_save_race_weather_audit(
                            year=selected_year,
                            gp_slug=selected_slug,
                            r_airtemp_mean=rw_air_mean,
                            r_airtemp_max=rw_air_max,
                            r_rain_sum=rw_rain_sum,
                            r_humidity_mean=rw_humidity,
                            weather_source_mode=rw_source,
                        )
                    )

                except Exception as exc:

                    st.error(
                        "R-weather save geblokkeerd."
                    )

                    with st.expander(
                        "Technische details",
                        expanded=False,
                    ):
                        st.code(
                            str(exc)
                        )

                else:

                    st.success(
                        "R-weather audit synchroon opgeslagen. "
                        "RaceStatus = race_weather_ready."
                    )

                    st.dataframe(
                        pd.DataFrame(
                            [
                                weather_result[
                                    "values"
                                ]
                            ]
                        ),
                        width="stretch",
                    )

                    st.rerun()




with tabs[3]:
    st.subheader("Racecontext bevestigen")
    context_test = helpers.safe_read_csv(DF_TEST_PATH)
    context_overview = helpers.safe_read_csv(DF_OVERVIEW_PATH)
    context_mt = get_target_mask(context_test, selected_year, selected_slug)
    context_mo = get_target_mask(context_overview, selected_year, selected_slug)
    context_allowed = False
    if int(context_mt.sum()) == int(context_mo.sum()) == 1:
        context_row = context_test.loc[context_mt].iloc[0]
        context_other = context_overview.loc[context_mo].iloc[0]
        context_allowed = str(context_row.get("RaceStatus")) == str(context_other.get("RaceStatus")) == "race_weather_ready"
        context_allowed = context_allowed and pd.isna(context_row.get("FeatureSnapshotTimestampUTC", np.nan))
    if st.button("Bevestig racecontext", key="confirm_canonical_race_context", disabled=not context_allowed):
        try:
            canonical_prepare_race_context_plausibility_ready(selected_year, selected_slug)
        except Exception as exc:
            st.error("Racecontext bevestigen geblokkeerd.")
            st.code(str(exc))
            st.stop()
        st.success("Racecontext gevalideerd en bevestigd.")
        st.rerun()

# Tab 5 — Diagnostiek
# ============================================================

# ============================================================
# Tab 6 extra — Step 5 SuggestedModel audit
# ============================================================
with tabs[5]:

    st.divider()

    st.header(
        "Model recommendation — audit only"
    )

    st.caption(
        "Bepaalt welk model op basis van uitsluitend eerdere "
        "races en racecontext de beste kandidaat is. "
        "De operationele voorspelling blijft GBR."
    )

    # Altijd opnieuw lezen
    suggestion_overview = helpers.safe_read_csv(
        DF_OVERVIEW_PATH
    )

    suggestion_test = helpers.safe_read_csv(
        DF_TEST_PATH
    )

    target_test_count = int(
        get_target_mask(
            suggestion_test,
            selected_year,
            selected_slug,
        ).sum()
    )

    target_overview_count = int(
        get_target_mask(
            suggestion_overview,
            selected_year,
            selected_slug,
        ).sum()
    )

    if (
        target_test_count != 1
        or target_overview_count != 1
    ):

        st.error(
            "Targetrace moet exact één keer voorkomen "
            "in df_test en df_overview."
        )

    else:
        current_prediction_status = infer_status(
            suggestion_test,
            suggestion_overview,
            selected_year,
            selected_slug,
        )

        if STATUS_ORDER.get(
            current_prediction_status,
            0,
        ) < STATUS_ORDER["prediction_ready"]:

            st.warning(
                "Model recommendation is geblokkeerd totdat "
                "de race prediction_ready is."
            )

        else:
            try:
                suggestion = (
                    build_future_model_suggestion(
                        suggestion_overview,
                        selected_year,
                        selected_slug,
                        min_n_segment=int(
                            MIN_N_SEGMENT
                        ),
                        min_improvement_ape=float(
                            MIN_IMPROVEMENT_APE
                        ),
                    )
                )

            except Exception as exc:

                st.error(
                    "Model recommendation kon niet worden "
                    f"berekend: {exc}"
                )

                suggestion = None

            if suggestion is not None:

                s1, s2, s3, s4 = st.columns(4)

                s1.metric(
                    "Suggested model",
                    suggestion[
                        "SuggestedModel"
                    ],
                )

                s2.metric(
                    "Segment n",
                    suggestion[
                        "SuggestedSegment_n"
                    ],
                )

                suggested_ape = suggestion[
                    "Suggested_APE"
                ]

                s3.metric(
                    "Historical mean APE",
                    (
                        f"{suggested_ape:.3f}%"
                        if pd.notna(
                            suggested_ape
                        )
                        else "n/a"
                    ),
                )

                s4.metric(
                    "Improvement vs GBR",
                    (
                        f"{suggestion['Suggested_vs_GBR_APE']:+.3f} pp"
                    ),
                )

                st.write(
                    "**Segment:**",
                    suggestion[
                        "SuggestedSegment"
                    ],
                )

                st.write(
                    "**Reden:**",
                    suggestion[
                        "SuggestedModelReason"
                    ],
                )

                # --------------------------------------------
                # Alle contextanalyses tonen
                # --------------------------------------------
                audit_df = suggestion[
                    "audit_df"
                ].copy()

                if not audit_df.empty:

                    for col in [
                        "GBR_APE",
                        "XGB_APE",
                        "CAT_APE",
                        "Suggested_APE",
                        "Improvement_vs_GBR",
                    ]:
                        if col in audit_df.columns:
                            audit_df[col] = (
                                pd.to_numeric(
                                    audit_df[col],
                                    errors="coerce",
                                )
                                .round(3)
                            )

                    st.subheader(
                        "Context evidence"
                    )

                    st.dataframe(
                        audit_df,
                        width="stretch",
                        hide_index=True,
                    )

                # --------------------------------------------
                # Duidelijke operational safeguard
                # --------------------------------------------
                st.info(
                    "Audit-only: deze aanbeveling wijzigt "
                    "Pred_GBR, Pred_XGB, Pred_CAT of "
                    "PredictedFastestLap_sec niet."
                )

                # --------------------------------------------
                # Opslaan
                # --------------------------------------------
                if st.button(
                    "SuggestedModel audit opslaan",
                    key="save_suggested_model_audit",
                    type="primary",
                ):
                    before_overview = suggestion_overview.copy()
                    try:
                        audit_result = canonical_save_suggested_model_audit(
                            selected_year,
                            selected_slug,
                            suggestion,
                        )
                    except Exception as exc:
                        st.error("SuggestedModel audit opslaan geblokkeerd.")
                        with st.expander(
                            "Technische details",
                            expanded=False,
                        ):
                            st.code(str(exc))
                        st.stop()
                    reread_overview = helpers.safe_read_csv(DF_OVERVIEW_PATH)
                    st.success("SuggestedModel audit opgeslagen.")
                    st.info(
                        "Alleen auditvelden in df_overview zijn via de "
                        "canonical evaluation writer bijgewerkt."
                    )






# ============================================================
# Prediction readiness — Feature snapshot freeze
# ============================================================
with tabs[5]:

    st.divider()

    st.subheader(
        "Feature snapshot"
    )

    st.caption(
        "Bevries de definitieve pre-race featureset. "
        "Bevries de inputs vóór de aangekondigde start. Na bevriezen zijn wijzigingen uitsluitend audit-only."
    )

    snapshot_test = helpers.safe_read_csv(
        DF_TEST_PATH
    )

    snapshot_overview = helpers.safe_read_csv(
        DF_OVERVIEW_PATH
    )

    snapshot_test_count = int(
        get_target_mask(
            snapshot_test,
            selected_year,
            selected_slug,
        ).sum()
    )

    snapshot_overview_count = int(
        get_target_mask(
            snapshot_overview,
            selected_year,
            selected_slug,
        ).sum()
    )

    if (
        snapshot_test_count != 1
        or snapshot_overview_count != 1
    ):

        st.error(
            "Feature snapshot niet mogelijk: "
            "target moet exact één keer bestaan in "
            "df_test en df_overview."
        )

    else:

        ms = require_one(
            snapshot_test,
            selected_year,
            selected_slug,
            "df_test",
        )

        snapshot_row = (
            snapshot_test
            .loc[ms]
            .iloc[0]
        )

        snapshot_status = str(
            snapshot_row.get(
                "RaceStatus",
                "",
            )
        ).strip()

        cutoff_display = str(
            snapshot_row.get(
                "AnnouncedRaceStartUTC",
                "",
            )
        )

        existing_snapshot = str(
            snapshot_row.get(
                "FeatureSnapshotTimestampUTC",
                "",
            )
        )

        st.write(
            f"Huidige status: `{snapshot_status}`"
        )

        st.write(
            f"Aangekondigde racestart: `{cutoff_display}`"
        )

        if (
            existing_snapshot
            and existing_snapshot.lower()
            not in {
                "nan",
                "none",
                "",
            }
        ):

            st.success(
                "Feature snapshot reeds bevroren: "
                f"{existing_snapshot}"
            )

        allowed_snapshot_statuses = {
            "race_weather_ready",
            "race_context_plausibility_ready",
        }

        if snapshot_status not in (
            allowed_snapshot_statuses
        ):

            st.warning(
                "Freeze is pas toegestaan vanaf "
                "race_weather_ready."
            )

        else:

            # AUTOMATIC_QUALIFYING_UI_V1
            st.caption("Kwalificatie-einde en bron worden automatisch opgehaald bij Freeze. Bij ontbrekend bewijs wordt niets bevroren.")

            if st.button(
                "Freeze feature snapshot",
                key="freeze_feature_snapshot_button",
            ):

                try:

                    snapshot_result = (
                        canonical_freeze_feature_snapshot(
                            year=selected_year,
                            gp_slug=selected_slug,
                        )
                    )

                except Exception as exc:

                    st.error(
                        "Feature snapshot geblokkeerd."
                    )

                    with st.expander(
                        "Technische details",
                        expanded=False,
                    ):
                        st.code(
                            str(exc)
                        )

                else:

                    st.success(
                        "Definitieve featureset bevroren."
                    )

                    st.write(
                        "FeatureSnapshotTimestampUTC:",
                        snapshot_result[
                            "FeatureSnapshotTimestampUTC"
                        ],
                    )

                    st.write(
                        "AnnouncedRaceStartUTC:",
                        snapshot_result.get("AnnouncedRaceStartUTC", "legacy timing"),
                    )

                    st.rerun()





# ============================================================
# Prediction readiness — Future Event Registry
# ============================================================
with tabs[5]:

    st.divider()

    st.subheader(
        "Future Event Registry"
    )

    st.caption(
        "RegistryEligible wordt alleen 1 als identity, layout, "
        "RecordLap, timing, actual-leeg en target uniqueness "
        "allemaal voldoen."
    )

    registry_test = helpers.safe_read_csv(
        DF_TEST_PATH
    )

    registry_overview = helpers.safe_read_csv(
        DF_OVERVIEW_PATH
    )

    test_count_registry = int(
        get_target_mask(
            registry_test,
            selected_year,
            selected_slug,
        ).sum()
    )

    overview_count_registry = int(
        get_target_mask(
            registry_overview,
            selected_year,
            selected_slug,
        ).sum()
    )

    if (
        test_count_registry != 1
        or overview_count_registry != 1
    ):

        st.error(
            "Registry kan niet worden gebouwd: "
            "target moet exact één keer bestaan in "
            "df_test én df_overview."
        )

    else:

        eligibility_preview = (
            canonical_build_prediction_readiness_report(
                selected_year,
                selected_slug,
            )
        )

        if eligibility_preview.source_eligible:

            st.success(
                "Registry eligibility preview: ✅ source eligible"
            )

        else:

            st.warning(
                "Registry eligibility preview: ⚠️ source nog niet eligible"
            )

            for reason in eligibility_preview.blockers:

                st.write(
                    "•",
                    reason,
                )

        if st.button(
            "Maak / update future event registry",
            key="build_future_event_registry_button",
        ):

            try:

                registry_result = (
                    sync_event_registry(source_eligible=canonical_build_prediction_readiness_report(selected_year, selected_slug).source_eligible, year=selected_year, gp_slug=selected_slug)
                )

            except Exception as exc:

                st.error(
                    "Registry update geblokkeerd."
                )

                with st.expander(
                    "Technische details",
                    expanded=False,
                ):
                    st.code(
                        str(exc)
                    )

            else:

                if (
                    registry_result.registry_eligible
                    == 1
                ):

                    st.success(
                        "Registry bijgewerkt: "
                        "RegistryEligible = 1"
                    )

                else:

                    st.warning(
                        "Registry bijgewerkt, maar target "
                        "is nog niet eligible."
                    )

                    with st.expander(
                        "Technische details",
                        expanded=False,
                    ):
                        st.code(
                            registry_state_message(registry_result)
                        )

                st.write(
                    "race_event_id:",
                    registry_result.race_event_id,
                )

                st.write(
                    "Registry:",
                    str(CONFIG.registry_path),
                )

                st.write(
                    "Eligible registry:",
                    str(CONFIG.registry_eligible_mirror_path),
                )





# ============================================================
# Prediction readiness — Canonical Readiness Panel
# ============================================================
with tabs[5]:

    st.divider()

    st.header(
        "Readiness panel"
    )

    st.caption(
        "prediction_ready wordt pas bevestigd wanneer "
        "alle pre-race gates groen zijn."
    )

    panel_test = helpers.safe_read_csv(
        DF_TEST_PATH
    )

    panel_overview = helpers.safe_read_csv(
        DF_OVERVIEW_PATH
    )

    # --------------------------------------------------------
    # Eén centrale workflow-samenvatting
    # --------------------------------------------------------
    workflow_state = (
        build_workflow_ux_state(
            panel_test,
            panel_overview,
            selected_year,
            selected_slug,
        )
    )

    if workflow_state["ok"]:

        current_workflow_status = (
            workflow_state[
                "RaceStatus"
            ]
        )

        next_workflow_status = (
            workflow_state[
                "next_status"
            ]
        )

        st.markdown(
            f"### Workflowstatus: `{current_workflow_status}`"
        )

        st.progress(
            min(
                1.0,
                max(
                    0.0,
                    workflow_state[
                        "progress"
                    ],
                ),
            )
        )

        if next_workflow_status:

            st.info(
                "Volgende actie: "
                f"**{workflow_state['action']}** "
                f"→ `{next_workflow_status}`"
            )

        else:

            st.success(
                "Workflow volledig afgerond."
            )

    else:

        st.error(
            "Workflowstatus is inconsistent."
        )

        st.write(
            "Herstelactie:",
            workflow_state[
                "action"
            ],
        )

        st.caption(
            workflow_state[
                "detail"
            ]
        )

    panel_test_count = int(
        get_target_mask(
            panel_test,
            selected_year,
            selected_slug,
        ).sum()
    )

    panel_overview_count = int(
        get_target_mask(
            panel_overview,
            selected_year,
            selected_slug,
        ).sum()
    )

    if (
        panel_test_count != 1
        or panel_overview_count != 1
    ):

        st.error(
            "❌ HARD STOP — target moet exact één keer "
            "bestaan in df_test én df_overview."
        )

        st.write(
            "Actie: herstel duplicate/missing target."
        )

    else:

        readiness_report = canonical_build_prediction_readiness_report(
            selected_year,
            selected_slug,
        )

        # ----------------------------------------------------
        # Eén zichtbaar panel
        # ----------------------------------------------------
        for gate_key, gate in readiness_report.gates.items():

            if gate.ok:

                icon = "✅"
                label = "OK"

            else:

                icon = "❌"
                label = "BLOKKEERT"

            st.markdown(
                f"**{icon} {gate.name} — {label}**"
            )

            if not gate.ok:

                st.caption(
                    gate.detail
                )

        # ----------------------------------------------------
        # Canonical readiness summary
        # ----------------------------------------------------
        if readiness_report.blockers:

            st.error(
                "❌ Prediction readiness geblokkeerd."
            )

            for blocker in readiness_report.blockers:

                st.caption(
                    blocker
                )

        else:

            st.success(
                "✅ Geen readiness blockers."
            )

        st.caption(
            "Registry aanwezig: "
            f"{readiness_report.registry_present} | "
            "Registry eligible: "
            f"{readiness_report.registry_eligible}"
        )


        all_ready = bool(readiness_report.ok)

        st.divider()

        # ----------------------------------------------------
        # Administratieve herstelacties
        # ----------------------------------------------------
        col1, col2, col3 = st.columns(
            3
        )

        # STEP 10.2 UI CLEANUP — snapshot deduplicated
        with col1:
            st.caption(
                "Feature snapshot wordt via de primaire "
                "snapshot-control hierboven uitgevoerd."
            )

        with col2:

            if st.button(
                "Maak registry eligible",
                key="readiness_registry_update",
            ):

                try:

                    registry_result = (
                        sync_event_registry(source_eligible=canonical_build_prediction_readiness_report(selected_year, selected_slug).source_eligible, year=selected_year, gp_slug=selected_slug)
                    )

                except Exception as exc:

                    st.error(
                        str(exc)
                    )

                else:

                    if (
                        registry_result.registry_eligible
                        == 1
                    ):

                        st.success(
                            "Registry eligible."
                        )

                    else:

                        st.warning(
                            registry_state_message(registry_result)
                        )

                    st.rerun()

        with col3:

            if st.button(
                "Refresh readiness",
                key="readiness_refresh",
            ):

                st.rerun()

        st.info(
            "Andere herstelacties: "
            "Event timing → Nieuwe GP; "
            "RecordLap / layout / identity → Circuit & regime; "
            "weather → Race Weather; "
            "plausibility → race-context plausibility."
        )

        # ----------------------------------------------------
        # Definitieve prediction_ready bevestiging
        # ----------------------------------------------------
        if all_ready:

            st.success(
                "✅ Alle prediction-readiness gates zijn groen."
            )

        else:

            st.warning(
                "⚠️ prediction_ready blijft geblokkeerd "
                "tot alle gates groen zijn."
            )

        confirm_ready = st.button(
            "Bevestig prediction_ready",
            key="confirm_prediction_ready_button",
            disabled=not all_ready,
        )

        if confirm_ready:

            try:

                result = (
                    prepare_prediction_ready(
                        year=selected_year,
                        gp_slug=selected_slug,
                    )
                )

            except Exception as exc:

                st.error(
                    "prediction_ready kon niet worden bevestigd."
                )

                with st.expander(
                    "Technische details",
                    expanded=False,
                ):
                    st.code(
                        str(exc)
                    )

            else:

                st.success(
                    "RaceStatus succesvol naar prediction_ready gezet "
                    "in df_test en df_overview."
                )

                st.write(
                    "RegistryEligible:",
                    result.registry_eligible,
                )

                st.rerun()





# ============================================================
# Prediction readiness — Run final prediction
# ============================================================
with tabs[5]:

    st.divider()

    st.header(
        "Final prediction"
    )

    st.caption(
        "Draait GBR + XGB + CAT. "
        "PredictedFastestLap_sec blijft altijd Pred_GBR. "
        "Predictionwaarden worden alleen naar df_overview geschreven."
    )

    final_test = helpers.safe_read_csv(
        DF_TEST_PATH
    )

    final_overview = helpers.safe_read_csv(
        DF_OVERVIEW_PATH
    )

    final_test_count = int(
        get_target_mask(
            final_test,
            selected_year,
            selected_slug,
        ).sum()
    )

    final_overview_count = int(
        get_target_mask(
            final_overview,
            selected_year,
            selected_slug,
        ).sum()
    )

    final_prediction_allowed = False

    if (
        final_test_count != 1
        or final_overview_count != 1
    ):

        st.error(
            "❌ Final prediction geblokkeerd: "
            "target moet exact één keer bestaan."
        )

    else:

        mt_final = require_one(
            final_test,
            selected_year,
            selected_slug,
            "df_test",
        )

        mo_final = require_one(
            final_overview,
            selected_year,
            selected_slug,
            "df_overview",
        )

        final_status_test = str(
            final_test
            .loc[mt_final]
            .iloc[0]
            .get(
                "RaceStatus",
                "",
            )
        ).strip()

        final_status_overview = str(
            final_overview
            .loc[mo_final]
            .iloc[0]
            .get(
                "RaceStatus",
                "",
            )
        ).strip()

        final_prediction_allowed = (
            final_status_test
            == "prediction_ready"
            and final_status_overview
            == "prediction_ready"
        )

        if final_prediction_allowed:
            # Gebruik de bestaande canonical immutable-runcontrole.
            from f1_pipeline.prediction import _assert_no_existing_final_prediction
            try:
                _assert_no_existing_final_prediction(
                    selected_year, selected_slug
                )
            except (ValueError, RuntimeError, OSError):
                final_prediction_allowed = False
                st.info(
                    "Een bestaande finale run of auditblokkade verhindert "
                    "een nieuwe voorspelling."
                )


        st.write(
            "df_test status:",
            f"`{final_status_test}`",
        )

        st.write(
            "df_overview status:",
            f"`{final_status_overview}`",
        )

        if final_prediction_allowed:

            st.success(
                "✅ Final prediction mag worden uitgevoerd."
            )

        else:

            st.warning(
                "⚠️ Run final prediction wordt pas actief "
                "wanneer beide datasets prediction_ready zijn."
            )

    # STEP 8.5 UI CALL — prediction dashboard

    _prediction_model_log_ui = None

    try:
        _prediction_model_log_path = (
            helpers.get_model_log_path()
        )

        if Path(_prediction_model_log_path).exists():
            _prediction_model_log_ui = (
                helpers.safe_read_csv(
                    _prediction_model_log_path
                )
            )

    except Exception as _prediction_log_exc:
        st.caption(
            "Model-log auditinformatie kon niet worden "
            f"geladen voor de UI: {_prediction_log_exc}"
        )

    render_prediction_dashboard(
        df_test=df_test,
        df_overview=df_overview,
        model_log_df=_prediction_model_log_ui,
        year=selected_year,
        gp_slug=selected_slug,
        race_status=current_status,
    )

    run_final_button = st.button(
        "Run final prediction",
        key="run_final_prediction_button",
        disabled=not final_prediction_allowed,
    )

    if run_final_button:

        try:

            with st.spinner(
                "GBR + XGB + CAT draaien..."
            ):

                final_result = (
                    run_final_prediction(
                        year=selected_year,
                        gp_slug=selected_slug,
                    )
                )

        except Exception as exc:

            st.error(
                "Final prediction geblokkeerd / mislukt."
            )

            with st.expander(
                "Technische details",
                expanded=False,
            ):
                st.code(
                    str(exc)
                )

        else:

            st.success(
                "✅ Final prediction volledig geschreven "
                "en audit-logs gevalideerd."
            )

            c1, c2, c3, c4 = st.columns(
                4
            )

            c1.metric(
                "GBR",
                f"{final_result['Pred_GBR']:.3f}s",
            )

            c2.metric(
                "XGB",
                f"{final_result['Pred_XGB']:.3f}s",
            )

            c3.metric(
                "CAT",
                f"{final_result['Pred_CAT']:.3f}s",
            )

            c4.metric(
                "Operational",
                f"{final_result['PredictedFastestLap_sec']:.3f}s",
            )

            if final_result[
                "CATFallbackUsed"
            ]:

                st.warning(
                    "CAT fallback gebruikt."
                )

                with st.expander(
                    "Technische details",
                    expanded=False,
                ):
                    st.code(
                        final_result[
                            "CATFallbackReason"
                        ]
                    )

            else:

                st.success(
                    "CatBoost normaal uitgevoerd."
                )

            st.write(
                "RunID:",
                final_result[
                    "RunID"
                ],
            )

            st.write(
                "Feature count:",
                final_result[
                    "FeatureCount"
                ],
            )

            st.info(
                "RaceStatus blijft in stap 7 nog prediction_ready. "
                "Stap 8 promoveert pas naar predicted na de "
                "finale statusvalidatie."
            )
            st.session_state["final_prediction_result_step9f"] = final_result
            st.rerun()





# ============================================================
# Prediction readiness — Promote to predicted
# ============================================================
with tabs[5]:

    st.divider()

    st.header(
        "Prediction promotion"
    )

    st.caption(
        "RaceStatus wordt pas predicted nadat overview, "
        "immutable ledger en model_log opnieuw zijn gevalideerd."
    )

    promote_test = (
        helpers.safe_read_csv(
            DF_TEST_PATH
        )
    )

    promote_overview = (
        helpers.safe_read_csv(
            DF_OVERVIEW_PATH
        )
    )

    promote_test_count = int(
        get_target_mask(
            promote_test,
            selected_year,
            selected_slug,
        ).sum()
    )

    promote_overview_count = int(
        get_target_mask(
            promote_overview,
            selected_year,
            selected_slug,
        ).sum()
    )

    promotion_allowed = False
    artifact_validation = None

    if (
        promote_test_count != 1
        or promote_overview_count != 1
    ):

        st.error(
            "❌ Promotion geblokkeerd: target niet uniek."
        )

    else:

        mt_promote = require_one(
            promote_test,
            selected_year,
            selected_slug,
            "df_test",
        )

        mo_promote = require_one(
            promote_overview,
            selected_year,
            selected_slug,
            "df_overview",
        )

        promote_status_test = str(
            promote_test
            .loc[
                mt_promote
            ]
            .iloc[0]
            .get(
                "RaceStatus",
                "",
            )
        ).strip()

        promote_status_overview = str(
            promote_overview
            .loc[
                mo_promote
            ]
            .iloc[0]
            .get(
                "RaceStatus",
                "",
            )
        ).strip()

        if (
            promote_status_test
            == "prediction_ready"
            and promote_status_overview
            == "prediction_ready"
        ):

            try:

                artifact_validation = (
                    canonical_validate_final_prediction_artifacts(selected_year, selected_slug)
                )

            except Exception as exc:

                st.error(
                    "Artifactvalidatie kon niet worden uitgevoerd."
                )

                with st.expander(
                    "Technische details",
                    expanded=False,
                ):
                    st.code(
                        str(exc)
                    )

            else:

                promotion_allowed = bool(
                    artifact_validation[
                        "ok"
                    ]
                )

                if promotion_allowed:

                    st.success(
                        "✅ Overview + ledger + model_log "
                        "zijn volledig gevalideerd."
                    )

                    for check_name, check_ok in (
                        artifact_validation[
                            "checks"
                        ].items()
                    ):

                        st.write(
                            "✅"
                            if check_ok
                            else "❌",
                            check_name,
                        )

                else:

                    st.error(
                        "❌ Promotion gate faalt."
                    )

                    for err in (
                        artifact_validation[
                            "errors"
                        ]
                    ):

                        st.write(
                            "•",
                            err,
                        )

        elif (
            promote_status_test
            == "predicted"
            and promote_status_overview
            == "predicted"
        ):

            st.success(
                "Race is reeds predicted."
            )

        else:

            st.warning(
                "Promotion is alleen mogelijk wanneer "
                "df_test en df_overview beide prediction_ready zijn."
            )

    promote_button = st.button(
        "Promote to predicted",
        key="promote_prediction_to_predicted_button",
        disabled=not promotion_allowed,
    )

    if promote_button:

        try:

            promotion_result = (
                promote_prediction_to_predicted(
                    year=selected_year,
                    gp_slug=selected_slug,
                )
            )

        except Exception as exc:

            st.error(
                "Promotion naar predicted mislukt/geblokkeerd."
            )

            with st.expander(
                "Technische details",
                expanded=False,
            ):
                st.code(
                    str(exc)
                )

        else:

            st.success(
                "✅ RaceStatus = predicted in "
                "df_test én df_overview."
            )

            st.write(
                "RunID:",
                promotion_result[
                    "RunID"
                ],
            )

            st.rerun()



with tabs[6]:
    # STEP 8.7 UI CALL — audit diagnostics summary
    render_audit_diagnostics_summary(
        df_test=df_test,
        df_overview=df_overview,
        year=selected_year,
        gp_slug=selected_slug,
        race_status=current_status,
    )

    # STEP 10.4 UI CLEANUP — diagnostics collapsed
    with st.expander(
        "Technische diagnostics",
        expanded=False,
    ):
        st.header("Diagnostiek")
        show_diag(df_test, "df_test")
        show_diag(df_overview, "df_overview")
        if current_exists:
            st.write("**Targetrij df_test**")
            st.dataframe(df_test.loc[get_target_mask(df_test, selected_year, selected_slug)])
            st.write("**Targetrij df_overview**")
            st.dataframe(df_overview.loc[get_target_mask(df_overview, selected_year, selected_slug)])

            st.write("**Validatie row_created**")
            st.write(canonical_validate_row_created_contract(df_test, df_overview, selected_year, selected_slug).errors)
            st.write("**Validatie circuit_regime_ready**")
            st.write(canonical_validate_circuit_regime_rows(df_test, df_overview, selected_year, selected_slug))
            st.write("**Validatie qualifying_ready**")
            st.write(
                canonical_validate_qualifying_ready_rows(df_test, df_overview, selected_year, selected_slug)
            )

            st.write("**Validatie race_weather_ready**")
            st.write(
                canonical_validate_race_weather_ready_rows(df_test, df_overview, selected_year, selected_slug)
            )

            st.write("**Validatie prediction_ready**")
            st.write(
                canonical_validate_prediction_ready_rows(df_test, df_overview, selected_year, selected_slug)
            )


# ============================================================


# ============================================================
# Diagnostiek — Italy 2026 end-to-end dry smoke
# ============================================================
with tabs[6]:

    st.divider()

    st.header(
    "Italy 2026 synthetic workflow dry smoke"
)

st.caption(
    "Niet-destructieve synthetische end-to-end controle. "
    "De echte Italy 2026 RaceStatus wordt niet gewijzigd. "
    "Deze test gebruikt tijdelijke in-memory data en tijdelijke "
    "registry/ledger/log-bestanden."
)

if st.button(
    "Run synthetic Italy 2026 dry smoke",
    ):

        with st.spinner(
            "Italy 2026 workflow controleren..."
        ):

            smoke_result = (
                dry_smoke_test_italy_2026()
            )

        for check_name, check_ok in (
            smoke_result[
                "checks"
            ].items()
        ):

            st.write(
                "✅"
                if check_ok
                else "❌",
                check_name,
            )

        if smoke_result[
            "warnings"
        ]:

            st.warning(
                "Warnings:"
            )

            for warning in smoke_result[
                "warnings"
            ]:

                st.write(
                    "•",
                    warning,
                )

        if smoke_result[
            "ok"
        ]:

            st.success(
                "✅ Italy 2026 dry smoke volledig geslaagd."
            )

        else:

            st.error(
                "❌ Italy 2026 dry smoke faalt."
            )

            for error in smoke_result[
                "errors"
            ]:

                st.write(
                    "•",
                    error,
                )



# Tab 8 — Dashboard
# ============================================================
with tabs[7]:
    # STEP 9.2 UI — canonical post-race action
    st.subheader("Post-race action")

    _evaluation_is_predicted = (
        current_status == "predicted"
    )

    if current_status == "evaluated":
        st.success(
            "Deze race is reeds canonical geëvalueerd."
        )
    elif not _evaluation_is_predicted:
        st.info(
            "Evaluate race wordt actief zodra "
            "RaceStatus = predicted."
        )

    _actual_fastest_lap_sec = st.number_input(
        "Actual fastest lap (sec)",
        min_value=0.0,
        value=0.0,
        step=0.001,
        format="%.3f",
        key="actual_fastest_lap_sec_input",
        help=(
            "Voer de officiële race fastest-lap tijd in seconden in. "
            "evaluate_race berekent en bewaart alle KPI's canonical."
        ),
    )

    _actual_fastest_lap_valid = (
        _actual_fastest_lap_sec > 0.0
    )

    _evaluate_race_allowed = (
        _evaluation_is_predicted
        and _actual_fastest_lap_valid
    )

    _evaluate_race_button = st.button(
        "Evaluate race",
        key="canonical_evaluate_race_button",
        disabled=not _evaluate_race_allowed,
        help=(
            "Alleen beschikbaar bij RaceStatus=predicted "
            "en een positieve actual fastest-lap tijd."
        ),
    )

    if _evaluate_race_button:

        try:
            _evaluation_result = (
                canonical_evaluate_race(
                    year=selected_year,
                    gp_slug=selected_slug,
                    actual_fastest_lap_sec=float(
                        _actual_fastest_lap_sec
                    ),
                )
            )

        except Exception as exc:
            st.error(
                "Canonical race-evaluatie mislukt/geblokkeerd."
            )
            with st.expander(
                "Technische details",
                expanded=False,
            ):
                st.exception(exc)

        else:
            st.success(
                "Race canonical geëvalueerd. "
                "RaceStatus wordt refreshed."
            )

            st.session_state[
                "post_race_evaluation_result"
            ] = _evaluation_result

            st.rerun()

    # STEP 8.6 UI CALL — post-race evaluation dashboard
    render_post_race_evaluation_dashboard(
        df_overview=df_overview,
        year=selected_year,
        gp_slug=selected_slug,
        race_status=current_status,
    )

    render_dashboard()
