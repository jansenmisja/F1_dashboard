"""Streamlit presentation helpers for prediction output."""

import pandas as pd
import streamlit as st

def _prediction_ui_value(row, column, default=None):
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


def _prediction_ui_float(row, column, default=None):
    value = _prediction_ui_value(
        row,
        column,
        default,
    )

    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def _prediction_ui_fmt_sec(value):
    if value is None:
        return "—"

    try:
        return f"{float(value):.3f} s"
    except Exception:
        return "—"


def _prediction_ui_fmt_value(value):
    if value is None:
        return "—"

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return "—"

    return text


def _prediction_ui_target_row(
    df,
    *,
    year,
    gp_slug,
):
    """
    Read-only exact target lookup.
    """

    if df is None or df.empty:
        return None

    if not {"Year", "gp_slug"}.issubset(df.columns):
        return None

    year_values = pd.to_numeric(
        df["Year"],
        errors="coerce",
    )

    slug_values = (
        df["gp_slug"]
        .astype(str)
        .str.strip()
    )

    mask = (
        year_values.eq(int(year))
        & slug_values.eq(str(gp_slug))
    )

    target = df.loc[mask]

    if len(target) != 1:
        return None

    return target.iloc[0]


def _prediction_ui_first_existing(
    row,
    columns,
):
    if row is None:
        return None

    for column in columns:
        value = _prediction_ui_value(
            row,
            column,
            None,
        )

        if value is not None:
            return value

    return None


def _prediction_ui_latest_log_row(
    model_log_df,
    *,
    year,
    gp_slug,
):
    """
    Read-only laatste matching append-only logregel.
    """

    if (
        model_log_df is None
        or model_log_df.empty
    ):
        return None

    year_col = next(
        (
            c for c in ("Year", "year")
            if c in model_log_df.columns
        ),
        None,
    )

    slug_col = next(
        (
            c for c in (
                "gp_slug",
                "GP_slug",
                "gp",
            )
            if c in model_log_df.columns
        ),
        None,
    )

    if year_col is None or slug_col is None:
        return None

    year_values = pd.to_numeric(
        model_log_df[year_col],
        errors="coerce",
    )

    slug_values = (
        model_log_df[slug_col]
        .astype(str)
        .str.strip()
    )

    target = model_log_df.loc[
        year_values.eq(int(year))
        & slug_values.eq(str(gp_slug))
    ]

    if target.empty:
        return None

    return target.iloc[-1]


def render_prediction_dashboard(
    *,
    df_test,
    df_overview,
    model_log_df,
    year,
    gp_slug,
    race_status,
):
    """
    Pure prediction presentation.

    Geen:
    - model fit/predict;
    - modelselectie;
    - lifecycle mutation;
    - readiness-validator;
    - persistence.
    """

    overview_row = _prediction_ui_target_row(
        df_overview,
        year=year,
        gp_slug=gp_slug,
    )

    test_row = _prediction_ui_target_row(
        df_test,
        year=year,
        gp_slug=gp_slug,
    )

    log_row = _prediction_ui_latest_log_row(
        model_log_df,
        year=year,
        gp_slug=gp_slug,
    )

    st.subheader("Prediction dashboard")

    if overview_row is None:
        st.info(
            "Nog geen unieke df_overview-rij beschikbaar "
            "voor deze race."
        )
        return

    # --------------------------------------------------------
    # Core values
    # --------------------------------------------------------
    pred_gbr = _prediction_ui_float(
        overview_row,
        "Pred_GBR",
    )

    pred_xgb = _prediction_ui_float(
        overview_row,
        "Pred_XGB",
    )

    pred_cat = _prediction_ui_float(
        overview_row,
        "Pred_CAT",
    )

    operational = _prediction_ui_float(
        overview_row,
        "PredictedFastestLap_sec",
    )

    record_lap = _prediction_ui_float(
        overview_row,
        "RecordLap_sec",
    )

    if record_lap is None:
        record_lap = _prediction_ui_float(
            test_row,
            "RecordLap_sec",
        )

    gbr_offset = _prediction_ui_float(
        overview_row,
        "Pred_GBR_offset_sec",
    )

    xgb_offset = _prediction_ui_float(
        overview_row,
        "Pred_XGB_offset_sec",
    )

    cat_offset = _prediction_ui_float(
        overview_row,
        "Pred_CAT_offset_sec",
    )

    operational_offset = _prediction_ui_float(
        overview_row,
        "pred_offset_sec",
    )

    # --------------------------------------------------------
    # Operational cards
    # --------------------------------------------------------
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Operationele prediction · GBR",
            _prediction_ui_fmt_sec(
                operational
            ),
        )

    with c2:
        st.metric(
            "RecordLap",
            _prediction_ui_fmt_sec(
                record_lap
            ),
        )

    with c3:
        st.metric(
            "Operationele offset",
            _prediction_ui_fmt_sec(
                operational_offset
            ),
        )

    if (
        pred_gbr is not None
        and operational is not None
    ):
        if abs(pred_gbr - operational) < 0.0005:
            st.success(
                "Operationele policy: "
                "PredictedFastestLap_sec = Pred_GBR."
            )
        else:
            st.warning(
                "PredictedFastestLap_sec wijkt af van Pred_GBR. "
                "De UI past niets aan; controleer de canonical "
                "prediction/audit-output."
            )

    # --------------------------------------------------------
    # Three model comparison
    # --------------------------------------------------------
    st.markdown("##### GBR / XGB / CAT")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "GBR · operationeel",
            _prediction_ui_fmt_sec(pred_gbr),
            delta=(
                _prediction_ui_fmt_sec(
                    gbr_offset
                )
                if gbr_offset is not None
                else None
            ),
        )

    with m2:
        st.metric(
            "XGB · challenger",
            _prediction_ui_fmt_sec(pred_xgb),
            delta=(
                _prediction_ui_fmt_sec(
                    xgb_offset
                )
                if xgb_offset is not None
                else None
            ),
        )

    with m3:
        st.metric(
            "CAT · challenger",
            _prediction_ui_fmt_sec(pred_cat),
            delta=(
                _prediction_ui_fmt_sec(
                    cat_offset
                )
                if cat_offset is not None
                else None
            ),
        )

    # --------------------------------------------------------
    # Backend / fallback
    # --------------------------------------------------------
    st.markdown("##### Backend / fallback")

    cat_backend = _prediction_ui_first_existing(
        log_row,
        (
            "CAT_Backend",
            "CatBackend",
            "Pred_CAT_Backend",
            "Backend_CAT",
            "ModelBackend_CAT",
            "cat_backend",
        ),
    )

    fallback_status = _prediction_ui_first_existing(
        log_row,
        (
            "FallbackStatus",
            "CAT_Fallback",
            "CatFallback",
            "FallbackBackend",
            "fallback_backend",
        ),
    )

    fallback_reason = _prediction_ui_first_existing(
        log_row,
        (
            "FallbackReason",
            "fallback_reason",
            "Opmerkingen",
            "Commentaar",
            "Beschrijving",
            "Description",
        ),
    )

    b1, b2, b3 = st.columns(3)

    b1.metric("GBR backend", "GBR")
    b2.metric("XGB backend", "XGB")

    b3.metric(
        "CAT backend",
        _prediction_ui_fmt_value(
            cat_backend
        ),
    )

    if fallback_status is not None:
        st.info(
            "CAT fallback-status: "
            + _prediction_ui_fmt_value(
                fallback_status
            )
        )

    if fallback_reason is not None:
        with st.expander(
            "Backend / fallback auditdetail"
        ):
            st.write(
                _prediction_ui_fmt_value(
                    fallback_reason
                )
            )

    if (
        cat_backend is None
        and fallback_status is None
    ):
        st.caption(
            "Geen expliciete CAT-backendinformatie "
            "gevonden in de laatste matching model_log-regel. "
            "De UI leidt de backend bewust niet af."
        )

    # --------------------------------------------------------
    # Timing / snapshot / registry
    # --------------------------------------------------------
    st.markdown("##### Timing & snapshot")

    def event_value(column):
        value = _prediction_ui_value(
            overview_row,
            column,
            None,
        )

        if value is None:
            value = _prediction_ui_value(
                test_row,
                column,
                None,
            )

        return value

    timing_table = pd.DataFrame(
        [
            {
                "Item": "RaceStatus",
                "Value": (
                    race_status or "—"
                ),
            },
            {
                "Item": "EventDateTimeUTC",
                "Value": _prediction_ui_fmt_value(
                    event_value(
                        "EventDateTimeUTC"
                    )
                ),
            },
            {
                "Item": "PredictionCutoffUTC",
                "Value": _prediction_ui_fmt_value(
                    event_value(
                        "PredictionCutoffUTC"
                    )
                ),
            },
            {
                "Item": "PredictionTimestampUTC",
                "Value": _prediction_ui_fmt_value(
                    event_value(
                        "PredictionTimestampUTC"
                    )
                ),
            },
            {
                "Item": "FeatureSnapshotAfterQualifyingUTC",
                "Value": _prediction_ui_fmt_value(
                    event_value(
                        "FeatureSnapshotAfterQualifyingUTC"
                    )
                ),
            },
            {
                "Item": "FeatureSnapshotTimestampUTC",
                "Value": _prediction_ui_fmt_value(
                    event_value(
                        "FeatureSnapshotTimestampUTC"
                    )
                ),
            },
            {
                "Item": "RegistryEligible",
                "Value": _prediction_ui_fmt_value(
                    event_value(
                        "RegistryEligible"
                    )
                ),
            },
        ]
    )

    st.dataframe(
        timing_table,
        hide_index=True,
        width="stretch",
    )

    # --------------------------------------------------------
    # Explicit evidence status only
    # --------------------------------------------------------
    evidence_status = _prediction_ui_first_existing(
        overview_row,
        (
            "PromotionGradeStatus",
            "PromotionGrade",
            "EvidenceClass",
            "ForwardEvidenceStatus",
        ),
    )

    if evidence_status is None:
        evidence_status = (
            _prediction_ui_first_existing(
                log_row,
                (
                    "PromotionGradeStatus",
                    "PromotionGrade",
                    "EvidenceClass",
                    "ForwardEvidenceStatus",
                ),
            )
        )

    if evidence_status is None:
        st.caption(
            "Geen expliciete promotion-grade/evidence-status "
            "gevonden. De UI berekent deze status bewust niet zelf."
        )
    else:
        st.info(
            "**Opgeslagen evidence-status:** "
            + _prediction_ui_fmt_value(
                evidence_status
            )
        )

    # --------------------------------------------------------
    # Details
    # --------------------------------------------------------
    with st.expander("Prediction detailwaarden"):

        detail = pd.DataFrame(
            [
                {
                    "Model": "GBR",
                    "Prediction_sec": pred_gbr,
                    "RecordOffset_sec": gbr_offset,
                    "Operational": True,
                },
                {
                    "Model": "XGB",
                    "Prediction_sec": pred_xgb,
                    "RecordOffset_sec": xgb_offset,
                    "Operational": False,
                },
                {
                    "Model": "CAT",
                    "Prediction_sec": pred_cat,
                    "RecordOffset_sec": cat_offset,
                    "Operational": False,
                },
            ]
        )

        st.dataframe(
            detail,
            hide_index=True,
            width="stretch",
        )

    st.divider()
