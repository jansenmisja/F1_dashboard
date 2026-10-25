"""Streamlit presentation helpers for post-race evaluation."""

import pandas as pd
import streamlit as st

def _evaluation_ui_value(
    row,
    column,
    default=None,
):
    """
    Read-only accessor.
    """

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


def _evaluation_ui_float(
    row,
    column,
):
    value = _evaluation_ui_value(
        row,
        column,
        None,
    )

    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def _evaluation_ui_fmt_sec(value):
    if value is None:
        return "—"

    try:
        return f"{float(value):.3f} s"
    except Exception:
        return "—"


def _evaluation_ui_fmt_pct(value):
    if value is None:
        return "—"

    try:
        return f"{float(value):.3f}%"
    except Exception:
        return "—"


def _evaluation_ui_fmt_number(value):
    if value is None:
        return "—"

    try:
        return f"{float(value):.3f}"
    except Exception:
        return "—"


def _evaluation_ui_target_row(
    df,
    *,
    year,
    gp_slug,
):
    """
    Exact read-only lookup.

    Geen statusinferentie.
    Geen lifecyclemutatie.
    """

    if df is None or df.empty:
        return None

    if not {"Year", "gp_slug"}.issubset(
        df.columns
    ):
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


def render_post_race_evaluation_dashboard(
    *,
    df_overview,
    year,
    gp_slug,
    race_status,
):
    """
    Pure post-race KPI presentation.

    Belangrijk:
    - KPI's worden NIET berekend;
    - alleen canonical opgeslagen df_overview-waarden;
    - evaluate_race blijft authoritative;
    - operationele GBR-policy wordt niet gewijzigd.
    """

    row = _evaluation_ui_target_row(
        df_overview,
        year=year,
        gp_slug=gp_slug,
    )

    st.header("Post-race evaluation")

    if row is None:
        st.info(
            "Geen unieke df_overview-rij beschikbaar "
            "voor deze race."
        )
        return

    actual = _evaluation_ui_float(
        row,
        "ActualFastestLap_sec",
    )

    operational = _evaluation_ui_float(
        row,
        "PredictedFastestLap_sec",
    )

    error = _evaluation_ui_float(
        row,
        "Error",
    )

    abs_error = _evaluation_ui_float(
        row,
        "AbsError",
    )

    ape = _evaluation_ui_float(
        row,
        "APE_pct",
    )

    # --------------------------------------------------------
    # Not evaluated yet
    # --------------------------------------------------------
    if actual is None:
        st.info(
            "ActualFastestLap_sec is nog niet beschikbaar. "
            "Post-race KPI's worden pas na de canonical "
            "evaluate_race-flow als geëvalueerd beschouwd."
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Operationele prediction",
            _evaluation_ui_fmt_sec(
                operational
            ),
        )

        c2.metric(
            "Actual fastest lap",
            "—",
        )

        st.caption(
            f"RaceStatus: {race_status or '—'}"
        )

        st.divider()
        return

    # --------------------------------------------------------
    # Primary race result
    # --------------------------------------------------------
    st.markdown("##### Race resultaat")

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Prediction",
        _evaluation_ui_fmt_sec(
            operational
        ),
    )

    c2.metric(
        "Actual",
        _evaluation_ui_fmt_sec(
            actual
        ),
    )

    c3.metric(
        "Error",
        _evaluation_ui_fmt_sec(
            error
        ),
        help=(
            "Prediction - Actual. "
            "Negatief = prediction was te snel."
        ),
    )

    c4.metric(
        "AbsError",
        _evaluation_ui_fmt_sec(
            abs_error
        ),
    )

    c5.metric(
        "APE",
        _evaluation_ui_fmt_pct(
            ape
        ),
    )

    # --------------------------------------------------------
    # Model-specific KPIs
    # --------------------------------------------------------
    st.markdown("##### Modelvergelijking")

    model_rows = []

    for model in (
        "GBR",
        "XGB",
        "CAT",
    ):
        model_rows.append(
            {
                "Model": model,
                "Prediction_sec":
                    _evaluation_ui_float(
                        row,
                        f"Pred_{model}",
                    ),
                "Error_sec":
                    _evaluation_ui_float(
                        row,
                        f"Error_{model}",
                    ),
                "AbsError_sec":
                    _evaluation_ui_float(
                        row,
                        f"AbsError_{model}",
                    ),
                "APE_pct":
                    _evaluation_ui_float(
                        row,
                        f"APE_pct_{model}",
                    ),
                "RecordOffset_sec":
                    _evaluation_ui_float(
                        row,
                        f"Pred_{model}_offset_sec",
                    ),
                "Operational":
                    model == "GBR",
            }
        )

    model_df = pd.DataFrame(
        model_rows
    )

    st.dataframe(
        model_df,
        hide_index=True,
        width="stretch",
    )

    st.caption(
        "GBR blijft operationeel. "
        "Deze vergelijking wijzigt de modelpolicy niet."
    )

    # --------------------------------------------------------
    # APE cards per model
    # --------------------------------------------------------
    a1, a2, a3 = st.columns(3)

    a1.metric(
        "GBR APE",
        _evaluation_ui_fmt_pct(
            _evaluation_ui_float(
                row,
                "APE_pct_GBR",
            )
        ),
    )

    a2.metric(
        "XGB APE",
        _evaluation_ui_fmt_pct(
            _evaluation_ui_float(
                row,
                "APE_pct_XGB",
            )
        ),
    )

    a3.metric(
        "CAT APE",
        _evaluation_ui_fmt_pct(
            _evaluation_ui_float(
                row,
                "APE_pct_CAT",
            )
        ),
    )

    # --------------------------------------------------------
    # Offset evaluation
    # --------------------------------------------------------
    st.markdown("##### RecordLap offset-evaluatie")

    o1, o2, o3, o4 = st.columns(4)

    o1.metric(
        "Pred offset",
        _evaluation_ui_fmt_sec(
            _evaluation_ui_float(
                row,
                "pred_offset_sec",
            )
        ),
    )

    o2.metric(
        "Actual offset",
        _evaluation_ui_fmt_sec(
            _evaluation_ui_float(
                row,
                "target_offset_sec",
            )
        ),
    )

    o3.metric(
        "Offset error",
        _evaluation_ui_fmt_sec(
            _evaluation_ui_float(
                row,
                "Error_offset",
            )
        ),
    )

    o4.metric(
        "Offset APE",
        _evaluation_ui_fmt_pct(
            _evaluation_ui_float(
                row,
                "APE_pct_offset",
            )
        ),
    )

    abs_error_offset = (
        _evaluation_ui_float(
            row,
            "AbsError_offset",
        )
    )

    if abs_error_offset is not None:
        st.caption(
            "AbsError_offset: "
            + _evaluation_ui_fmt_sec(
                abs_error_offset
            )
        )

    # --------------------------------------------------------
    # Cumulative KPIs
    # --------------------------------------------------------
    st.markdown("##### Cumulative metrics")

    cumulative_values = {
        "Cumulative MAE":
            _evaluation_ui_float(
                row,
                "Cumulative_MAE",
            ),
        "Cumulative RMSE":
            _evaluation_ui_float(
                row,
                "Cumulative_RMSE",
            ),
        "Cumulative R²":
            _evaluation_ui_float(
                row,
                "Cumulative_R2",
            ),
    }

    cc1, cc2, cc3 = st.columns(3)

    cc1.metric(
        "Cumulative MAE",
        _evaluation_ui_fmt_sec(
            cumulative_values[
                "Cumulative MAE"
            ]
        ),
    )

    cc2.metric(
        "Cumulative RMSE",
        _evaluation_ui_fmt_sec(
            cumulative_values[
                "Cumulative RMSE"
            ]
        ),
    )

    cc3.metric(
        "Cumulative R²",
        _evaluation_ui_fmt_number(
            cumulative_values[
                "Cumulative R²"
            ]
        ),
    )

    # --------------------------------------------------------
    # Canonical completeness display
    #
    # Alleen PRESENTATIE van ontbrekende opgeslagen KPI's.
    # Geen berekening / validator / mutation.
    # --------------------------------------------------------
    expected_evaluation_fields = (
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
        "APE_pct_offset",
    )

    missing_stored_fields = [
        column
        for column in expected_evaluation_fields
        if _evaluation_ui_value(
            row,
            column,
            None,
        ) is None
    ]

    if missing_stored_fields:
        st.warning(
            "Actual is aanwezig, maar niet alle verwachte "
            "canonical evaluatievelden zijn gevuld."
        )

        with st.expander(
            "Ontbrekende opgeslagen evaluatievelden"
        ):
            st.write(
                missing_stored_fields
            )

            st.caption(
                "Gebruik de canonical evaluate_race-flow. "
                "De Streamlit UI berekent deze velden niet zelf."
            )
    else:
        st.success(
            "Race-level canonical evaluatievelden zijn gevuld."
        )

    # --------------------------------------------------------
    # Compact raw detail
    # --------------------------------------------------------
    with st.expander(
        "Race-level KPI detail"
    ):
        detail_columns = [
            "Year",
            "gp_slug",
            "Circuit",
            "RaceStatus",
            "RecordLap_sec",
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
        ]

        available = [
            c
            for c in detail_columns
            if c in row.index
        ]

        if available:
            detail_df = pd.DataFrame(
                [
                    {
                        c: row[c]
                        for c in available
                    }
                ]
            )

            st.dataframe(
                detail_df,
                hide_index=True,
                width="stretch",
            )

    st.divider()
