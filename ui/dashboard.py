"""Presentation helpers for the legacy model-performance dashboard."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def render_dashboard_diagnostic_tables(
    *,
    by_alt,
    by_sprint,
    by_track,
):
    st.subheader(
        "Diagnostische tabellen"
    )

    dt1, dt2, dt3 = st.tabs([
        "Track type",
        "Altitude bucket",
        "Sprint weekend",
    ])

    with dt1:
        st.dataframe(
            by_track.round(3),
            width="stretch",
            hide_index=True,
        )

    with dt2:
        st.dataframe(
            by_alt.round(3),
            width="stretch",
            hide_index=True,
        )

    with dt3:
        st.dataframe(
            by_sprint.round(3),
            width="stretch",
            hide_index=True,
        )

def render_dashboard_per_race_table(
    *,
    d_kpi,
    max_rows,
):
    st.subheader(
        "Per-race cijfers"
    )

    table_cols = [
        "Race #",
        "Year",
        "gp_slug",
        "Circuit",
        "Track_type",
        "Altitude_bucket",
        "IsSprintWeekend",
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
    ]

    table_cols = [
        col
        for col in table_cols
        if col in d_kpi.columns
    ]

    d_table = d_kpi[
        table_cols
    ].copy()

    d_table.insert(
        0,
        "Circuit_Year",
        d_kpi[
            "Circuit_Year"
        ],
    )

    if (
        "IsSprintWeekend"
        in d_table.columns
    ):
        d_table[
            "WeekendType"
        ] = np.where(
            pd.to_numeric(
                d_table[
                    "IsSprintWeekend"
                ],
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
            .eq(1),
            "Sprint weekend",
            "Non-sprint weekend",
        )

    float_cols = (
        d_table
        .select_dtypes(
            include=[
                "float64",
                "float32",
            ]
        )
        .columns
    )

    d_table[
        float_cols
    ] = d_table[
        float_cols
    ].round(3)

    st.dataframe(
        d_table.head(
            max_rows
        ),
        width="stretch",
        hide_index=True,
    )


    # ========================================================
    # MODEL PERFORMANCE BY RACE CONTEXT
    # ========================================================

def render_dashboard_context_mean_ape(
    *,
    context_stats,
    selected_context,
):
    st.markdown(
        "#### Mean APE% per model en segment"
    )

    chart_df = context_stats.copy()

    segments = (
        chart_df[
            "ContextSegment"
        ]
        .dropna()
        .unique()
        .tolist()
    )

    segments = sorted(
        segments
    )

    x_ctx = np.arange(
        len(segments)
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    model_offsets = {
        "GBR": -width,
        "XGB": 0,
        "CAT": width,
    }

    for model_name in [
        "GBR",
        "XGB",
        "CAT",
    ]:
        vals = []

        ns = []

        for segment in segments:

            row = chart_df[
                (
                    chart_df[
                        "ContextSegment"
                    ]
                    == segment
                )
                & (
                    chart_df[
                        "Model"
                    ]
                    == model_name
                )
            ]

            if row.empty:
                vals.append(
                    np.nan
                )
                ns.append(
                    0
                )
            else:
                vals.append(
                    float(
                        row[
                            "mean_APE"
                        ].iloc[0]
                    )
                )
                ns.append(
                    int(
                        row[
                            "n"
                        ].iloc[0]
                    )
                )

        bars = ax.bar(
            x_ctx
            + model_offsets[
                model_name
            ],
            vals,
            width=width,
            label=model_name,
        )

        for bar, n_val in zip(
            bars,
            ns,
        ):
            if n_val > 0:
                ax.text(
                    bar.get_x()
                    + bar.get_width()
                    / 2,
                    bar.get_height(),
                    f"n={n_val}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.axhline(
        y=2.0,
        color="black",
        linewidth=2.5,
        linestyle="-",
        label="Target 2%",
    )

    ax.set_title(
        f"Mean APE% by "
        f"{selected_context}"
    )

    ax.set_xlabel(
        selected_context
    )

    ax.set_ylabel(
        "Mean APE_pct"
    )

    ax.set_xticks(
        x_ctx
    )

    ax.set_xticklabels(
        segments,
        rotation=45,
        ha="right",
    )

    ax.legend()

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True,
    )

    # --------------------------------------------
    # Detailtabellen
    # --------------------------------------------

def render_dashboard_dynamic_advantage(
    *,
    step4_eval,
):
    st.markdown(
        "##### Dynamic advantage per race"
    )

    step4_eval[
        "RaceLabel"
    ] = (
        step4_eval[
            "Circuit"
        ].astype(str)
        + " ("
        + step4_eval[
            "Year"
        ]
        .astype(int)
        .astype(str)
        + ")"
    )

    x_delta = np.arange(
        len(step4_eval)
    )

    fig, ax = plt.subplots(
        figsize=(15, 5)
    )

    ax.bar(
        x_delta,
        step4_eval[
            "Dynamic_Improvement_AbsError_sec"
        ].values,
    )

    ax.axhline(
        y=0,
        color="black",
        linewidth=2.0,
    )

    ax.set_title(
        "Dynamic improvement in "
        "AbsError per race"
    )

    ax.set_ylabel(
        "GBR AbsError - "
        "Dynamic AbsError (sec)"
    )

    ax.set_xticks(
        x_delta
    )

    ax.set_xticklabels(
        step4_eval[
            "RaceLabel"
        ].tolist(),
        rotation=60,
        ha="right",
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True,
    )

    st.caption(
        "> 0 = dynamic beter; "
        "< 0 = GBR beter."
    )

    # ====================================
    # DETAIL TABLE
    # ====================================

def render_dashboard_step4_detail(
    *,
    step4_eval,
):
    with st.expander(
        "Toon Step 4 detaildata"
    ):

        detail_cols = [
            "Year",
            "Race #",
            "gp_slug",
            "Circuit",
            "ContextSegment",
            "History_n",
            "SelectedModel",
            "PolicyPassed",
            "Actual_sec",
            "GBRPrediction_sec",
            "DynamicPrediction_sec",
            "GBR_AbsError_sec",
            "Dynamic_AbsError_sec",
            "GBR_APE_pct_calc",
            "Dynamic_APE_pct_calc",
            "Dynamic_Improvement_AbsError_sec",
            "SelectionReason",
        ]

        detail_cols = [
            c
            for c in detail_cols
            if c in step4_eval.columns
        ]

        detail = step4_eval[
            detail_cols
        ].copy()

        float_cols = (
            detail
            .select_dtypes(
                include=[
                    "float32",
                    "float64",
                ]
            )
            .columns
        )

        detail[
            float_cols
        ] = detail[
            float_cols
        ].round(3)

        st.dataframe(
            detail,
            width="stretch",
            hide_index=True,
        )

def render_dashboard_kpi_comparison(
    *,
    dyn_kpi,
    gbr_kpi,
):
    st.markdown(
        "##### KPI comparison"
    )

    metric_keys = [
        "MAE_sec",
        "RMSE_sec",
        "Mean_APE_pct",
        "Median_APE_pct",
        "P90_APE_pct",
    ]

    labels_kpi = [
        "MAE sec",
        "RMSE sec",
        "Mean APE %",
        "Median APE %",
        "P90 APE %",
    ]

    x_kpi = np.arange(
        len(metric_keys)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.bar(
        x_kpi - width / 2,
        [
            gbr_kpi[k]
            for k in metric_keys
        ],
        width,
        label="Always GBR",
    )

    ax.bar(
        x_kpi + width / 2,
        [
            dyn_kpi[k]
            for k in metric_keys
        ],
        width,
        label="Dynamic selection",
    )

    ax.set_title(
        "Always GBR vs Dynamic selection"
    )

    ax.set_xticks(x_kpi)

    ax.set_xticklabels(
        labels_kpi,
        rotation=25,
        ha="right",
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True,
    )

def render_dashboard_model_selection_policy(
    *,
    summary,
):
    st.markdown(
        "#### Model selection policy — audit only"
    )

    policy_passed_count = int(
        summary[
            "PolicyPassed"
        ].fillna(False).sum()
    )

    policy_total_segments = int(
        len(summary)
    )

    pc1, pc2 = st.columns(2)

    pc1.metric(
        "Segments met alternatief model",
        policy_passed_count,
    )

    pc2.metric(
        "Totaal aantal segmenten",
        policy_total_segments,
    )

    suggested_counts = (
        summary[
            "SuggestedModel"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "SuggestedModel"
        )
        .reset_index(
            name="Segments"
        )
    )

    st.dataframe(
        suggested_counts,
        width="stretch",
        hide_index=True,
    )

    st.info(
        "SuggestedModel is uitsluitend een audit-uitkomst. "
        "PredictedFastestLap_sec blijft operationeel GBR."
    )

def render_dashboard_segment_summary(
    *,
    summary,
):
    st.markdown(
        "#### Segment summary"
    )

    display_cols = [
        c
        for c in [
            "ContextSegment",
            "n_races",
            "n_GBR",
            "n_XGB",
            "n_CAT",
            "GBR_mean_APE",
            "XGB_mean_APE",
            "CAT_mean_APE",
            "BestModel",
            "Improvement_vs_GBR_pp",
            "SuggestedModel",
            "SuggestedSegment_n",
            "Suggested_APE",
            "Suggested_vs_GBR_APE",
            "PolicyPassed",
            "SuggestedModelReason",
        ]
        if c in summary.columns
    ]

    st.dataframe(
        summary[
            display_cols
        ].sort_values(
            "ContextSegment"
        ),
        width="stretch",
        hide_index=True,
    )

def render_dashboard_detail_statistics(
    *,
    context_stats,
):
    with st.expander(
        "Toon detailstatistieken"
    ):

        detail_stats = (
            context_stats
            .copy()
        )

        for col in [
            "mean_APE",
            "median_APE",
            "p90_APE",
        ]:
            detail_stats[
                col
            ] = detail_stats[
                col
            ].round(3)

        st.dataframe(
            detail_stats,
            width="stretch",
            hide_index=True,
        )

    st.info(
        "Dit is alleen een contextanalyse. "
        "BestModel is nog geen operationele "
        "modelkeuze en verandert "
        "PredictedFastestLap_sec niet."
    )

def render_dashboard_dynamic_delta_summary(
    *,
    step4_eval,
):
    st.caption(
        "Positief = dynamic selection "
        "presteert beter dan Always GBR."
    )

    # ====================================
    # WINS / LOSSES
    # ====================================
    delta = pd.to_numeric(
        step4_eval[
            "Dynamic_Improvement_AbsError_sec"
        ],
        errors="coerce",
    )

    wins = int(
        (delta > 0.001).sum()
    )

    losses = int(
        (delta < -0.001).sum()
    )

    ties = int(
        (
            delta.abs()
            <= 0.001
        ).sum()
    )

    win_rate = (
        100.0
        * wins
        / len(step4_eval)
    )

    w1, w2, w3, w4 = (
        st.columns(4)
    )

    w1.metric(
        "Dynamic wins",
        wins,
    )

    w2.metric(
        "Dynamic losses",
        losses,
    )

    w3.metric(
        "Ties",
        ties,
    )

    w4.metric(
        "Win rate",
        f"{win_rate:.1f}%",
    )

def render_dashboard_main_charts(
    *,
    d_kpi,
    rolling_window,
):
    # CHART 1 — APE per race
    # ========================================================
    x = np.arange(
        len(d_kpi)
    )

    labels = (
        d_kpi["Circuit_Year"]
        .tolist()
    )

    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.plot(
        x,
        d_kpi["APE_pct"].values,
        marker="o",
        linestyle="-",
        label="APE%",
    )

    ax.plot(
        x,
        d_kpi[
            "APE_roll_median"
        ].values,
        linestyle="--",
        label=(
            f"Rolling median "
            f"(w={rolling_window})"
        ),
    )

    # Targetlijn APE = 2%
    ax.axhline(
        y=2.0,
        color="black",
        linewidth=2.5,
        linestyle="-",
        label="Target 2%",
    )

    ax.set_title(
        "APE% per race"
    )

    ax.set_xlabel(
        "Circuit (Year)"
    )

    ax.set_ylabel(
        "APE_pct"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.legend()

    fig.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True,
    )

    # ========================================================
    # CHART 2 — AbsError
    # ========================================================
    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.plot(
        x,
        d_kpi["AbsError"].values,
        marker="o",
        linestyle="-",
    )

    ax.set_title(
        "AbsError (sec)"
    )

    ax.set_xlabel(
        "Circuit (Year)"
    )

    ax.set_ylabel(
        "AbsError"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True,
    )

    # ========================================================
    # CHART 3 — Rolling P90
    # ========================================================
    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.plot(
        x,
        d_kpi[
            "APE_roll_p90"
        ].values,
        marker="o",
        linestyle="-",
    )

    ax.set_title(
        f"Rolling P90 APE% "
        f"(window={rolling_window})"
    )

    ax.set_xlabel(
        "Circuit (Year)"
    )

    ax.set_ylabel(
        "APE_pct (P90)"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True,
    )

    # ========================================================
    # CHART 4 — Signed Error
    # ========================================================
    fig, ax = plt.subplots(
        figsize=(14, 5)
    )

    ax.plot(
        x,
        d_kpi["Error"].values,
        marker="o",
        linestyle="-",
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_title(
        "Signed Error (sec)"
    )

    ax.set_xlabel(
        "Circuit (Year)"
    )

    ax.set_ylabel(
        "Error = Pred - Actual"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    fig.tight_layout()

    st.pyplot(
        fig,
        clear_figure=True,
    )

    # ========================================================


def render_dashboard_context_bar_charts(
    *,
    by_track,
    by_alt,
    by_sprint,
):
    # Helper voor bar charts
    # ========================================================
    def dashboard_bar_chart(
        table,
        category_col,
        title,
        xlabel,
    ):
        if table.empty:
            st.info(
                f"Geen data voor {title}"
            )
            return

        fig, ax = plt.subplots(
            figsize=(9, 4.5)
        )

        xb = np.arange(
            len(table)
        )

        ax.bar(
            xb,
            table["mean_APE"].values,
        )

        ax.set_title(title)
        ax.set_xlabel(xlabel)

        ax.set_ylabel(
            "mean APE_pct"
        )

        ax.set_xticks(xb)

        ax.set_xticklabels(
            table[
                category_col
            ]
            .astype(str)
            .tolist(),
            rotation=45,
            ha="right",
        )

        ax.grid(
            True,
            axis="y",
            alpha=0.3,
        )

        for i, row in (
            table
            .reset_index(drop=True)
            .iterrows()
        ):
            ax.text(
                i,
                row["mean_APE"],
                f"n={int(row['n'])}",
                ha="center",
                va="bottom",
            )

        fig.tight_layout()

        st.pyplot(
            fig,
            clear_figure=True,
        )

    # ========================================================
    # CHART 5 + 6
    # ========================================================
    col_a, col_b = st.columns(2)

    with col_a:
        dashboard_bar_chart(
            by_track,
            "Track_type",
            "Mean APE% by Track_type",
            "Track_type",
        )

    with col_b:
        dashboard_bar_chart(
            by_alt,
            "Altitude_bucket",
            "Mean APE% by Altitude_bucket",
            "Altitude_bucket",
        )

    # ========================================================
    # CHART 7 — Sprint
    # ========================================================
    dashboard_bar_chart(
        by_sprint,
        "WeekendType",
        "Mean APE% by Sprint weekend",
        "Weekend type",
    )


    # ========================================================


def render_dashboard_model_comparison(
    *,
    df,
):
    # MODELVERGELIJKING — GBR / XGB / CAT APE%
    # ========================================================
    st.subheader(
        "APE% vergelijking GBR vs XGB vs CAT"
    )

    required_model_ape_cols = [
        "APE_pct_GBR",
        "APE_pct_XGB",
        "APE_pct_CAT",
    ]

    # --------------------------------------------------------
    # Eigen season-filter voor GBR / XGB / CAT grafiek
    # Onafhankelijk van de algemene dashboardfilter
    # --------------------------------------------------------
    model_source_df = df.copy()

    model_years = sorted(
        pd.to_numeric(
            model_source_df["Year"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    model_season = st.selectbox(
        "Season — GBR / XGB / CAT vergelijking",
        options=["All"] + model_years,
        index=0,
        key="dashboard_model_ape_season",
    )

    if model_season != "All":
        model_source_df = model_source_df[
            pd.to_numeric(
                model_source_df["Year"],
                errors="coerce",
            ).eq(int(model_season))
        ].copy()

    missing_model_ape_cols = [
        c
        for c in required_model_ape_cols
        if c not in model_source_df.columns
    ]

    if missing_model_ape_cols:
        st.warning(
            "Model-APE grafiek niet beschikbaar. "
            f"Ontbrekende kolommen: {missing_model_ape_cols}"
        )

    else:
        model_ape_df = model_source_df[
            model_source_df[
                required_model_ape_cols
            ]
            .notna()
            .any(axis=1)
        ].copy()

        if model_ape_df.empty:
            st.info(
                "Geen rijen gevonden met "
                "APE_pct_GBR / APE_pct_XGB / APE_pct_CAT."
            )

        else:
            # -----------------------------------------------
            # Numeric conversie
            # -----------------------------------------------
            for col in required_model_ape_cols:
                model_ape_df[col] = pd.to_numeric(
                    model_ape_df[col],
                    errors="coerce",
                )

            # -----------------------------------------------
            # Consistente sortering
            # -----------------------------------------------
            sort_cols = [
                c
                for c in ["Year", "Race #"]
                if c in model_ape_df.columns
            ]

            if sort_cols:
                model_ape_df = (
                    model_ape_df
                    .sort_values(sort_cols)
                    .reset_index(drop=True)
                )

            # -----------------------------------------------
            # X-as label
            # -----------------------------------------------
            if (
                "Circuit" in model_ape_df.columns
                and "Year" in model_ape_df.columns
            ):
                model_ape_df["Race_Year_Label"] = (
                    model_ape_df["Circuit"].astype(str)
                    + " ("
                    + model_ape_df["Year"]
                    .astype(int)
                    .astype(str)
                    + ")"
                )

            elif (
                "gp_slug" in model_ape_df.columns
                and "Year" in model_ape_df.columns
            ):
                model_ape_df["Race_Year_Label"] = (
                    model_ape_df["gp_slug"].astype(str)
                    + " ("
                    + model_ape_df["Year"]
                    .astype(int)
                    .astype(str)
                    + ")"
                )

            else:
                model_ape_df["Race_Year_Label"] = (
                    model_ape_df.index.astype(str)
                )

            # -----------------------------------------------
            # Plot
            # -----------------------------------------------
            x_model = np.arange(
                len(model_ape_df)
            )

            labels_model = (
                model_ape_df[
                    "Race_Year_Label"
                ]
                .tolist()
            )

            fig, ax = plt.subplots(
                figsize=(16, 7)
            )

            ax.plot(
                x_model,
                model_ape_df[
                    "APE_pct_GBR"
                ].values,
                marker="o",
                linestyle="-",
                label="GBR",
            )

            ax.plot(
                x_model,
                model_ape_df[
                    "APE_pct_XGB"
                ].values,
                marker="o",
                linestyle="-",
                label="XGB",
            )

            ax.plot(
                x_model,
                model_ape_df[
                    "APE_pct_CAT"
                ].values,
                marker="o",
                linestyle="-",
                label="CAT",
            )

            # Targetlijn op 2%
            ax.axhline(
                y=2.0,
                color="black",
                linewidth=2.5,
                linestyle="-",
                label="Target 2%",
            )

            ax.set_title(
                "APE_pct vergelijking per race"
            )

            ax.set_xlabel(
                "Race (Circuit + Year)"
            )

            ax.set_ylabel(
                "APE_pct"
            )

            ax.set_xticks(
                x_model
            )

            ax.set_xticklabels(
                labels_model,
                rotation=60,
                ha="right",
            )

            ax.legend()

            ax.grid(
                True,
                alpha=0.3,
            )

            fig.tight_layout()

            st.pyplot(
                fig,
                clear_figure=True,
            )

            # -----------------------------------------------
            # Onderliggende data
            # -----------------------------------------------
            st.caption(
                "Onderliggende data voor de modelvergelijking"
            )

            model_print_cols = [
                c
                for c in [
                    "Race #",
                    "Year",
                    "gp_slug",
                    "Circuit",
                    "Race_Year_Label",
                    "APE_pct_GBR",
                    "APE_pct_XGB",
                    "APE_pct_CAT",
                ]
                if c in model_ape_df.columns
            ]

            model_print_df = (
                model_ape_df[
                    model_print_cols
                ]
                .copy()
            )

            for col in [
                "APE_pct_GBR",
                "APE_pct_XGB",
                "APE_pct_CAT",
            ]:
                if col in model_print_df.columns:
                    model_print_df[col] = (
                        pd.to_numeric(
                            model_print_df[col],
                            errors="coerce",
                        )
                        .round(3)
                    )

            st.dataframe(
                model_print_df,
                width="stretch",
                hide_index=True,
            )



    # ========================================================

