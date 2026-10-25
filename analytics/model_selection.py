"""Pure model-selection analytics for dashboard and future-event analysis."""

import numpy as np
import pandas as pd

MIN_N_SEGMENT = 6

MIN_IMPROVEMENT_APE = 0.15

def select_suggested_model(
    segment_summary_row,
    min_n_segment=MIN_N_SEGMENT,
    min_improvement_ape=MIN_IMPROVEMENT_APE,
):
    """
    Audit-only modelselectie.

    Regels:
    1. GBR blijft fallback.
    2. Segment moet voldoende observaties hebben.
    3. XGB/CAT mag alleen worden voorgesteld als mean APE
       minimaal min_improvement_ape procentpunt beter is dan GBR.

    Returns:
        dict met SuggestedModel, reason, n, best APE,
        GBR APE en improvement versus GBR.
    """

    result = {
        "SuggestedModel": "GBR",
        "SuggestedModelReason": "",
        "SuggestedSegment_n": 0,
        "Suggested_APE": np.nan,
        "GBR_APE": np.nan,
        "Suggested_vs_GBR_APE": 0.0,
        "PolicyPassed": False,
    }

    # --------------------------------------------------------
    # Segment n bepalen
    # --------------------------------------------------------
    n_value = pd.to_numeric(
        segment_summary_row.get("n_races"),
        errors="coerce",
    )

    if pd.isna(n_value):
        n_value = 0

    n_value = int(n_value)

    result["SuggestedSegment_n"] = n_value

    # --------------------------------------------------------
    # Model APE's
    # --------------------------------------------------------
    model_apes = {
        "GBR": pd.to_numeric(
            segment_summary_row.get("GBR_mean_APE"),
            errors="coerce",
        ),
        "XGB": pd.to_numeric(
            segment_summary_row.get("XGB_mean_APE"),
            errors="coerce",
        ),
        "CAT": pd.to_numeric(
            segment_summary_row.get("CAT_mean_APE"),
            errors="coerce",
        ),
    }

    gbr_ape = model_apes["GBR"]

    result["GBR_APE"] = (
        float(gbr_ape)
        if pd.notna(gbr_ape)
        else np.nan
    )

    # Zonder GBR-baseline geen geldige vergelijking
    if pd.isna(gbr_ape):
        result["SuggestedModelReason"] = (
            "GBR baseline APE ontbreekt; fallback naar GBR."
        )
        return result

    # --------------------------------------------------------
    # Te weinig races
    # --------------------------------------------------------
    if n_value < min_n_segment:
        result["Suggested_APE"] = float(gbr_ape)
        result["SuggestedModelReason"] = (
            f"Segment n={n_value} < minimum {min_n_segment}; "
            "GBR fallback."
        )
        return result

    # --------------------------------------------------------
    # Beste beschikbare model zoeken
    # --------------------------------------------------------
    valid_model_apes = {
        model: ape
        for model, ape in model_apes.items()
        if pd.notna(ape)
    }

    if not valid_model_apes:
        result["SuggestedModelReason"] = (
            "Geen geldige model-APE waarden; GBR fallback."
        )
        return result

    best_model = min(
        valid_model_apes,
        key=valid_model_apes.get,
    )

    best_ape = float(
        valid_model_apes[best_model]
    )

    improvement = (
        float(gbr_ape)
        - best_ape
    )

    result["Suggested_APE"] = best_ape
    result["Suggested_vs_GBR_APE"] = improvement

    # --------------------------------------------------------
    # GBR is zelf beste
    # --------------------------------------------------------
    if best_model == "GBR":
        result["SuggestedModelReason"] = (
            f"GBR heeft laagste mean APE "
            f"({float(gbr_ape):.3f}%)."
        )
        return result

    # --------------------------------------------------------
    # Verbetering te klein
    # --------------------------------------------------------
    if improvement < min_improvement_ape:
        result["SuggestedModelReason"] = (
            f"{best_model} is slechts "
            f"{improvement:.3f} pp beter dan GBR; "
            f"minimum vereist is "
            f"{min_improvement_ape:.3f} pp. "
            "GBR fallback."
        )
        return result

    # --------------------------------------------------------
    # Policy geslaagd
    # --------------------------------------------------------
    result["SuggestedModel"] = best_model
    result["PolicyPassed"] = True

    result["SuggestedModelReason"] = (
        f"{best_model} geselecteerd: "
        f"segment n={n_value}; "
        f"{best_model} mean APE={best_ape:.3f}% vs "
        f"GBR={float(gbr_ape):.3f}%; "
        f"verbetering={improvement:.3f} pp."
    )

    return result


def _context_segment_value(
    row: pd.Series,
    context_col: str,
):
    """
    Zet de contextwaarde van één race om naar dezelfde
    segmentdefinitie als in het dashboard.
    """

    if context_col == "IsSprintWeekend":
        value = pd.to_numeric(
            row.get(context_col),
            errors="coerce",
        )

        if pd.isna(value):
            return np.nan

        return (
            "Sprint weekend"
            if int(value) == 1
            else "Non-sprint weekend"
        )

    value = row.get(
        context_col,
        np.nan,
    )

    if pd.isna(value):
        return np.nan

    return str(value)


def walk_forward_model_selection(
    source_df: pd.DataFrame,
    context_col: str,
    min_n_segment: int = MIN_N_SEGMENT,
    min_improvement_ape: float = MIN_IMPROVEMENT_APE,
) -> pd.DataFrame:
    """
    Leakage-vrije walk-forward simulatie.

    Voor iedere race:
    1. gebruik uitsluitend eerdere races;
    2. selecteer eerdere races binnen hetzelfde contextsegment;
    3. bereken historische mean APE voor GBR/XGB/CAT;
    4. pas select_suggested_model() toe;
    5. evalueer de gekozen model-APE op de huidige race.

    Er wordt niets weggeschreven naar df_overview.
    """

    required = [
        "Year",
        "Race #",
        context_col,
        "APE_pct_GBR",
        "APE_pct_XGB",
        "APE_pct_CAT",
    ]

    missing = [
        c
        for c in required
        if c not in source_df.columns
    ]

    if missing:
        raise ValueError(
            "Walk-forward mist vereiste kolommen: "
            f"{missing}"
        )

    d = source_df.copy()

    # --------------------------------------------------------
    # Numeriek
    # --------------------------------------------------------
    for col in [
        "Year",
        "Race #",
        "APE_pct_GBR",
        "APE_pct_XGB",
        "APE_pct_CAT",
    ]:
        d[col] = pd.to_numeric(
            d[col],
            errors="coerce",
        )

    if context_col == "IsSprintWeekend":
        d[context_col] = pd.to_numeric(
            d[context_col],
            errors="coerce",
        )

    # Alleen chronologisch bruikbare races
    d = d.dropna(
        subset=[
            "Year",
            "Race #",
        ]
    ).copy()

    # Minstens één model-APE nodig voor evaluatie
    d = d[
        d[
            [
                "APE_pct_GBR",
                "APE_pct_XGB",
                "APE_pct_CAT",
            ]
        ]
        .notna()
        .any(axis=1)
    ].copy()

    d = (
        d
        .sort_values(
            [
                "Year",
                "Race #",
            ]
        )
        .reset_index(drop=True)
    )

    d["ContextSegment"] = d.apply(
        lambda row: _context_segment_value(
            row,
            context_col,
        ),
        axis=1,
    )

    results = []

    model_ape_cols = {
        "GBR": "APE_pct_GBR",
        "XGB": "APE_pct_XGB",
        "CAT": "APE_pct_CAT",
    }

    # --------------------------------------------------------
    # Walk forward
    # --------------------------------------------------------
    for current_idx in range(
        len(d)
    ):

        current = d.iloc[
            current_idx
        ]

        current_segment = current[
            "ContextSegment"
        ]

        # HARDE NO-LEAKAGE REGEL:
        # uitsluitend rijen vóór de huidige race
        history = d.iloc[
            :current_idx
        ].copy()

        history_segment = history[
            history["ContextSegment"]
            == current_segment
        ].copy()

        # --------------------------------------------
        # Historische segmentstatistieken
        # --------------------------------------------
        n_races = int(
            len(history_segment)
        )

        mean_apes = {}

        model_ns = {}

        for (
            model_name,
            ape_col,
        ) in model_ape_cols.items():

            valid_values = pd.to_numeric(
                history_segment[
                    ape_col
                ],
                errors="coerce",
            ).dropna()

            model_ns[
                model_name
            ] = int(
                len(valid_values)
            )

            mean_apes[
                model_name
            ] = (
                float(
                    valid_values.mean()
                )
                if len(valid_values)
                else np.nan
            )

        # --------------------------------------------
        # Zelfde inputstructuur als Stap 2 policy
        # --------------------------------------------
        policy_input = pd.Series(
            {
                "n_races": n_races,
                "GBR_mean_APE": mean_apes[
                    "GBR"
                ],
                "XGB_mean_APE": mean_apes[
                    "XGB"
                ],
                "CAT_mean_APE": mean_apes[
                    "CAT"
                ],
            }
        )

        policy_result = (
            select_suggested_model(
                policy_input,
                min_n_segment=int(
                    min_n_segment
                ),
                min_improvement_ape=float(
                    min_improvement_ape
                ),
            )
        )

        selected_model = (
            policy_result[
                "SuggestedModel"
            ]
        )

        selected_ape_col = (
            model_ape_cols[
                selected_model
            ]
        )

        selected_ape = pd.to_numeric(
            current.get(
                selected_ape_col
            ),
            errors="coerce",
        )

        gbr_ape_current = (
            pd.to_numeric(
                current.get(
                    "APE_pct_GBR"
                ),
                errors="coerce",
            )
        )

        # Als gekozen model voor deze historische race
        # geen APE heeft, niet kunstmatig vervangen.
        evaluable = (
            pd.notna(
                selected_ape
            )
            and pd.notna(
                gbr_ape_current
            )
        )

        if evaluable:
            improvement_current = (
                float(
                    gbr_ape_current
                )
                - float(
                    selected_ape
                )
            )
        else:
            improvement_current = np.nan

        result = {
            "Year": int(
                current["Year"]
            ),
            "Race #": int(
                current["Race #"]
            ),
            "gp_slug": current.get(
                "gp_slug",
                np.nan,
            ),
            "Circuit": current.get(
                "Circuit",
                np.nan,
            ),
            "Context": context_col,
            "ContextSegment": (
                current_segment
            ),

            "History_n": n_races,

            "History_n_GBR": model_ns[
                "GBR"
            ],
            "History_n_XGB": model_ns[
                "XGB"
            ],
            "History_n_CAT": model_ns[
                "CAT"
            ],

            "History_GBR_mean_APE": mean_apes[
                "GBR"
            ],
            "History_XGB_mean_APE": mean_apes[
                "XGB"
            ],
            "History_CAT_mean_APE": mean_apes[
                "CAT"
            ],

            "SelectedModel": (
                selected_model
            ),

            "PolicyPassed": (
                policy_result[
                    "PolicyPassed"
                ]
            ),

            "SelectionReason": (
                policy_result[
                    "SuggestedModelReason"
                ]
            ),

            "GBR_APE_current": (
                float(
                    gbr_ape_current
                )
                if pd.notna(
                    gbr_ape_current
                )
                else np.nan
            ),

            "Selected_APE_current": (
                float(
                    selected_ape
                )
                if pd.notna(
                    selected_ape
                )
                else np.nan
            ),

            "Improvement_vs_GBR_current_pp": (
                improvement_current
            ),

            "Evaluable": bool(
                evaluable
            ),
        }

        results.append(
            result
        )

    return pd.DataFrame(
        results
    )


def build_context_model_analysis(
    context_df,
    context_long,
    context_min_improvement,
    context_min_n,
):
    """Build context/model segment analytics for dashboard rendering."""
    # Aggregatie per segment + model
    # --------------------------------------------
    context_stats = (
        context_long
        .groupby(
            [
                "ContextSegment",
                "Model",
            ]
        )["APE_pct"]
        .agg(
            n="count",
            mean_APE="mean",
            median_APE="median",
            p90_APE=lambda x: (
                np.percentile(
                    x,
                    90,
                )
                if len(x)
                else np.nan
            ),
        )
        .reset_index()
    )

    # --------------------------------------------
    # Wide tabel voor vergelijking
    # --------------------------------------------
    mean_wide = (
        context_stats
        .pivot(
            index="ContextSegment",
            columns="Model",
            values="mean_APE",
        )
        .reset_index()
    )

    n_wide = (
        context_stats
        .pivot(
            index="ContextSegment",
            columns="Model",
            values="n",
        )
        .reset_index()
    )

    # Algemene n per segment
    segment_n = (
        context_df
        .dropna(
            subset=["ContextSegment"]
        )
        .groupby(
            "ContextSegment"
        )
        .size()
        .rename("n_races")
        .reset_index()
    )

    summary = mean_wide.merge(
        segment_n,
        on="ContextSegment",
        how="left",
    )

    # --------------------------------------------
    # Beste model
    # --------------------------------------------
    model_cols_present = [
        c
        for c in [
            "GBR",
            "XGB",
            "CAT",
        ]
        if c in summary.columns
    ]

    def best_model_from_row(row):
        values = {
            model: row.get(
                model,
                np.nan,
            )
            for model in model_cols_present
        }

        values = {
            k: v
            for k, v in values.items()
            if pd.notna(v)
        }

        if not values:
            return np.nan

        return min(
            values,
            key=values.get,
        )

    summary[
        "BestModel"
    ] = summary.apply(
        best_model_from_row,
        axis=1,
    )

    # --------------------------------------------
    # Improvement vs GBR
    # --------------------------------------------
    def improvement_vs_gbr(row):

        if (
            "GBR" not in row.index
            or pd.isna(
                row.get("GBR")
            )
        ):
            return np.nan

        best_model = row.get(
            "BestModel"
        )

        if pd.isna(best_model):
            return np.nan

        best_value = row.get(
            best_model
        )

        if pd.isna(best_value):
            return np.nan

        return (
            float(row["GBR"])
            - float(best_value)
        )

    summary[
        "Improvement_vs_GBR_pp"
    ] = summary.apply(
        improvement_vs_gbr,
        axis=1,
    )

    # --------------------------------------------
    # N per individueel model toevoegen
    # --------------------------------------------
    for model_name in [
        "GBR",
        "XGB",
        "CAT",
    ]:
        if model_name in n_wide.columns:

            n_map = (
                n_wide[
                    [
                        "ContextSegment",
                        model_name,
                    ]
                ]
                .rename(
                    columns={
                        model_name:
                        f"n_{model_name}"
                    }
                )
            )

            summary = summary.merge(
                n_map,
                on="ContextSegment",
                how="left",
            )

    # --------------------------------------------
    # Kolomnamen leesbaar maken
    # --------------------------------------------
    summary = summary.rename(
        columns={
            "GBR": "GBR_mean_APE",
            "XGB": "XGB_mean_APE",
            "CAT": "CAT_mean_APE",
        }
    )

    # --------------------------------------------
    # Audit-only selectiepolicy toepassen
    # --------------------------------------------
    policy_results = summary.apply(
        lambda row: select_suggested_model(
            row,
            min_n_segment=int(context_min_n),
            min_improvement_ape=float(
                context_min_improvement
            ),
        ),
        axis=1,
    )

    policy_df = pd.DataFrame(
        policy_results.tolist(),
        index=summary.index,
    )

    for col in policy_df.columns:
        summary[col] = policy_df[col]

    # --------------------------------------------
    # Afronden
    # --------------------------------------------
    numeric_summary_cols = [
        "GBR_mean_APE",
        "XGB_mean_APE",
        "CAT_mean_APE",
        "Improvement_vs_GBR_pp",
        "Suggested_APE",
        "GBR_APE",
        "Suggested_vs_GBR_APE",
    ]

    for col in numeric_summary_cols:
        if col in summary.columns:
            summary[col] = pd.to_numeric(
                summary[col],
                errors="coerce",
            ).round(3)

    # --------------------------------------------

    return {
        "context_stats": context_stats,
        "summary": summary,
    }
