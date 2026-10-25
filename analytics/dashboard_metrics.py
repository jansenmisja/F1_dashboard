"""Pure analytical helpers used by the legacy performance dashboard."""

import numpy as np
import pandas as pd

def dashboard_rolling_p90(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(
        w,
        min_periods=max(2, w // 2),
    ).quantile(0.90)


def dashboard_rolling_median(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(
        w,
        min_periods=max(2, w // 2),
    ).median()


def dashboard_agg_ape_by_group(
    d_kpi: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:

    if group_col not in d_kpi.columns:
        return pd.DataFrame(
            columns=[
                group_col,
                "n",
                "mean_APE",
                "median_APE",
                "p90_APE",
            ]
        )

    return (
        d_kpi
        .dropna(subset=[group_col, "APE_pct"])
        .groupby(group_col)["APE_pct"]
        .agg(
            n="count",
            mean_APE="mean",
            median_APE="median",
            p90_APE=lambda x: (
                np.percentile(x, 90)
                if len(x)
                else np.nan
            ),
        )
        .reset_index()
        .sort_values(
            "mean_APE",
            ascending=False,
        )
    )


def dashboard_agg_ape_by_sprint(
    d_kpi: pd.DataFrame,
) -> pd.DataFrame:

    if "IsSprintWeekend" not in d_kpi.columns:
        return pd.DataFrame(
            columns=[
                "WeekendType",
                "n",
                "mean_APE",
                "median_APE",
                "p90_APE",
            ]
        )

    tmp = d_kpi.dropna(
        subset=[
            "IsSprintWeekend",
            "APE_pct",
        ]
    ).copy()

    if tmp.empty:
        return pd.DataFrame()

    tmp["IsSprintWeekend"] = pd.to_numeric(
        tmp["IsSprintWeekend"],
        errors="coerce",
    )

    tmp = tmp.dropna(
        subset=["IsSprintWeekend"]
    )

    tmp["WeekendType"] = np.where(
        tmp["IsSprintWeekend"]
        .astype(int)
        .eq(1),
        "Sprint weekend",
        "Non-sprint weekend",
    )

    return (
        tmp
        .groupby("WeekendType")["APE_pct"]
        .agg(
            n="count",
            mean_APE="mean",
            median_APE="median",
            p90_APE=lambda x: (
                np.percentile(x, 90)
                if len(x)
                else np.nan
            ),
        )
        .reset_index()
        .sort_values(
            "mean_APE",
            ascending=False,
        )
    )


def build_walk_forward_kpi_comparison(
    wf_results: pd.DataFrame,
    source_df: pd.DataFrame,
) -> pd.DataFrame:

    if wf_results.empty:
        return pd.DataFrame()

    required = [
        "Year",
        "Race #",
        "ActualFastestLap_sec",
        "Pred_GBR",
        "Pred_XGB",
        "Pred_CAT",
    ]

    missing = [
        c for c in required
        if c not in source_df.columns
    ]

    if missing:
        raise ValueError(
            f"Ontbrekende kolommen: {missing}"
        )

    source = source_df[
        required
    ].copy()

    for col in required:
        source[col] = pd.to_numeric(
            source[col],
            errors="coerce",
        )

    result = wf_results.copy()

    result["Year"] = pd.to_numeric(
        result["Year"],
        errors="coerce",
    )

    result["Race #"] = pd.to_numeric(
        result["Race #"],
        errors="coerce",
    )

    result = result.merge(
        source,
        on=["Year", "Race #"],
        how="left",
        validate="many_to_one",
    )

    pred_map = {
        "GBR": "Pred_GBR",
        "XGB": "Pred_XGB",
        "CAT": "Pred_CAT",
    }

    def dynamic_prediction(row):
        model = row.get("SelectedModel")
        col = pred_map.get(model)

        if col is None:
            return np.nan

        return pd.to_numeric(
            row.get(col),
            errors="coerce",
        )

    result["DynamicPrediction_sec"] = (
        result.apply(
            dynamic_prediction,
            axis=1,
        )
    )

    result["GBRPrediction_sec"] = pd.to_numeric(
        result["Pred_GBR"],
        errors="coerce",
    )

    result["Actual_sec"] = pd.to_numeric(
        result["ActualFastestLap_sec"],
        errors="coerce",
    )

    # Errors
    result["GBR_Error_sec"] = (
        result["GBRPrediction_sec"]
        - result["Actual_sec"]
    )

    result["Dynamic_Error_sec"] = (
        result["DynamicPrediction_sec"]
        - result["Actual_sec"]
    )

    result["GBR_AbsError_sec"] = (
        result["GBR_Error_sec"].abs()
    )

    result["Dynamic_AbsError_sec"] = (
        result["Dynamic_Error_sec"].abs()
    )

    # APE
    valid_actual = (
        result["Actual_sec"].notna()
        & result["Actual_sec"].ne(0)
    )

    result["GBR_APE_pct_calc"] = np.nan
    result["Dynamic_APE_pct_calc"] = np.nan

    result.loc[
        valid_actual,
        "GBR_APE_pct_calc",
    ] = (
        result.loc[
            valid_actual,
            "GBR_AbsError_sec",
        ]
        / result.loc[
            valid_actual,
            "Actual_sec",
        ]
        * 100
    )

    result.loc[
        valid_actual,
        "Dynamic_APE_pct_calc",
    ] = (
        result.loc[
            valid_actual,
            "Dynamic_AbsError_sec",
        ]
        / result.loc[
            valid_actual,
            "Actual_sec",
        ]
        * 100
    )

    result["KPI_Evaluable"] = (
        result["Actual_sec"].notna()
        & result["GBRPrediction_sec"].notna()
        & result["DynamicPrediction_sec"].notna()
    )

    result[
        "Dynamic_Improvement_AbsError_sec"
    ] = (
        result["GBR_AbsError_sec"]
        - result["Dynamic_AbsError_sec"]
    )

    return result


def strategy_kpis(
    df: pd.DataFrame,
    error_col: str,
    abs_col: str,
    ape_col: str,
    name: str,
):

    d = df[
        df["KPI_Evaluable"].eq(True)
    ].copy()

    if d.empty:
        return {
            "Strategy": name,
            "n": 0,
            "MAE_sec": np.nan,
            "RMSE_sec": np.nan,
            "Mean_APE_pct": np.nan,
            "Median_APE_pct": np.nan,
            "P90_APE_pct": np.nan,
        }

    err = pd.to_numeric(
        d[error_col],
        errors="coerce",
    )

    abs_err = pd.to_numeric(
        d[abs_col],
        errors="coerce",
    )

    ape = pd.to_numeric(
        d[ape_col],
        errors="coerce",
    )

    return {
        "Strategy": name,
        "n": len(d),
        "MAE_sec": abs_err.mean(),
        "RMSE_sec": np.sqrt(
            np.nanmean(
                np.square(err)
            )
        ),
        "Mean_APE_pct": ape.mean(),
        "Median_APE_pct": ape.median(),
        "P90_APE_pct": np.nanpercentile(
            ape,
            90,
        ),
    }
