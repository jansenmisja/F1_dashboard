import os
from pathlib import Path
from datetime import datetime
import shutil
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from f1_pipeline.config import CONFIG as _HELPER_CONFIG
PROJECT_ROOT = _HELPER_CONFIG.root
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
BACKUP_DIR = DATA_PROCESSED / "backups"


def safe_read_csv(path, **kwargs):
    path = Path(path)
    logger.info(f"Loading CSV: {path}")
    if not path.exists():
        raise FileNotFoundError(f"Bestand niet gevonden: {path}")
    df = pd.read_csv(path, **kwargs)
    logger.info(f"Loaded: {path} rows={len(df)} cols={len(df.columns)}")
    return df












# =========================================================
# AUTO-ADDED REQUIRED HELPERS — F1 ML PROJECT
# =========================================================

import os
import re
import math
import numpy as np
import pandas as pd
from pathlib import Path

def rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def time_str_to_sec(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)

    s = str(x).strip()

    if s == "":
        return np.nan

    s = s.replace(",", ".")

    # formats zoals:
    # "1:23.456" -> 83.456
    # "83.456"   -> 83.456
    # "00:01:23.456" -> 83.456
    parts = s.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
        elif len(parts) == 3:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        else:
            return np.nan
    except Exception:
        return np.nan

def convert_column_time_to_sec(df, col):
    df = df.copy()
    if col not in df.columns:
        raise KeyError(f"Kolom niet gevonden: {col}")
    df[col] = df[col].apply(time_str_to_sec)
    return df

def agg_weather_per_gp(weather_df, gp_col="gp_slug", session_col="session"):
    if weather_df is None or len(weather_df) == 0:
        return pd.DataFrame()

    df = weather_df.copy()

    agg_map = {}
    for c in df.columns:
        if c in [gp_col, session_col]:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            if "rain" in c.lower():
                agg_map[c] = "sum"
            else:
                agg_map[c] = "mean"

    if not agg_map:
        return df[[gp_col, session_col]].drop_duplicates()

    out = (
        df.groupby([gp_col, session_col], dropna=False)
          .agg(agg_map)
          .reset_index()
    )
    return out

def merge_with_fallback(left_df, right_df, primary_keys, fallback_keys, how="left", verbose=True):
    left_df = left_df.copy()
    right_df = right_df.copy()

    # eerst primary merge
    merged_primary = left_df.merge(
        right_df,
        on=primary_keys,
        how=how,
        suffixes=("", "_r")
    )

    right_payload_cols = [c for c in right_df.columns if c not in primary_keys]
    missing_mask = pd.Series(False, index=merged_primary.index)

    if right_payload_cols:
        missing_mask = merged_primary[right_payload_cols].isna().all(axis=1)

    used_fallback = False

    if missing_mask.any():
        used_fallback = True

        left_missing = left_df.loc[missing_mask].copy()

        right_for_fallback = right_df.copy()

        primary_only_cols = [c for c in primary_keys if c not in fallback_keys and c in right_for_fallback.columns]
        right_for_fallback = right_for_fallback.drop(columns=primary_only_cols, errors="ignore")

        fallback_merged = left_missing.merge(
            right_for_fallback,
            on=fallback_keys,
            how=how,
            suffixes=("", "_r")
        )

        for col in fallback_merged.columns:
            if col in merged_primary.columns:
                merged_primary.loc[missing_mask, col] = fallback_merged[col].values

    if verbose:
        if used_fallback:
            print(f"[MERGE] Fallback gebruikt op keys: {fallback_keys}")
        else:
            print(f"[MERGE] Primary key succesvol: {primary_keys}")

    # suffix cleanup
    dup_suffix_cols = [c for c in merged_primary.columns if c.endswith("_r")]
    if dup_suffix_cols:
        merged_primary = merged_primary.drop(columns=dup_suffix_cols, errors="ignore")

    return merged_primary

def clean_feature_names(cols):
    cleaned = []
    for c in cols:
        x = str(c)
        x = x.strip()
        x = x.replace("/", "_per_")
        x = x.replace(" ", "_")
        x = x.replace("-", "_")
        x = re.sub(r"[^A-Za-z0-9_]", "", x)
        x = re.sub(r"_+", "_", x).strip("_")
        cleaned.append(x)
    return cleaned

def dump_feature_weights(model, feature_names, out_path):
    out_path = Path(out_path)

    values = None

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = model.coef_
    else:
        raise AttributeError("Model heeft geen feature_importances_ of coef_ attribuut.")

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": values
    }).sort_values("importance", ascending=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_imp.to_csv(out_path, index=False)
    return df_imp

# ============================================================
# Canonical patched helpers — F1 Fastest Lap Project
# ============================================================

def diag_df(df, name=None, head_n=5, tail_n=5):
    """
    Print standaard diagnostiek:
    shape, columns, missings top, head en tail.
    """
    label = name if name is not None else "df"

    print("")
    print(f"=== DIAG: {label} ===")
    print("shape:", df.shape)

    print("")
    print("columns:")
    print(list(df.columns))

    print("")
    print("missings top:")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    print("")
    print(f"head({head_n}):")
    print(df.head(head_n))

    print("")
    print(f"tail({tail_n}):")
    print(df.tail(tail_n))

    return None


def normalize_name(name):
    """
    Normaliseer namen/slugs conservatief:
    lowercase, trim, non-alnum naar underscore.
    """
    import re
    import pandas as pd

    if pd.isna(name):
        return name

    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")

    return s


def resolve_gp_slug(name_or_circuit, circuit_info_df=None):
    """
    Resolve circuit/racenaam naar gp_slug.
    Voorkeur:
    1. match op Circuit in circuit_info_df
    2. match op gp_slug in circuit_info_df
    3. fallback normalize_name(name_or_circuit)
    """
    import pandas as pd

    if pd.isna(name_or_circuit):
        return name_or_circuit

    key = normalize_name(name_or_circuit)

    if circuit_info_df is not None:
        df = circuit_info_df.copy()

        if "Circuit" in df.columns and "gp_slug" in df.columns:
            tmp = df[["Circuit", "gp_slug"]].dropna().copy()
            tmp["_norm_circuit"] = tmp["Circuit"].apply(normalize_name)

            hit = tmp.loc[tmp["_norm_circuit"] == key, "gp_slug"]
            if len(hit) > 0:
                return hit.iloc[0]

        if "gp_slug" in df.columns:
            tmp = df[["gp_slug"]].dropna().copy()
            tmp["_norm_slug"] = tmp["gp_slug"].apply(normalize_name)

            hit = tmp.loc[tmp["_norm_slug"] == key, "gp_slug"]
            if len(hit) > 0:
                return hit.iloc[0]

    manual_map = {
        "australian": "australia",
        "chinese": "china",
        "british": "great_britain",
        "canadian": "canada",
        "monaco_grand_prix": "monaco",
        "circuit_gilles_villeneuve": "canada",
        "gilles_villeneuve": "canada",
    }

    return manual_map.get(key, key)


def round_float_columns(df, ndigits=3):
    """
    Rond alle float-kolommen af op ndigits.
    Geeft een kopie terug.
    """
    out = df.copy()

    float_cols = out.select_dtypes(include=["float", "float64", "float32"]).columns
    if len(float_cols) > 0:
        out[float_cols] = out[float_cols].round(ndigits)

    return out



def backup_file(path):
    """
    Maak vóór een write altijd een backup van een bestaand bestand.

    - Gebruikt centrale _BACKUP_DIR.
    - Overschrijft nooit een bestaande backup.
    - Meerdere backups binnen dezelfde seconde krijgen:
        _01, _02, _03, ...
    - Retourneert Path naar backup.
    - Retourneert None als bronbestand niet bestaat.
    """

    from pathlib import Path
    from datetime import datetime
    import shutil

    src = Path(path)

    if not src.exists():
        return None

    if not src.is_file():
        raise ValueError(
            f"backup_file verwacht een bestand: {src}"
        )

    backup_dir = Path(_BACKUP_DIR)

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    base_name = (
        f"{src.name}.bak.{timestamp}"
    )

    destination = (
        backup_dir / base_name
    )

    if destination.exists():

        counter = 1

        while True:

            candidate = (
                backup_dir
                / f"{base_name}_{counter:02d}"
            )

            if not candidate.exists():

                destination = candidate
                break

            counter += 1

            if counter > 9999:

                raise RuntimeError(
                    "Geen unieke backupnaam beschikbaar "
                    f"voor {src}"
                )

    shutil.copy2(
        src,
        destination,
    )

    if not destination.exists():

        raise RuntimeError(
            f"Backup niet aangemaakt: {destination}"
        )

    if (
        destination.stat().st_size
        != src.stat().st_size
    ):

        raise RuntimeError(
            "Backup size mismatch:\n"
            f"source={src.stat().st_size}\n"
            f"backup={destination.stat().st_size}"
        )

    return destination


def safe_to_csv(df, path, index=False):
    """
    Veilig schrijven:
    1. maak backup vóór overschrijven als bestand bestaat
    2. rond float-kolommen af op 3 decimals
    3. schrijf CSV
    """
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        backup_file(path)

    out = round_float_columns(df, ndigits=3)
    out.to_csv(path, index=index)

    return path


def get_model_log_path():
    """
    Canoniek model logpad.
    """
    return str(_HELPER_CONFIG.model_log_path)


# === AUTO-PATCH: F1_HELPERS_CANONICAL_V1 ===
# Deze sectie is automatisch toegevoegd door de fase-1 preflight.
# Functies onderaan het bestand overschrijven oudere definities veilig.


import re as _re
import shutil as _shutil
import unicodedata as _unicodedata
from datetime import datetime as _datetime
from pathlib import Path as _Path

import numpy as _np
import pandas as _pd


_PROJECT_ROOT = _HELPER_CONFIG.root
_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
_BACKUP_DIR = _PROCESSED_DIR / "backups"
_MODEL_LOG_PATH = _PROJECT_ROOT / "log" / "model_log.csv"


def round_float_columns(df, ndigits=3):
    """Retourneer kopie met alle float-kolommen afgerond."""
    out = df.copy()
    float_cols = out.select_dtypes(include=["float", "float16", "float32", "float64"]).columns
    if len(float_cols) > 0:
        out.loc[:, float_cols] = out.loc[:, float_cols].round(ndigits)
    return out




def safe_read_csv(path):
    """Lees CSV via absoluut pad met duidelijke foutmelding."""
    csv_path = _Path(path)

    if not csv_path.is_absolute():
        raise ValueError(f"Gebruik een absoluut pad: {csv_path}")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV niet gevonden: {csv_path}")

    return _pd.read_csv(csv_path)


def safe_to_csv(df, path, index=False):
    """
    Veilig opslaan:
      1. absoluut pad afdwingen;
      2. backup maken vóór overschrijven;
      3. float-kolommen afronden op 3 decimalen;
      4. CSV schrijven.
    """
    csv_path = _Path(path)

    if not csv_path.is_absolute():
        raise ValueError(f"Gebruik een absoluut pad: {csv_path}")

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Verplichte volgorde: eerst backup, daarna rounding, daarna schrijven.
    backup_file(csv_path)
    out = round_float_columns(df, ndigits=3)
    out.to_csv(csv_path, index=index)

    return str(csv_path)


def diag_df(df, name=None, head_n=5, tail_n=5):
    """Standaarddiagnostiek na dataset-update of wijziging."""
    label = name or "dataframe"

    print(f"\n=== DIAGNOSTIEK: {label} ===")
    print(f"shape: {df.shape}")

    print("\ncolumns:")
    print(list(df.columns))

    print("\nmissings top:")
    missings = df.isna().sum().sort_values(ascending=False)
    print(missings[missings > 0].head(15))

    print(f"\nhead({head_n}):")
    print(df.head(head_n))

    print(f"\ntail({tail_n}):")
    print(df.tail(tail_n))


def rmse(y_true, y_pred):
    """Root mean squared error."""
    true = _np.asarray(y_true, dtype=float)
    pred = _np.asarray(y_pred, dtype=float)
    return float(_np.sqrt(_np.mean((true - pred) ** 2)))


def normalize_name(name):
    """Normaliseer tekst voor robuuste matching."""
    if name is None or (_pd.isna(name) if not isinstance(name, str) else False):
        return ""

    text = str(name).strip().lower()
    text = _unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not _unicodedata.combining(char))
    text = text.replace("&", " and ")
    text = _re.sub(r"[^a-z0-9]+", "_", text)
    text = _re.sub(r"_+", "_", text).strip("_")

    aliases = {
        "australian": "australia",
        "chinese": "china",
        "british": "great_britain",
        "great_british": "great_britain",
        "monte_carlo": "monaco",
        "monaco_gp": "monaco",
    }

    return aliases.get(text, text)


def resolve_gp_slug(name_or_circuit, circuit_info_df=None):
    """
    Resolve gp_slug:
      1. match op bestaande gp_slug;
      2. fallback match op Circuit;
      3. fallback naar genormaliseerde input.
    """
    target = normalize_name(name_or_circuit)

    if circuit_info_df is not None and not circuit_info_df.empty:
        info = circuit_info_df.copy()

        if "gp_slug" in info.columns:
            slug_norm = info["gp_slug"].map(normalize_name)
            matches = info.loc[slug_norm == target, "gp_slug"]

            if len(matches) == 1:
                return str(matches.iloc[0])

        if "Circuit" in info.columns and "gp_slug" in info.columns:
            circuit_norm = info["Circuit"].map(normalize_name)
            matches = info.loc[circuit_norm == target, "gp_slug"]

            if len(matches) == 1:
                return str(matches.iloc[0])

    return target


def time_str_to_sec(x):
    """Converteer tijdwaarden naar seconden als float."""
    if x is None or (_pd.isna(x) if not isinstance(x, str) else False):
        return _np.nan

    if isinstance(x, (int, float, _np.integer, _np.floating)):
        return float(x)

    text = str(x).strip()

    if not text:
        return _np.nan

    # Ondersteunt bijvoorbeeld:
    #   78.123
    #   1:18.123
    #   01:18.123
    #   0:01:18.123
    parts = text.split(":")

    try:
        if len(parts) == 1:
            return float(parts[0])

        if len(parts) == 2:
            minutes, seconds = parts
            return (float(minutes) * 60.0) + float(seconds)

        if len(parts) == 3:
            hours, minutes, seconds = parts
            return (
                (float(hours) * 3600.0)
                + (float(minutes) * 60.0)
                + float(seconds)
            )

    except ValueError:
        return _np.nan

    return _np.nan


def convert_column_time_to_sec(df, col):
    """Converteer één dataframekolom naar seconden."""
    if col not in df.columns:
        raise KeyError(f"Kolom ontbreekt: {col}")

    out = df.copy()
    out[col] = out[col].map(time_str_to_sec).astype(float)
    return out


def agg_weather_per_gp(weather_df, gp_col="gp_slug", session_col="session"):
    """
    Aggregeer hourly weather naar Q_* en R_* features.
    Verwachte inputkolommen:
      - gp_slug
      - session
      - temp_c
      - rain_mm
      - humidity_pct
    """
    required = {gp_col, session_col, "temp_c", "rain_mm", "humidity_pct"}
    missing = sorted(required - set(weather_df.columns))

    if missing:
        raise KeyError(f"Weather-kolommen ontbreken: {missing}")

    grouped = (
        weather_df
        .groupby([gp_col, session_col], dropna=False)
        .agg(
            airtemp_mean=("temp_c", "mean"),
            airtemp_max=("temp_c", "max"),
            rain_sum=("rain_mm", "sum"),
            humidity_mean=("humidity_pct", "mean"),
        )
        .reset_index()
    )

    wide = grouped.pivot(index=gp_col, columns=session_col)
    wide.columns = [
        f"{str(session).upper()}_{metric}"
        for metric, session in wide.columns
    ]

    return wide.reset_index()


def merge_with_fallback(
    left_df,
    right_df,
    primary_keys,
    fallback_keys,
    how="left",
    verbose=True,
):
    """
    Merge primair op ['Year', 'gp_slug'].
    Vul ontbrekende rechterkolommen aan via fallback ['Year', 'Circuit'].
    """
    if how != "left":
        raise ValueError("merge_with_fallback ondersteunt operationeel alleen how='left'.")

    left = left_df.copy()
    right = right_df.copy()

    left["__merge_row_id__"] = _np.arange(len(left))

    right_value_cols = [
        col for col in right.columns
        if col not in set(primary_keys) | set(fallback_keys)
    ]

    primary_right = right.drop_duplicates(subset=primary_keys, keep="last")

    merged = left.merge(
        primary_right[primary_keys + right_value_cols],
        on=primary_keys,
        how="left",
        validate="m:1",
    )

    if right_value_cols:
        missing_primary = merged[right_value_cols].isna().all(axis=1)
    else:
        missing_primary = _pd.Series(False, index=merged.index)

    fallback_count = int(missing_primary.sum())

    if fallback_count > 0:
        fallback_right = right.drop_duplicates(subset=fallback_keys, keep="last")

        fallback_values = (
            left.loc[missing_primary, ["__merge_row_id__"] + fallback_keys]
            .merge(
                fallback_right[fallback_keys + right_value_cols],
                on=fallback_keys,
                how="left",
                validate="m:1",
            )
            .set_index("__merge_row_id__")
        )

        merged = merged.set_index("__merge_row_id__")

        for col in right_value_cols:
            merged.loc[fallback_values.index, col] = (
                merged.loc[fallback_values.index, col]
                .fillna(fallback_values[col])
            )

        merged = merged.reset_index()

    merged = (
        merged
        .sort_values("__merge_row_id__")
        .drop(columns="__merge_row_id__")
        .reset_index(drop=True)
    )

    if verbose:
        print(
            "merge_with_fallback: "
            f"primary_keys={primary_keys}, "
            f"fallback_keys={fallback_keys}, "
            f"fallback_rows={fallback_count}"
        )

    return merged


def clean_feature_names(cols):
    """Maak featurenamen modelveilig en reproduceerbaar."""
    cleaned = []

    for col in cols:
        name = str(col).strip()
        name = _re.sub(r"[^A-Za-z0-9_]+", "_", name)
        name = _re.sub(r"_+", "_", name).strip("_")
        cleaned.append(name)

    return cleaned


def dump_feature_weights(model, feature_names, out_path):
    """Schrijf feature weights veilig weg naar CSV."""
    if hasattr(model, "feature_importances_"):
        weights = _np.asarray(model.feature_importances_, dtype=float)

    elif hasattr(model, "coef_"):
        weights = _np.asarray(model.coef_, dtype=float).reshape(-1)

    else:
        raise AttributeError(
            "Model bevat geen feature_importances_ en geen coef_."
        )

    if len(weights) != len(feature_names):
        raise ValueError(
            "Aantal feature weights wijkt af van aantal featurenamen."
        )

    out = _pd.DataFrame(
        {
            "feature": clean_feature_names(feature_names),
            "weight": weights,
        }
    ).sort_values("weight", ascending=False)

    return safe_to_csv(out, out_path, index=False)


def get_model_log_path():
    """Retourneer het canonieke append-only model-logpad."""
    return str(_MODEL_LOG_PATH)




# ============================================================
# PATCH: canonical safe_to_csv return behaviour
# ============================================================

def safe_to_csv(df, path, index=False):
    """
    Veilig opslaan:
    1. backup vóór overschrijven;
    2. floatkolommen afronden op 3 decimalen;
    3. CSV schrijven;
    4. opgeslagen dataframe retourneren.
    """
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    backup_file(path)

    out = round_float_columns(
        df,
        ndigits=3,
    )

    out.to_csv(
        path,
        index=index,
    )

    return out



# ------------------------------------------------------------
# Canonieke projectpaden voor Streamlit control panel
# ------------------------------------------------------------
PROJECT_ROOT = str(_HELPER_CONFIG.root)
PROCESSED_DIR = f"{PROJECT_ROOT}/data/processed"
RAW_DIR = f"{PROJECT_ROOT}/data/raw"
BACKUP_DIR = f"{PROCESSED_DIR}/backups"
LOG_DIR = f"{PROJECT_ROOT}/log"

DF_TRAINING_PATH = f"{PROCESSED_DIR}/df_training.csv"
DF_TEST_PATH = f"{PROCESSED_DIR}/df_test.csv"
DF_OVERVIEW_PATH = f"{PROCESSED_DIR}/df_overview.csv"
CIRCUIT_INFO_PATH = f"{RAW_DIR}/circuit_info.csv"
MODEL_LOG_PATH = f"{LOG_DIR}/model_log.csv"


# ------------------------------------------------------------
# Helper preflight check — Streamlit + pipeline safety gate
# ------------------------------------------------------------
def helper_preflight_check():
    import inspect
    import os
    import tempfile
    from pathlib import Path

    import pandas as pd

    expected_backup_dir = (
        str(_HELPER_CONFIG.backup_dir)
    )
    expected_log_path = (
        str(_HELPER_CONFIG.model_log_path)
    )

    required_signatures = {
        "safe_read_csv": ["path"],
        "safe_to_csv": ["df", "path", "index"],
        "backup_file": ["path"],
        "diag_df": ["df", "name", "head_n", "tail_n"],
        "rmse": ["y_true", "y_pred"],
        "normalize_name": ["name"],
        "resolve_gp_slug": ["name_or_circuit", "circuit_info_df"],
        "time_str_to_sec": ["x"],
        "convert_column_time_to_sec": ["df", "col"],
        "agg_weather_per_gp": ["weather_df", "gp_col", "session_col"],
        "merge_with_fallback": [
            "left_df",
            "right_df",
            "primary_keys",
            "fallback_keys",
            "how",
            "verbose",
        ],
        "clean_feature_names": ["cols"],
        "dump_feature_weights": ["model", "feature_names", "out_path"],
        "get_model_log_path": [],
        "round_float_columns": ["df", "ndigits"],
    }

    errors = []
    warnings = []

    # 1. Aanwezigheid en signatures controleren
    for helper_name, expected_params in required_signatures.items():
        helper = globals().get(helper_name)

        if helper is None:
            errors.append(f"Helper ontbreekt: {helper_name}")
            continue

        if not callable(helper):
            errors.append(f"Helper is niet callable: {helper_name}")
            continue

        actual_params = list(inspect.signature(helper).parameters.keys())

        if actual_params != expected_params:
            errors.append(
                f"Signature wijkt af voor {helper_name}: "
                f"verwacht {expected_params}, gevonden {actual_params}"
            )

    # Geen verdere tests uitvoeren als kernhelpers ontbreken
    blocking_helpers = [
        "safe_to_csv",
        "backup_file",
        "round_float_columns",
        "get_model_log_path",
    ]

    if any(globals().get(name) is None for name in blocking_helpers):
        return {
            "ok": False,
            "errors": errors,
            "warnings": warnings,
            "checked_helpers": list(required_signatures.keys()),
            "backup_dir": expected_backup_dir,
            "model_log_path": expected_log_path,
        }

    # 2. Vast logpad controleren
    try:
        actual_log_path = get_model_log_path()
        if actual_log_path != expected_log_path:
            errors.append(
                "get_model_log_path verwijst niet naar het canonieke logpad: "
                f"{actual_log_path}"
            )
    except Exception as exc:
        errors.append(f"get_model_log_path faalt: {exc}")

    # 3. Backupmap controleren
    actual_backup_dir = globals().get("BACKUP_DIR")

    if actual_backup_dir != expected_backup_dir:
        errors.append(
            "BACKUP_DIR wijkt af van de vaste backupmap: "
            f"{actual_backup_dir}"
        )

    # 4. Rounding functioneel testen
    try:
        test_df = pd.DataFrame({"value": [1.23456]})
        rounded_df = round_float_columns(test_df, ndigits=3)

        if float(rounded_df.loc[0, "value"]) != 1.235:
            errors.append(
                "round_float_columns(df, ndigits=3) rondt niet correct af."
            )
    except Exception as exc:
        errors.append(f"round_float_columns rooktest faalt: {exc}")

    # 5. safe_to_csv functioneel testen:
    #    backup vóór overschrijven + afronding vóór schrijven
    created_backups = []

    try:
        os.makedirs(expected_backup_dir, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=str(_HELPER_CONFIG.root)) as temp_dir:
            smoke_path = os.path.join(temp_dir, "preflight_smoke.csv")

            # Maak bestaand bestand zodat backup verplicht is
            pd.DataFrame({"value": [9.99999]}).to_csv(
                smoke_path,
                index=False,
            )

            before = set(Path(expected_backup_dir).glob(
                "preflight_smoke_before_write_*.csv"
            ))

            from f1_pipeline.io import storage_transaction
            with storage_transaction():
                safe_to_csv(
                    pd.DataFrame({"value": [1.23456]}),
                    smoke_path,
                    index=False,
                )

            after = set(Path(expected_backup_dir).glob(
                "preflight_smoke_before_write_*.csv"
            ))

            created_backups = list(after - before)

            if not created_backups:
                errors.append(
                    "safe_to_csv rooktest: geen backup aangemaakt vóór overschrijven."
                )

            written = pd.read_csv(smoke_path)

            if float(written.loc[0, "value"]) != 1.235:
                errors.append(
                    "safe_to_csv rooktest: float niet afgerond op 3 decimalen."
                )

    except Exception as exc:
        errors.append(f"safe_to_csv rooktest faalt: {exc}")

    finally:
        for backup_path in created_backups:
            try:
                backup_path.unlink()
            except Exception:
                warnings.append(
                    f"Tijdelijke rooktest-backup kon niet worden verwijderd: "
                    f"{backup_path}"
                )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "checked_helpers": list(required_signatures.keys()),
        "backup_dir": expected_backup_dir,
        "model_log_path": expected_log_path,
    }


# ------------------------------------------------------------
# PATCH: diag_df retourneert diagnostiek voor Streamlit
# ------------------------------------------------------------
def diag_df(df, name=None, head_n=5, tail_n=5):
    title = name or "dataframe"

    result = {
        "name": title,
        "shape": df.shape,
        "columns": list(df.columns),
        "missings_top": (
            df.isna()
            .sum()
            .sort_values(ascending=False)
            .head(15)
        ),
        "head": df.head(head_n),
        "tail": df.tail(tail_n),
    }

    return result
# Step9E: helpers must not bypass coordinated canonical CSV storage.
def safe_read_csv(path, **kwargs):
    from f1_pipeline.io import read_csv_checked
    return read_csv_checked(path, **kwargs)

def safe_to_csv(df, path, index=False):
    from f1_pipeline.io import safe_commit_csv, read_csv_checked
    safe_commit_csv(df, path, index=index)
    return read_csv_checked(path)

# Step9F1: exact app helper contract; canonical storage remains in use.
def safe_read_csv(path):
    from f1_pipeline.io import read_csv_checked
    return read_csv_checked(path)
