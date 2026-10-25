
"""
Transactional CSV I/O for the F1 Fastest Lap pipeline.

Official write path:

backup
→ stage temporary file
→ readback
→ validation
→ atomic replace
→ final readback
→ validation

No Streamlit dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional, Union
import os
import shutil
import uuid

import pandas as pd

from .config import CONFIG


Validator = Callable[[pd.DataFrame], object]


@dataclass
class CommitResult:
    ok: bool
    path: Path
    backup_path: Optional[Path]
    rows: int
    columns: int
    message: str


# ============================================================
# Generic read
# ============================================================

def read_csv_checked(
    path: Union[str, Path],
    *,
    required: bool = True,
    **kwargs,
) -> pd.DataFrame:

    path = Path(path)

    if not path.exists():

        if required:
            raise FileNotFoundError(
                f"CSV not found: {path}"
            )

        return pd.DataFrame()

    df = pd.read_csv(
        path,
        **kwargs,
    )

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            f"CSV read did not return DataFrame: {path}"
        )

    return df


# ============================================================
# Canonical float rounding
# ============================================================

def round_float_columns(
    df: pd.DataFrame,
    decimals: int = 3,
) -> pd.DataFrame:

    result = df.copy()

    float_cols = (
        result
        .select_dtypes(
            include=["float"]
        )
        .columns
    )

    if len(float_cols):

        result.loc[
            :,
            float_cols,
        ] = (
            result[
                float_cols
            ]
            .round(
                decimals
            )
        )

    return result


# ============================================================
# Validator adapter
# ============================================================

def _run_validator(
    validator: Optional[Validator],
    df: pd.DataFrame,
) -> None:

    if validator is None:
        return

    result = validator(
        df
    )

    # ContractResult-style object
    if hasattr(
        result,
        "ok",
    ):

        if not bool(
            result.ok
        ):

            errors = getattr(
                result,
                "errors",
                [],
            )

            raise ValueError(
                "Validation failed: "
                + " | ".join(
                    str(x)
                    for x in errors
                )
            )

        return

    # bool validator
    if isinstance(
        result,
        bool,
    ):

        if not result:
            raise ValueError(
                "Validation returned False."
            )

        return

    # None = validator raised itself on failure
    if result is None:
        return

    raise TypeError(
        "Validator must return ContractResult-like object, "
        "bool or None."
    )


# ============================================================
# Backup
# ============================================================

def backup_csv(
    path: Union[str, Path],
    *,
    backup_dir: Optional[Union[str, Path]] = None,
) -> Optional[Path]:

    path = Path(path)

    if not path.exists():
        return None

    if backup_dir is None:
        backup_dir = CONFIG.backup_dir

    backup_dir = Path(
        backup_dir
    )

    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )

    backup_path = (
        backup_dir
        / (
            f"{path.stem}_before_write_"
            f"{stamp}{path.suffix}"
        )
    )

    shutil.copy2(
        path,
        backup_path,
    )

    return backup_path


# ============================================================
# Single transactional commit
# ============================================================

def safe_commit_csv(
    df: pd.DataFrame,
    path: Union[str, Path],
    *,
    validator: Optional[Validator] = None,
    backup_dir: Optional[Union[str, Path]] = None,
    round_decimals: int = 3,
    index: bool = False,
) -> CommitResult:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared = round_float_columns(
        df,
        decimals=round_decimals,
    )

    # Validate before touching destination
    _run_validator(
        validator,
        prepared,
    )

    backup_path = backup_csv(
        path,
        backup_dir=backup_dir,
    )

    temp_path = (
        path.parent
        / (
            f".{path.name}."
            f"{uuid.uuid4().hex}.tmp"
        )
    )

    destination_existed = (
        path.exists()
    )

    try:

        # --------------------------------------------
        # Stage
        # --------------------------------------------
        prepared.to_csv(
            temp_path,
            index=index,
        )

        # --------------------------------------------
        # Stage readback
        # --------------------------------------------
        staged = read_csv_checked(
            temp_path
        )

        _run_validator(
            validator,
            staged,
        )

        if len(staged) != len(prepared):

            raise ValueError(
                "Stage readback row-count mismatch: "
                f"{len(staged)} != {len(prepared)}"
            )

        if list(
            staged.columns
        ) != list(
            prepared.columns
        ):

            raise ValueError(
                "Stage readback column-order mismatch."
            )

        # --------------------------------------------
        # Atomic destination replacement
        # --------------------------------------------
        os.replace(
            temp_path,
            path,
        )

        # --------------------------------------------
        # Final readback
        # --------------------------------------------
        final = read_csv_checked(
            path
        )

        _run_validator(
            validator,
            final,
        )

        if len(final) != len(prepared):

            raise ValueError(
                "Final readback row-count mismatch."
            )

        if list(
            final.columns
        ) != list(
            prepared.columns
        ):

            raise ValueError(
                "Final readback column-order mismatch."
            )

        return CommitResult(
            ok=True,
            path=path,
            backup_path=backup_path,
            rows=len(final),
            columns=len(final.columns),
            message="commit successful",
        )

    except Exception:

        # --------------------------------------------
        # Cleanup temp
        # --------------------------------------------
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception:
            pass

        # --------------------------------------------
        # Rollback destination
        # --------------------------------------------
        if backup_path is not None:

            shutil.copy2(
                backup_path,
                path,
            )

        elif not destination_existed:

            try:
                if path.exists():
                    path.unlink()
            except Exception:
                pass

        raise


# ============================================================
# Multi-file transactional commit
# ============================================================

def safe_commit_many(
    items: Dict[
        Union[str, Path],
        pd.DataFrame,
    ],
    *,
    validators: Optional[
        Dict[
            Union[str, Path],
            Validator,
        ]
    ] = None,
    backup_dir: Optional[Union[str, Path]] = None,
    round_decimals: int = 3,
    prepare_commit=None,
    commit_guard=None,
    after_commit=None,
) -> Dict[Path, CommitResult]:
    """
    Logical transaction for multiple CSV files.

    All DataFrames are:
    - prepared
    - validated
    - staged
    - readback validated

    before any destination is replaced.

    If a destination replacement/final validation fails,
    all already-replaced destinations are restored.
    """

    validators = validators or {}

    normalized = {
        Path(path):
            round_float_columns(
                df,
                decimals=round_decimals,
            )
        for path, df in items.items()
    }

    # --------------------------------------------
    # Prevalidate everything
    # --------------------------------------------
    for path, df in normalized.items():

        validator = (
            validators.get(path)
            or validators.get(str(path))
        )

        _run_validator(
            validator,
            df,
        )

    backups = {}
    staged_paths = {}
    existed_before = {path: path.exists() for path in normalized}

    try:

        # ----------------------------------------
        # Backup + stage all
        # ----------------------------------------
        for path, df in normalized.items():

            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            existed_before[path] = (
                path.exists()
            )

            backups[path] = backup_csv(
                path,
                backup_dir=backup_dir,
            )

            temp_path = (
                path.parent
                / (
                    f".{path.name}."
                    f"{uuid.uuid4().hex}.tmp"
                )
            )

            staged_paths[path] = temp_path

            df.to_csv(
                temp_path,
                index=False,
            )

            staged = read_csv_checked(
                temp_path
            )

            validator = (
                validators.get(path)
                or validators.get(str(path))
            )

            _run_validator(
                validator,
                staged,
            )

            if len(staged) != len(df):
                raise ValueError(
                    f"Stage row mismatch for {path}"
                )

            if list(
                staged.columns
            ) != list(
                df.columns
            ):
                raise ValueError(
                    f"Stage column mismatch for {path}"
                )

            staged_paths[
                path
            ] = temp_path

        # Optional owning-service finalization of timing/hash metadata.
        # Re-stage and revalidate the finalized payloads before replacing anything.
        if prepare_commit is not None:
            prepare_commit(normalized)
            for path, frame in normalized.items():
                frame = round_float_columns(frame, decimals=round_decimals)
                normalized[path] = frame
                frame.to_csv(staged_paths[path], index=False)
                staged = read_csv_checked(staged_paths[path])
                validator = validators.get(path) or validators.get(str(path))
                _run_validator(validator, staged)
                if len(staged) != len(frame) or list(staged.columns) != list(frame.columns):
                    raise ValueError("Finalized stage shape mismatch")

        # ----------------------------------------
        # Commit all
        # ----------------------------------------
        committed = []

        for path, temp_path in (
            staged_paths.items()
        ):

            if commit_guard is not None:
                commit_guard()

            os.replace(
                temp_path,
                path,
            )

            committed.append(
                path
            )

        # ----------------------------------------
        # Final validation all
        # ----------------------------------------
        results = {}

        for path, expected_df in (
            normalized.items()
        ):

            final = read_csv_checked(
                path
            )

            validator = (
                validators.get(path)
                or validators.get(str(path))
            )

            _run_validator(
                validator,
                final,
            )

            if len(final) != len(expected_df):
                raise ValueError(
                    f"Final row mismatch for {path}"
                )

            results[path] = CommitResult(
                ok=True,
                path=path,
                backup_path=backups[path],
                rows=len(final),
                columns=len(final.columns),
                message="multi-file commit successful",
            )

        if after_commit is not None:
            after_commit()

        return results

    except Exception:

        # ----------------------------------------
        # Remove staged temp files
        # ----------------------------------------
        for temp_path in (
            staged_paths.values()
        ):

            try:
                if temp_path.exists():
                    temp_path.unlink()
            except Exception:
                pass

        # ----------------------------------------
        # Roll back every destination
        # ----------------------------------------
        for path in normalized:

            backup_path = backups.get(
                path
            )

            if (
                backup_path is not None
                and backup_path.exists()
            ):

                shutil.copy2(
                    backup_path,
                    path,
                )

            elif not existed_before.get(
                path,
                False,
            ):

                try:
                    if path.exists():
                        path.unlink()
                except Exception:
                    pass

        raise


# ============================================================
# Lightweight infrastructure preflight
# ============================================================

def io_preflight() -> Dict[str, object]:

    checks = {
        "project_root_exists":
            CONFIG.root.exists(),

        "processed_dir_exists":
            CONFIG.processed_dir.exists(),

        "training_exists":
            CONFIG.training_path.exists(),

        "test_exists":
            CONFIG.test_path.exists(),

        "overview_exists":
            CONFIG.overview_path.exists(),

        "backup_dir_parent_exists":
            CONFIG.backup_dir.parent.exists(),
    }

    return {
        "ok":
            all(
                checks.values()
            ),

        "checks":
            checks,
    }

# Step9C sandbox proposal; service integration is still pending.
_step9c_original_read = read_csv_checked
_step9c_original_preflight = io_preflight

def storage_transaction():
    from .storage_transactions import transaction
    return transaction()

def storage_versions(paths):
    from .storage_transactions import versions
    return versions(paths)

def recover_storage():
    from .storage_transactions import recover
    return recover()

def read_csv_checked(path, *, required=True, **kwargs):
    from .storage_transactions import read
    return read(path, required=required, **kwargs)

def io_preflight():
    recover_storage()
    return _step9c_original_preflight()

def safe_commit_many(items, *, validators=None, backup_dir=None, round_decimals=3,
                     prepare_commit=None, commit_guard=None, after_commit=None,
                     expected_versions=None):
    from .storage_transactions import commit
    return commit(items, validators=validators, backup_dir=backup_dir,
                  round_decimals=round_decimals, prepare_commit=prepare_commit,
                  commit_guard=commit_guard, after_commit=after_commit,
                  expected_versions=expected_versions)

def safe_commit_csv(df, path, *, validator=None, backup_dir=None, round_decimals=3, index=False):
    if index:
        raise ValueError('Step9C sandbox proposal supports canonical index=False CSVs only')
    path = Path(path)
    return safe_commit_many({path: df}, validators={path: validator},
                            backup_dir=backup_dir, round_decimals=round_decimals)[path.resolve()]
