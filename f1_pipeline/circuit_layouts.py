
"""Register a missing circuit layout and synchronize one unfrozen event."""

def fill_missing_circuit_layout(year, gp_slug, layout_slug, *, commit=False):
    import re
    import pandas as pd
    from pandas.testing import assert_frame_equal

    from .config import CONFIG
    from .io import read_csv_checked, safe_commit_many
    from .storage_transactions import transaction
    from .event_circuits import resolve_event_circuit
    from .readiness import (
        _assert_inputs_not_frozen,
        _canonical_operational_validators,
        require_one_event,
    )

    year = int(year)
    gp_slug = str(gp_slug).strip()
    layout_slug = str(layout_slug).strip()
    if not re.fullmatch(r"[a-z0-9_]+", layout_slug):
        raise ValueError("Ongeldige layoutslug.")

    def missing(value):
        return pd.isna(value) or not str(value).strip()

    def equivalent(actual, expected):
        if list(actual.columns) != list(expected.columns):
            raise ValueError("Onverwachte kolomwijziging.")
        if len(actual) != len(expected):
            raise ValueError("Onverwachte rijwijziging.")

        # Vergelijk CSV-readback onafhankelijk van inferred datatypes.
        left = actual.reset_index(drop=True).copy()
        right = expected.reset_index(drop=True).copy()
        for col in left.columns:
            a = left[col].astype("string").fillna("")
            b = right[col].astype("string").fillna("")
            na = pd.to_numeric(a.replace("", pd.NA), errors="coerce")
            nb = pd.to_numeric(b.replace("", pd.NA), errors="coerce")
            numeric_a = ((a == "") | na.notna()).all()
            numeric_b = ((b == "") | nb.notna()).all()
            if numeric_a and numeric_b:
                left[col] = na.astype("Float64")
                right[col] = nb.astype("Float64")
            else:
                left[col] = a
                right[col] = b
        assert_frame_equal(
            left, right, check_dtype=False,
            check_exact=False, rtol=0, atol=1e-12,
        )

    with transaction():
        dt, do, mt, mo, rt, ro = _assert_inputs_not_frozen(year, gp_slug)

        # Deze herstelroute is bewust beperkt tot de aangetroffen status.
        for row in (rt, ro):
            if str(row["RaceStatus"]) != "circuit_regime_ready":
                raise ValueError("Herstel vereist circuit_regime_ready.")
            for field in (
                "FeatureSnapshotAfterQualifyingUTC",
                "FeatureSnapshotTimestampUTC",
                "ActualFastestLap_sec",
                "PredictedFastestLap_sec",
                "Pred_GBR", "Pred_XGB", "Pred_CAT",
            ):
                if not missing(row.get(field)):
                    raise ValueError("Bestaande input/output blokkeert herstel: " + field)

        ci = read_csv_checked(CONFIG.circuit_info_path)
        _, reference_slug, resolved_layout = resolve_event_circuit(
            year, gp_slug, ci
        )
        if reference_slug != gp_slug:
            raise ValueError("Event met afwijkende fysieke circuitmapping.")
        if resolved_layout is not None and resolved_layout != layout_slug:
            raise ValueError("Bestaande eventreferentie heeft een andere layout.")

        mask = ci["gp_slug"].astype("string").str.strip().eq(reference_slug)
        if int(mask.sum()) != 1:
            raise ValueError("Circuitreferentie is niet uniek.")
        if ci["gp_slug"].isna().any() or ci["gp_slug"].astype(
            "string"
        ).str.strip().duplicated().any():
            raise ValueError("Circuitreferentie bevat lege of dubbele sleutels.")

        originals = {
            CONFIG.circuit_info_path: ci,
            CONFIG.test_path: dt,
            CONFIG.overview_path: do,
        }
        candidates = {}
        counts = {}

        for path, original in originals.items():
            candidate = original.copy(deep=True)
            target = (
                mask if path == CONFIG.circuit_info_path
                else require_one_event(
                    candidate, year, gp_slug, dataset_name=path.name
                )
            )
            if "CircuitLayoutSlug" not in candidate:
                candidate["CircuitLayoutSlug"] = pd.Series(
                    pd.NA, index=candidate.index, dtype="object"
                )

            current = candidate.loc[target, "CircuitLayoutSlug"]
            for value in current:
                if not missing(value) and str(value) != layout_slug:
                    raise ValueError("Conflicterende layout in " + path.name)

            changed = sum(missing(value) for value in current)
            candidate["CircuitLayoutSlug"] = candidate[
                "CircuitLayoutSlug"
            ].astype("object")
            candidate.loc[target, "CircuitLayoutSlug"] = layout_slug

            counts[path.name] = changed
            if changed or "CircuitLayoutSlug" not in original:
                candidates[path] = candidate

        canonical = _canonical_operational_validators()
        validators = {}

        for path, expected in candidates.items():
            base_validator = canonical.get(path)

            def validate(frame, expected=expected, base=base_validator):
                if base is not None:
                    result = base(frame)
                    if result is False:
                        raise ValueError("Canonical datasetvalidatie faalt.")
                equivalent(frame, expected)
                return True

            validators[path] = validate
            validate(expected)

        if commit and candidates:
            safe_commit_many(
                candidates,
                validators=validators,
                backup_dir=CONFIG.backup_dir,
            )

        # Na commit opnieuw resolutie controleren, binnen de transactie.
        if commit:
            stored_ci = read_csv_checked(CONFIG.circuit_info_path)
            _, _, stored_layout = resolve_event_circuit(year, gp_slug, stored_ci)
            if stored_layout != layout_slug:
                raise RuntimeError("Layoutresolutie na opslag wijkt af.")

        return {
            "ok": True,
            "committed": bool(commit and candidates),
            "layout": layout_slug,
            "changed_cells": counts,
            "frames": {
                str(path): (
                    read_csv_checked(path) if commit else frame
                )
                for path, frame in {
                    **originals, **candidates
                }.items()
            },
        }
