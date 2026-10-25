
"""Fill missing master layout identifiers without rewriting event history."""

def fill_missing_master_layouts(*, commit=False):
    import re
    import math
    import pandas as pd
    from pandas.testing import assert_frame_equal

    from .config import CONFIG
    from .io import read_csv_checked, safe_commit_many
    from .storage_transactions import transaction
    from .event_circuits import (
        resolve_event_circuit,
        _validated_reference_records,
    )

    def present(value):
        return not pd.isna(value) and bool(str(value).strip())

    def same_geometry(values, row):
        try:
            return (
                math.isclose(
                    float(values["Length_km"]),
                    float(row["Length_km"]),
                    rel_tol=0, abs_tol=0.0005,
                )
                and float(values["Turns"]) == float(row["Turns"])
            )
        except (KeyError, TypeError, ValueError):
            return False

    def compare_frames(actual, expected):
        if list(actual.columns) != list(expected.columns):
            raise ValueError("Onverwachte kolomwijziging.")
        a = actual.reset_index(drop=True).copy()
        b = expected.reset_index(drop=True).copy()
        for col in a.columns:
            x = a[col].astype("string").fillna("")
            y = b[col].astype("string").fillna("")
            nx = pd.to_numeric(x.replace("", pd.NA), errors="coerce")
            ny = pd.to_numeric(y.replace("", pd.NA), errors="coerce")
            if (((x == "") | nx.notna()).all()
                    and ((y == "") | ny.notna()).all()):
                a[col] = nx.astype("Float64")
                b[col] = ny.astype("Float64")
            else:
                a[col], b[col] = x, y
        assert_frame_equal(
            a, b, check_dtype=False, check_exact=False,
            rtol=0, atol=1e-12,
        )

    with transaction():
        master = read_csv_checked(CONFIG.circuit_info_path)
        required = {"gp_slug", "Circuit", "Length_km", "Turns"}
        if not required.issubset(master.columns):
            raise ValueError("Verplichte circuitkolommen ontbreken.")

        keys = master["gp_slug"].astype("string").str.strip()
        if keys.isna().any() or keys.eq("").any() or keys.duplicated().any():
            raise ValueError("Lege of dubbele circuitidentiteit.")

        references = read_csv_checked(
            CONFIG.event_circuit_references_path, required=False
        )
        records = _validated_reference_records(references)

        # Bestaande operationele labels zijn aanvullende referenties.
        historical_rows = []
        for path in (CONFIG.test_path, CONFIG.overview_path):
            frame = read_csv_checked(path)
            if {
                "gp_slug", "CircuitLayoutSlug", "Length_km", "Turns"
            }.issubset(frame.columns):
                historical_rows.extend(frame.to_dict("records"))

        candidate = master.copy(deep=True)
        if "CircuitLayoutSlug" not in candidate:
            candidate["CircuitLayoutSlug"] = pd.Series(
                pd.NA, index=candidate.index, dtype="object"
            )
        candidate["CircuitLayoutSlug"] = candidate[
            "CircuitLayoutSlug"
        ].astype("object")

        decisions = []
        conflicts = []

        for index, row in master.iterrows():
            slug = str(row["gp_slug"]).strip()
            old = candidate.at[index, "CircuitLayoutSlug"]

            if present(old):
                if not re.fullmatch(r"[a-z0-9_]+", str(old)):
                    conflicts.append(f"{slug}: ongeldige bestaande slug {old!r}")
                decisions.append({
                    "gp_slug": slug,
                    "CircuitLayoutSlug": old,
                    "actie": "behouden",
                })
                continue

            options = set()

            # Geregistreerde fysieke circuitreferenties gaan op reference_slug.
            for payload in records.values():
                if (
                    payload["reference_slug"] == slug
                    and same_geometry(payload["values"], row)
                ):
                    options.add(str(payload["values"]["CircuitLayoutSlug"]))

            # Raadpleeg ook expliciete eventmapping, waaronder Sepang 2026.
            if slug == "malaysia":
                resolved, reference, layout = resolve_event_circuit(
                    2026, "bahrain", master
                )
                if reference == slug and layout and same_geometry(resolved, row):
                    options.add(str(layout))

            for existing in historical_rows:
                if (
                    str(existing["gp_slug"]).strip() == slug
                    and present(existing["CircuitLayoutSlug"])
                    and same_geometry(existing, row)
                ):
                    options.add(str(existing["CircuitLayoutSlug"]).strip())

            if len(options) > 1:
                conflicts.append(f"{slug}: meerdere bestaande labels {sorted(options)}")
                continue

            layout = next(iter(options)) if options else f"{slug}_gp"
            if not re.fullmatch(r"[a-z0-9_]+", layout):
                conflicts.append(f"{slug}: ongeldige layoutslug {layout!r}")
                continue

            candidate.at[index, "CircuitLayoutSlug"] = layout
            decisions.append({
                "gp_slug": slug,
                "CircuitLayoutSlug": layout,
                "actie": (
                    "bestaande referentie hergebruikt"
                    if options else "interne code toegekend"
                ),
            })

        if conflicts:
            raise ValueError(
                "Geen opslag uitgevoerd. Eerst conflicten oplossen:\n"
                + "\n".join(conflicts)
            )

        def validate(frame):
            compare_frames(frame, candidate)
            if frame["CircuitLayoutSlug"].isna().any():
                raise ValueError("Layoutslug ontbreekt na aanvullen.")
            if not frame["CircuitLayoutSlug"].astype(
                "string"
            ).str.fullmatch(r"[a-z0-9_]+").all():
                raise ValueError("Ongeldige layoutslug.")
            return True

        validate(candidate)
        changes = sum(
            item["actie"] != "behouden" for item in decisions
        )

        if commit and changes:
            safe_commit_many(
                {CONFIG.circuit_info_path: candidate},
                validators={CONFIG.circuit_info_path: validate},
                backup_dir=CONFIG.backup_dir,
            )

        final = (
            read_csv_checked(CONFIG.circuit_info_path)
            if commit else candidate
        )

        return {
            "ok": True,
            "committed": bool(commit and changes),
            "newly_filled": changes,
            "total_circuits": len(final),
            "missing_layouts": int(final["CircuitLayoutSlug"].isna().sum()),
            "decisions": pd.DataFrame(decisions),
            "frame": final,
        }
