"""Canonical event-specific circuit selection.
EVENT_CIRCUIT_SELECTION_V1
"""

def resolve_event_circuit(year, gp_slug, circuit_info):
    event_slug = str(gp_slug).strip()
    reference_slug = event_slug
    layout_slug = None

    # Event identity remains bahrain; physical venue is Sepang.
    if (int(year), event_slug) == (2026, "bahrain"):
        reference_slug = "malaysia"
        layout_slug = "sepang_gp"

    match = circuit_info.loc[
        circuit_info["gp_slug"].astype(str).eq(reference_slug)
    ]
    if len(match) != 1:
        raise ValueError(
            f"Circuit reference {reference_slug}: "
            f"expected one row, found {len(match)}."
        )

    row = match.iloc[0].copy()
    if layout_slug is None:
        explicit_layout = row.get("CircuitLayoutSlug")
        if isinstance(explicit_layout, str) and explicit_layout.strip():
            layout_slug = explicit_layout.strip()
    override = _load_event_reference(int(year), event_slug)
    if override is not None:
        if override["reference_slug"] != reference_slug:
            raise ValueError("Event reference conflicts with physical circuit mapping.")
        for key, value in override["values"].items():
            row[key] = value
        row["Turns/Length_km"] = float(row["Turns"]) / float(row["Length_km"])
        layout_slug = str(row["CircuitLayoutSlug"])
    return row, reference_slug, layout_slug


# Event references are registered once, before intake. Existing references are immutable.
_REFERENCE_COLUMNS = ("Year", "gp_slug", "PayloadJSON", "ReferenceHash", "CreatedAtUTC")
_VALUE_FIELDS = {"CircuitLayoutSlug", "Length_km", "Turns", "RecordLap_sec", "avg_speed_kmh"}
_EVIDENCE_FIELDS = {"LayoutSource", "RecordLapSource", "SpeedSource", "SpeedPolicy",
                    "ReviewStatus", "ReviewedAtUTC", "AnnouncedRaceStartUTC"}


def _validate_payload(payload):
    import math
    import re
    import pandas as pd
    from urllib.parse import urlparse
    if set(payload) != {"version", "reference_slug", "values", "evidence"}:
        raise ValueError("Invalid event reference payload fields.")
    if payload["version"] != "event_circuit_reference_v1":
        raise ValueError("Unknown event reference version.")
    if not isinstance(payload["reference_slug"], str) or not payload["reference_slug"].strip():
        raise ValueError("Circuit reference slug missing.")
    values, evidence = payload["values"], payload["evidence"]
    if set(values) != _VALUE_FIELDS or set(evidence) != _EVIDENCE_FIELDS:
        raise ValueError("Complete circuit values and review evidence are required.")
    if not re.fullmatch(r"[a-z0-9_]+", str(values["CircuitLayoutSlug"])):
        raise ValueError("Invalid internal layout identifier.")
    for key in _VALUE_FIELDS - {"CircuitLayoutSlug"}:
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid circuit numeric value: " + key)
    if int(values["Turns"]) != values["Turns"]:
        raise ValueError("Turns must be an integer.")
    if evidence["ReviewStatus"] != "approved":
        raise ValueError("Reference values require explicit reviewed evidence.")
    if evidence["SpeedPolicy"] not in {"record_speed_proxy_v1", "documented_reference_speed_v1"}:
        raise ValueError("Speed policy must be explicit.")
    if evidence["SpeedPolicy"] == "record_speed_proxy_v1":
        expected = round(values["Length_km"] * 3600 / values["RecordLap_sec"], 3)
        if abs(values["avg_speed_kmh"] - expected) > 0.0005:
            raise ValueError("Record-speed proxy does not match its declared formula.")
    for key in ("LayoutSource", "RecordLapSource", "SpeedSource"):
        parsed = urlparse(str(evidence[key]))
        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError("Evidence source must be an HTTPS URL: " + key)
    reviewed = pd.to_datetime(evidence["ReviewedAtUTC"], utc=True, errors="raise")
    start = pd.to_datetime(evidence["AnnouncedRaceStartUTC"], utc=True, errors="raise")
    if pd.isna(reviewed) or pd.isna(start) or not reviewed < start:
        raise ValueError("Invalid evidence chronology.")


def _validated_reference_records(frame):
    import hashlib
    import json
    import pandas as pd
    if frame.empty:
        return {}
    if set(frame.columns) != set(_REFERENCE_COLUMNS):
        raise ValueError("Event circuit reference schema mismatch.")
    records = {}
    for row in frame.to_dict("records"):
        year = float(row["Year"])
        if not year.is_integer():
            raise ValueError("Invalid event year.")
        key = (int(year), str(row["gp_slug"]).strip())
        if not key[1] or key in records:
            raise ValueError("Duplicate or empty event circuit reference identity.")
        payload = json.loads(row["PayloadJSON"])
        _validate_payload(payload)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if _reference_hash(key[0], key[1], canonical, row["CreatedAtUTC"]) != row["ReferenceHash"]:
            raise ValueError("Event circuit reference hash mismatch.")
        created = pd.to_datetime(row["CreatedAtUTC"], utc=True, errors="raise")
        reviewed = pd.to_datetime(payload["evidence"]["ReviewedAtUTC"], utc=True)
        start = pd.to_datetime(payload["evidence"]["AnnouncedRaceStartUTC"], utc=True)
        if pd.isna(created) or not reviewed <= created < start or start.year != key[0]:
            raise ValueError("Invalid event circuit reference creation chronology.")
        records[key] = payload
    return records


def _load_event_reference(year, gp_slug):
    from .config import CONFIG
    from .io import read_csv_checked
    frame = read_csv_checked(CONFIG.event_circuit_references_path, required=False)
    return _validated_reference_records(frame).get((int(year), str(gp_slug).strip()))


def register_event_circuit_reference(year, gp_slug, *, values, evidence):
    """Register reviewed structural inputs once, before event intake; never rewrite history."""
    import hashlib
    import json
    import pandas as pd
    from .config import CONFIG
    from .io import read_csv_checked, safe_commit_many
    from .storage_transactions import transaction
    from .lifecycle import get_event_state

    year = int(year)
    gp_slug = str(gp_slug).strip()
    with transaction():
        master = read_csv_checked(CONFIG.circuit_info_path)
        _, reference_slug, _ = resolve_event_circuit(year, gp_slug, master)
        payload = {"version": "event_circuit_reference_v1", "reference_slug": reference_slug,
                   "values": dict(values), "evidence": dict(evidence)}
        _validate_payload(payload)
        frame = read_csv_checked(CONFIG.event_circuit_references_path, required=False)
        existing = _validated_reference_records(frame)
        if (year, gp_slug) in existing:
            if existing[(year, gp_slug)] != payload:
                raise ValueError("Existing event circuit reference is immutable.")
            return {"ok": True, "created": False, "event": f"{year}_{gp_slug}"}
        state = get_event_state(year, gp_slug)
        if state.test_count or state.overview_count:
            raise ValueError("Register the circuit reference before creating event rows.")
        # Orphan ledger/log entries must also prevent new reference registration.
        for path in (CONFIG.prediction_ledger_path, CONFIG.model_log_path, CONFIG.registry_path):
            data = read_csv_checked(path, required=False)
            if data.empty:
                continue
            event_id = f"{year}_{gp_slug}"
            matches = pd.Series(False, index=data.index)
            supported = False
            if {"Year", "gp_slug"}.issubset(data.columns):
                matches |= pd.to_numeric(data["Year"], errors="coerce").eq(year) & data["gp_slug"].astype(str).str.strip().eq(gp_slug)
                supported = True
            for key in ("race_event_id", "RaceEventID"):
                if key in data:
                    matches |= data[key].astype(str).str.strip().eq(event_id)
                    supported = True
            if not supported or matches.any():
                raise ValueError("Existing or unresolvable event evidence blocks registration: " + path.name)
        now = pd.Timestamp.now(tz="UTC")
        start = pd.to_datetime(evidence["AnnouncedRaceStartUTC"], utc=True)
        reviewed = pd.to_datetime(evidence["ReviewedAtUTC"], utc=True)
        if reviewed > now or start <= now or start.year != year:
            raise ValueError("Registration must use available evidence before the announced event start.")
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        row = {"Year": year, "gp_slug": gp_slug, "PayloadJSON": serialized,
               "ReferenceHash": _reference_hash(year, gp_slug, serialized, now.isoformat()),
               "CreatedAtUTC": now.isoformat()}
        if frame.empty:
            frame = pd.DataFrame(columns=_REFERENCE_COLUMNS)
        candidate = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        def validate(df):
            _validated_reference_records(df)
            return True
        safe_commit_many({CONFIG.event_circuit_references_path: candidate},
                         validators={CONFIG.event_circuit_references_path: validate},
                         backup_dir=CONFIG.backup_dir)
        stored = read_csv_checked(CONFIG.event_circuit_references_path)
        if _validated_reference_records(stored).get((year, gp_slug)) != payload:
            raise RuntimeError("Event reference readback mismatch.")
        return {"ok": True, "created": True, "event": f"{year}_{gp_slug}",
                "ReferenceHash": row["ReferenceHash"]}


def _reference_hash(year, gp_slug, payload_text, created_at):
    import hashlib
    import json
    material = json.dumps({"Year": int(year), "gp_slug": str(gp_slug),
                           "PayloadJSON": payload_text, "CreatedAtUTC": str(created_at)},
                          sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(material.encode()).hexdigest()

# Include readback in the existing canonical all-or-nothing service transaction.
from .storage_transactions import transactional as _transactional
register_event_circuit_reference = _transactional(register_event_circuit_reference)
