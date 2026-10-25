"""Automatic qualifying evidence intake; canonical writers own proof and freeze."""
import hashlib
import json
import re
import time
import unicodedata
import pandas as pd
from urllib.request import urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from .storage_transactions import transactional

BASE = 'https://api.openf1.org/v1'

def _utc(value, field):
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError()
        return stamp.tz_convert('UTC')
    except (ValueError, TypeError):
        raise ValueError(f'OpenF1: {field} ontbreekt of heeft geen geldige tijdzone.') from None

def _normal(value):
    value = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode()
    return re.sub('[^a-z0-9]', '', value.lower())

def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def _get(endpoint, **params):
    time.sleep(0.35)  # Respect the public API's per-second limit.
    try:
        with urlopen(f'{BASE}/{endpoint}?{urlencode(params)}', timeout=15) as response:
            rows = json.loads(response.read())
    except HTTPError as exc:
        raise ValueError(f'OpenF1 {endpoint} niet beschikbaar (HTTP {exc.code}). Probeer later opnieuw.') from None
    except (URLError, TimeoutError, OSError):
        raise ValueError(f'OpenF1 {endpoint}: netwerkfout. Probeer later opnieuw.') from None
    except ValueError:
        raise ValueError(f'OpenF1 {endpoint}: ongeldige JSON.') from None
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f'OpenF1 {endpoint}: benodigde sessie-/eindbewijzen ontbreken.')
    return rows

def _resolve_proof(year, gp_slug, announced, circuit, reference_slug, *, now, get=_get):
    """No writes. Match exact race identity, published qualifying results and track finish evidence."""
    races = get('sessions', year=int(year), session_name='Race')
    matches = []
    for row in races:
        if row.get('session_name') != 'Race' or row.get('session_type') != 'Race':
            continue
        if row.get('is_cancelled') is not False or int(row.get('year', 0)) != int(year):
            continue
        if abs((_utc(row.get('date_start'), 'race date_start') - announced).total_seconds()) <= 60:
            matches.append(row)
    if len(matches) != 1:
        raise ValueError('OpenF1: geen unieke hoofdrace binnen 60 seconden van de opgeslagen starttijd; controleer eventplanning.')
    race = matches[0]
    # Time/meeting plus physical venue. Country-name match is disambiguated by unique race time.
    expected = {_normal(circuit['Circuit']), _normal(reference_slug)}
    received = {_normal(race.get(key, '')) for key in ('circuit_short_name', 'country_name', 'location')}
    aliases = {'unitedstates': {'austin', 'circuitoftheamericas'},
               'greatbritain': {'silverstone', 'unitedkingdom'},
               'malaysia': {'sepang'}}
    for value in list(expected):
        expected.update(aliases.get(value, set()))
    if not ((expected - {''}) & (received - {''})):
        raise ValueError('OpenF1: fysieke circuitidentiteit komt niet overeen met de centrale eventreferentie.')
    if race.get('meeting_key') is None or race.get('circuit_key') is None:
        raise ValueError('OpenF1: meeting_key/circuit_key ontbreekt.')
    qualifiers = get('sessions', meeting_key=race['meeting_key'], session_name='Qualifying')
    qualifiers = [q for q in qualifiers if
        q.get('session_name') == q.get('session_type') == 'Qualifying'
        and q.get('is_cancelled') is False
        and int(q.get('year', 0)) == int(year)
        and q.get('meeting_key') == race['meeting_key']
        and q.get('circuit_key') == race['circuit_key']]
    if len(qualifiers) != 1:
        raise ValueError('OpenF1: geen unieke hoofdracekwalificatie; sprintkwalificatie wordt niet gebruikt.')
    qualifying = qualifiers[0]
    start = _utc(qualifying.get('date_start'), 'qualifying date_start')
    recorded_end = _utc(qualifying.get('date_end'), 'qualifying date_end')
    if not start < recorded_end <= now < announced:
        raise ValueError('OpenF1: kwalificatie nog niet afgelopen of tijdvolgorde ongeldig.')
    key = qualifying.get('session_key')
    if key is None:
        raise ValueError('OpenF1: qualifying session_key ontbreekt.')
    results = get('session_result', session_key=key)
    if any(row.get('session_key') != key or row.get('meeting_key') != race['meeting_key'] for row in results):
        raise ValueError('OpenF1: resultaat hoort bij een andere sessie.')
    # Results are published after the session; don't infer completion from the planned date_end alone.
    if not any(isinstance(row.get('duration'), list) and len(row['duration']) == 3
               and isinstance(row['duration'][2], (int, float))
               and not isinstance(row['duration'][2], bool)
               and float(row['duration'][2]) > 0 for row in results):
        raise ValueError('OpenF1: gepubliceerd Q3-resultaat ontbreekt; geen voltooiing afleiden uit alleen de planning.')
    controls = get('race_control', session_key=key)
    finishes = []
    for row in controls:
        if row.get('session_key') != key or row.get('meeting_key') != race['meeting_key']:
            raise ValueError('OpenF1: race-controlbericht hoort bij een andere sessie.')
        if str(row.get('scope', '')).upper() != 'TRACK':
            continue
        message = str(row.get('message', '')).upper()
        flag = str(row.get('flag', '')).upper()
        if flag == 'CHEQUERED' or 'SESSION FINISHED' in message or 'SESSION ENDED' in message:
            stamp = _utc(row.get('date'), 'finishbericht date')
            if not start <= stamp <= now or stamp >= announced:
                raise ValueError('OpenF1: eindbericht heeft ongeldige tijdvolgorde.')
            finishes.append((stamp, row))
    if not finishes:
        raise ValueError('OpenF1: einde-/finishbericht ontbreekt; freeze blijft geblokkeerd.')
    completion, finish = max(finishes, key=lambda item: item[0])
    # Reject an early-segment finish being mistaken for overall completion.
    if completion < recorded_end - pd.Timedelta(minutes=10):
        raise ValueError('OpenF1: eindbericht ligt ruim vóór geregistreerd sessie-einde; handmatige bronreview nodig.')
    proof = {'method': 'openf1_qualifying_completion_v1',
             'race_event_id': f'{int(year)}_{gp_slug}', 'meeting_key': race['meeting_key'],
             'session_key': key, 'circuit_key': race['circuit_key'],
             'session_url': f'{BASE}/sessions?session_key={key}',
             'results_url': f'{BASE}/session_result?session_key={key}',
             'race_control_url': f'{BASE}/race_control?session_key={key}',
             'retrieved_at_utc': pd.Timestamp.now(tz='UTC').isoformat(),
             'session_record': qualifying, 'race_record': race,
             'finish_record': finish, 'results_sha256': _digest(results),
             'race_control_sha256': _digest(controls)}
    return completion, json.dumps(proof, sort_keys=True, separators=(',', ':'), allow_nan=False)

@transactional
def freeze_with_automatic_qualifying(year, gp_slug):
    from . import readiness as r
    from .event_circuits import resolve_event_circuit
    from .config import CONFIG
    from .io import read_csv_checked
    dt, do, mt, mo, rt, ro = r._forward_rows(year, gp_slug)
    if r._present(rt.get('FeatureSnapshotTimestampUTC')) or r._present(ro.get('FeatureSnapshotTimestampUTC')):
        return r.freeze_feature_snapshot(year, gp_slug)
    r._assert_inputs_not_frozen(year, gp_slug)
    if str(rt.get('RaceStatus')) != 'race_context_plausibility_ready' or str(ro.get('RaceStatus')) != 'race_context_plausibility_ready':
        raise ValueError('Bevestig eerst racecontext voordat je de definitieve inputs bevriest.')
    if str(rt.get('PredictionTimingPolicyVersion')) != 'actual_race_start_v1':
        raise ValueError('Automatische intake vereist actual_race_start_v1; historische records blijven intact.')
    for validator in (r.validate_qualifying, r.validate_race_weather, r.validate_event_timing):
        gate = validator(rt, ro)
        if not gate.ok:
            raise ValueError(gate.detail)
    now = pd.Timestamp.now(tz='UTC')
    announced = _utc(rt.get('AnnouncedRaceStartUTC'), 'AnnouncedRaceStartUTC')
    if now >= announced:
        raise ValueError('Aangekondigde racestart verstreken; geen pre-race intake mogelijk.')
    # Preserve an already captured valid proof rather than rewriting its intake timestamp.
    if r._present(rt.get('QualifyingCompletionSource')) and r._present(ro.get('QualifyingCompletionSource')):
        return r.freeze_feature_snapshot(year, gp_slug)
    ci = read_csv_checked(CONFIG.circuit_info_path)
    circuit, reference, _ = resolve_event_circuit(year, gp_slug, ci)
    completion, source = _resolve_proof(year, gp_slug, announced, circuit, reference, now=now)
    r.save_pre_race_input_proof(year, gp_slug,
        qualifying_completed_at_utc=completion.isoformat(), qualifying_source=source)
    return r.freeze_feature_snapshot(year, gp_slug)
