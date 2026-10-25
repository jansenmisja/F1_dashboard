import datetime as dt
import json
import urllib.error
import urllib.request

def aware_time(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Tijd ontbreekt.')
    parsed = dt.datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError('Tijdzone ontbreekt.')
    return parsed.astimezone(dt.timezone.utc)

def fetch_session_context(mapping, *, opener_factory=None):
    source_url = 'https://api.openf1.org/v1/sessions?session_key=' + str(mapping['expected']['session_key'])
    state = {'last_good': None, 'error': None}

    class NoRedirect(urllib.request.HTTPRedirectHandler):

        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None
    opener = (opener_factory or urllib.request.build_opener)(NoRedirect())
    request = urllib.request.Request(source_url, headers={'Accept': 'application/json'}, method='GET')
    try:
        with opener.open(request, timeout=15) as response:
            raw = response.read(1000001)
        if len(raw) > 1000000:
            raise ValueError('Antwoord overschrijdt de toegestane omvang.')
        sessions = json.loads(raw)
        if not isinstance(sessions, list) or len(sessions) != 1:
            raise ValueError('Geen eenduidige sessie ontvangen.')
        session = sessions[0]
        if not isinstance(session, dict):
            raise ValueError('Ongeldig sessieantwoord.')
        expected = dict(mapping['expected'])
        for field, value in expected.items():
            actual = session.get(field)
            if type(actual) is not type(value) or actual != value:
                raise ValueError(f'Sessie-identiteit wijkt af bij {field}.')
        if type(session.get('is_cancelled')) is not bool:
            raise ValueError('Annuleringsveld is niet eenduidig.')
        start = aware_time(session.get('date_start'))
        end = aware_time(session.get('date_end'))
        if end <= start:
            raise ValueError('Sessie-einde ligt niet na de start.')
        state['last_good'] = {'scheduled_start_utc': start.isoformat(), 'scheduled_end_utc': end.isoformat(), 'is_cancelled': session['is_cancelled'], 'retrieved_at_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            detail = ''
            try:
                body = json.loads(exc.read(8192))
                if isinstance(body, dict):
                    detail = str(body.get('detail', '')).lower()
            except Exception:
                pass
            live_restriction = 'live' in detail and 'restricted' in detail
            state['error'] = 'Gratis toegang tijdelijk geblokkeerd door OpenF1: tijdens de live sessie is authenticatie vereist, ook voor historische sessies. Probeer later handmatig opnieuw.' if live_restriction else 'OpenF1 vereist authenticatie (HTTP 401). Gratis ophalen is nu niet beschikbaar.'
        elif exc.code == 429:
            state['error'] = 'OpenF1 meldt te veel aanvragen (HTTP 429). Wacht en probeer later handmatig opnieuw.'
        elif exc.code == 403:
            state['error'] = 'OpenF1 weigert toegang (HTTP 403).'
        else:
            state['error'] = f'OpenF1-aanvraag mislukt (HTTP {exc.code}).'
    except (ValueError, TypeError):
        state['error'] = 'Het bronantwoord voldoet niet aan de contextvalidatie. Er zijn geen nieuwe gegevens opgenomen.'
    except Exception:
        state['error'] = 'Verbinding met OpenF1 mislukt of duurde te lang. De oorzaak is niet vastgesteld.'
    if state['last_good'] is not None:
        state['last_good']['provider_identity'] = dict(mapping['expected'])
    return {'status': 'success' if state['error'] is None else 'error', 'observation': state['last_good'], 'error': state['error'], 'source_url': source_url}

def fetch_singapore_session_context(*, opener_factory=None):
    from f1_external_context_mappings import MAPPINGS
    return fetch_session_context(MAPPINGS['2026_singapore'], opener_factory=opener_factory)
