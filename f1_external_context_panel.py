"""Externe context op basis van de geselecteerde dashboardrij."""
DOCUMENT_TEMPLATE = '<!doctype html>\n<html lang="nl">\n<head>\n<meta charset="utf-8">\n<meta http-equiv="Content-Security-Policy"\ncontent="default-src \'none\'; script-src \'unsafe-inline\';\nstyle-src \'unsafe-inline\'; connect-src https://api.openf1.org;\nbase-uri \'none\'; form-action \'none\'">\n<style>\n*{box-sizing:border-box}\nbody{margin:0;padding:20px;font:14px system-ui;color:#172235;\nbackground:#f5f7fb}\n.panel{max-width:1050px;margin:auto}\n.top{display:flex;justify-content:space-between;align-items:center;\ngap:16px;flex-wrap:wrap}\nh1{font-size:25px;margin:4px 0}\n.subtitle,.muted{color:#627086}\nbutton{background:#175cd3;color:white;border:0;border-radius:8px;\npadding:11px 16px;font-weight:600;cursor:pointer}\nbutton:disabled{opacity:.55;cursor:wait}\n.cards{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));\ngap:12px;margin:20px 0}\n.card{background:white;border:1px solid #dde4ed;border-radius:12px;\npadding:16px}\n.label{color:#627086;font-size:12px;margin-bottom:9px}\n.value{font-size:21px;font-weight:700;line-height:1.3}\n.small{font-size:12px;color:#627086;margin-top:7px}\n.notice{padding:12px 14px;border-radius:8px;background:#e9eef7;\nmargin:12px 0}\n.ok{background:#e8f5ed;color:#17653b}\n.warn{background:#fff2d8;color:#835000}\n.error{background:#feecec;color:#9c2525}\ndetails{background:white;border:1px solid #dde4ed;border-radius:10px;\npadding:13px;margin-top:18px}\nsummary{cursor:pointer;font-weight:600}\ntable{width:100%;border-collapse:collapse;margin-top:12px}\ntd{padding:8px;border-bottom:1px solid #edf0f4;vertical-align:top;\noverflow-wrap:anywhere}\ntd:first-child{width:40%;color:#627086}\ntextarea{width:100%;height:200px;margin-top:10px;font:12px monospace}\na{color:#175cd3}\n@media(max-width:700px){.cards{grid-template-columns:repeat(2,1fr)}}\n@media(max-width:400px){.cards{grid-template-columns:1fr}}\n</style>\n</head>\n<body>\n<div class="panel">\n<div class="top">\n <div>\n  <div class="subtitle">Externe racecontext · Singapore 2026</div>\n  <h1>Singapore — geplande racesessie</h1>\n  <div class="muted">OpenF1 · Marina Bay · tijden in Nederland</div>\n </div>\n <button id="refresh">Haal sessiecontext op</button>\n</div>\n\n<div class="cards">\n <div class="card">\n  <div class="label">Geplande start volgens OpenF1</div>\n  <div class="value" id="start">Nog niet opgehaald</div>\n  <div class="small" id="date">—</div>\n </div>\n <div class="card">\n  <div class="label">Vergelijking met projectplanning</div>\n  <div class="value" id="comparison">Nog niet vergeleken</div>\n  <div class="small" id="difference">Vaste projectsnapshot</div>\n </div>\n <div class="card">\n  <div class="label">Annulering volgens OpenF1</div>\n  <div class="value" id="cancelled">Onbekend</div>\n  <div class="small">Externe bronmelding</div>\n </div>\n <div class="card">\n  <div class="label">Laatste succesvolle ophaalactie</div>\n  <div class="value" id="retrieved">Nog niet opgehaald</div>\n  <div class="small" id="age">Handmatig verversen</div>\n </div>\n</div>\n\n<div id="notice" class="notice" role="status" aria-live="polite">\nKlik op “Haal sessiecontext op” om OpenF1 te raadplegen.\n</div>\n<p class="muted">\nDit paneel vergelijkt de geplande sessiestart met een vaste projectkopie.\nHet bevestigt niet de daadwerkelijke racestart of de exacte circuitlayout. De projectwaarden worden opnieuw ingelezen bij een dashboard-rerun.\n</p>\n<p class="muted" id="snapshot"></p>\n\n<details>\n <summary>Bron en controlegegevens</summary>\n <table><tbody id="technical"></tbody></table>\n <p><a id="source" target="_blank" rel="noopener noreferrer">\n OpenF1-bron openen</a></p>\n <p class="muted">\n De bronwijzigingstijd is onbekend. Recent opgehaald betekent niet\n automatisch dat OpenF1 zijn gegevens recent heeft gewijzigd.\n </p>\n <details>\n  <summary>JSON-rapport voor controle</summary>\n  <textarea id="report" readonly></textarea>\n </details>\n</details>\n</div>\n\n\n<script id="f1-context-fetch-errors-v1">\n(() => {\n  const sourceURL =\n    "https://api.openf1.org/v1/sessions?session_key=11388";\n  const originalFetch = window.fetch.bind(window);\n\n  window.fetch = async function(input, options) {\n    const requestURL =\n      typeof input === "string" ? input :\n      input instanceof URL ? input.href :\n      input && input.url;\n\n    // Andere aanvragen houden hun bestaande gedrag.\n    if (requestURL !== sourceURL) {\n      return originalFetch(input, options);\n    }\n\n    let response;\n    try {\n      response = await originalFetch(input, options);\n    } catch (error) {\n      if (error && error.name === "AbortError") {\n        throw new Error(\n          "Het ophalen duurde te lang. Probeer later handmatig opnieuw."\n        );\n      }\n      if (error instanceof TypeError) {\n        throw new Error(\n          "Geen leesbaar antwoord ontvangen. De oorzaak is onbekend: " +\n          "een netwerk-, browser- of bronbeperking kan dit veroorzaken. " +\n          "Open ter controle rechtstreeks: " + sourceURL +\n          " . Toont die pagina een authenticatiemelding, " +\n          "dan is toegang door OpenF1 beperkt."\n        );\n      }\n      throw error;\n    }\n\n    if (response.status === 401) {\n      throw new Error(\n        "OpenF1 vereist authenticatie (HTTP 401). " +\n        "Deze aanvraag levert geen nieuwe context op. " +\n        "Controleer de bronpagina voor de reden en probeer later " +\n        "handmatig opnieuw. Dit paneel gebruikt geen API-key."\n      );\n    }\n    if (response.status === 403) {\n      throw new Error(\n        "OpenF1 weigert toegang (HTTP 403). " +\n        "Er is geen nieuwe context opgehaald."\n      );\n    }\n    if (response.status === 429) {\n      throw new Error(\n        "OpenF1 meldt te veel aanvragen (HTTP 429). " +\n        "Wacht en probeer later handmatig opnieuw."\n      );\n    }\n\n    return response;\n  };\n})();\n</script>\n<script>\n"use strict";\nconst snapshot=__PROJECT_SNAPSHOT_JSON__;\nconst url="https://api.openf1.org/v1/sessions?session_key=11388";\nconst expected={\n session_key:11388,meeting_key:1296,circuit_key:61,year:2026,\n country_name:"Singapore",circuit_short_name:"Singapore",\n location:"Marina Bay",session_name:"Race",session_type:"Race"\n};\nlet latest=null,lastError=null,lastAttempt=null,busy=false;\nconst byId=id=>document.getElementById(id);\nconst timeFormat=new Intl.DateTimeFormat("nl-NL",{\n timeZone:"Europe/Amsterdam",hour:"2-digit",minute:"2-digit"\n});\nconst dateFormat=new Intl.DateTimeFormat("nl-NL",{\n timeZone:"Europe/Amsterdam",day:"numeric",month:"long",year:"numeric"\n});\nfunction utc(value){\n if(typeof value!=="string"||!/(Z|\\+00:00)$/.test(value))\n   throw Error("Ongeldige UTC-tijdstempel.");\n const parsed=Date.parse(value);\n if(!Number.isFinite(parsed))throw Error("Tijdstempel niet leesbaar.");\n return parsed;\n}\nfunction local(value){\n const date=new Date(utc(value));\n return dateFormat.format(date)+" om "+timeFormat.format(date);\n}\nfunction render(){\n byId("snapshot").textContent=\n   "Projectwaarden in deze weergave ingelezen: "+local(snapshot.snapshot_at_utc)+\n   ". Verversen haalt alleen OpenF1 op; een dashboard-rerun leest het project opnieuw.";\n byId("source").href=url;\n\n if(latest){\n  const start=new Date(utc(latest.scheduled_start_utc));\n  const fetched=new Date(utc(latest.retrieved_at_utc));\n  const age=Math.max(0,Math.floor((Date.now()-fetched.getTime())/60000));\n  const delta=latest.start_difference_seconds;\n  byId("start").textContent=timeFormat.format(start);\n  byId("date").textContent=dateFormat.format(start);\n  byId("comparison").textContent=delta===0?"Komt overeen":"Wijkt af";\n  byId("difference").textContent=delta===0\n    ?"Geen verschil in geplande start"\n    :"Verschil: "+delta+" seconden";\n  byId("cancelled").textContent=latest.is_cancelled\n    ?"Geannuleerd":"Niet geannuleerd";\n  byId("retrieved").textContent=timeFormat.format(fetched);\n  byId("age").textContent=\n    dateFormat.format(fetched)+" · "+\n    (age===0?"minder dan 1 minuut geleden":age+" minuten geleden");\n  if(!busy){\n   if(lastError){\n    byId("notice").className="notice error";\n    byId("notice").textContent=\n      "Verversen mislukt: "+lastError+\n      " De vorige succesvolle response blijft zichtbaar.";\n   }else if(delta!==0||latest.is_cancelled||age>=15){\n    byId("notice").className="notice warn";\n    const warnings=[];\n    if(delta!==0)warnings.push("De geplande start wijkt af van de projectkopie.");\n    if(latest.is_cancelled)warnings.push("OpenF1 meldt een annulering.");\n    if(age>=15)warnings.push("Deze response is minstens 15 minuten geleden opgehaald.");\n    byId("notice").textContent=warnings.join(" ");\n   }else{\n    byId("notice").className="notice ok";\n    byId("notice").textContent=\n      "OpenF1 opgehaald. De geplande start komt overeen met de projectkopie.";\n   }\n  }\n }else if(lastError&&!busy){\n  byId("notice").className="notice error";\n  byId("notice").textContent="Ophalen mislukt: "+lastError;\n }\n\n const rows=[\n  ["Event",snapshot.event_key],\n  ["Opgeslagen workflowstatus",snapshot.RaceStatus],["Circuitlayout in project",snapshot.CircuitLayoutSlug],["Sprintweekend in project",snapshot.IsSprintWeekend],\n  ["Geplande start in projectkopie",local(snapshot.start_utc)],\n  ["Projectstart in UTC",snapshot.start_utc],\n  ["Bronwijzigingstijd","Onbekend"]\n ];\n if(latest)rows.push(\n  ["OpenF1-sessiesleutel",latest.session_key],\n  ["Circuit / locatie",latest.circuit+" / "+latest.location],\n  ["OpenF1-start in UTC",latest.scheduled_start_utc],\n  ["OpenF1-ophaaltijd",local(latest.retrieved_at_utc)]\n );\n byId("technical").replaceChildren();\n for(const row of rows){\n  const tr=document.createElement("tr");\n  for(const value of row){\n   const td=document.createElement("td");\n   td.textContent=String(value);tr.appendChild(td);\n  }\n  byId("technical").appendChild(tr);\n }\n byId("report").value=JSON.stringify({\n  request_succeeded:latest!==null&&lastError===null,\n  mode:"sandboxed_browser_context",\n  current_request_error:lastError,\n  previous_success_retained:latest!==null&&lastError!==null,\n  latest_context:latest,\n  adapter_connection_approved:false\n },null,2);\n}\nbyId("refresh").onclick=async()=>{\n if(busy)return;\n const now=performance.now();\n if(lastAttempt!==null&&now-lastAttempt<30000){\n  byId("notice").className="notice";\n  byId("notice").textContent="Wacht 30 seconden tussen ophaalacties.";\n  return;\n }\n busy=true;lastAttempt=now;byId("refresh").disabled=true;\n byId("notice").className="notice";\n byId("notice").textContent="OpenF1 raadplegen…";\n const controller=new AbortController();\n const timer=setTimeout(()=>controller.abort(),25000);\n try{\n  const response=await fetch(url,{\n   mode:"cors",credentials:"omit",redirect:"error",cache:"no-store",\n   referrerPolicy:"no-referrer",signal:controller.signal\n  });\n  if(!response.ok)throw Error("HTTP-status "+response.status);\n  const reader=response.body.getReader(),chunks=[];\n  let size=0;\n  while(true){\n   const part=await reader.read();\n   if(part.done)break;\n   size+=part.value.byteLength;\n   if(size>1000000){\n    await reader.cancel();throw Error("Response groter dan 1 MB.");\n   }\n   chunks.push(part.value);\n  }\n  const bytes=new Uint8Array(size);let offset=0;\n  for(const chunk of chunks){bytes.set(chunk,offset);offset+=chunk.length;}\n  const records=JSON.parse(new TextDecoder().decode(bytes));\n  if(!Array.isArray(records)||records.length!==1)\n    throw Error("Precies één sessie vereist.");\n  const s=records[0];\n  if(!s||typeof s!=="object"||Array.isArray(s))\n    throw Error("Ongeldig sessierecord.");\n  for(const [key,value] of Object.entries(expected)){\n   if(typeof s[key]!==typeof value||s[key]!==value)\n     throw Error("Sessiekoppeling wijkt af: "+key);\n  }\n  if(typeof s.is_cancelled!=="boolean")\n    throw Error("Annuleringsveld niet geldig.");\n  const start=utc(s.date_start),end=utc(s.date_end);\n  if(end<=start)throw Error("Ongeldig sessievenster.");\n  latest={\n   event_key:snapshot.event_key,session_key:s.session_key,\n   circuit:s.circuit_short_name,location:s.location,\n   scheduled_start_utc:s.date_start,scheduled_end_utc:s.date_end,\n   start_difference_seconds:(start-utc(snapshot.start_utc))/1000,\n   is_cancelled:s.is_cancelled,retrieved_at_utc:new Date().toISOString(),\n   source_updated_at_utc:null,source_url:url,\n   layout_verified:false,actual_race_start_verified:false\n  };\n  lastError=null;\n  byId("refresh").textContent="Ververs sessiecontext";\n }catch(error){\n  lastError=error.name==="AbortError"\n    ?"Geen antwoord binnen 25 seconden."\n    :String(error.message||error);\n }finally{\n  clearTimeout(timer);busy=false;\n  byId("refresh").disabled=false;render();\n }\n};\nrender();\nsetInterval(render,60000); // Alleen weergaveleeftijd; geen netwerkpolling.\n</script>\n</body></html>\n'

def render_external_context_panel(*, df_overview, year, gp_slug):
    import datetime as dt
    import json
    import math
    import time
    import urllib.error
    import urllib.request
    import pandas as pd
    import streamlit as st
    if not st.checkbox('Toon externe context voor geselecteerde race', value=False, key='external_singapore_context_enabled'):
        return
    if not {'Year', 'gp_slug'}.issubset(df_overview.columns):
        st.error('Year of gp_slug ontbreekt in de dashboardgegevens.')
        return
    try:
        selected_year = int(year)
        slug = str(gp_slug).strip()
    except (ValueError, TypeError):
        st.error('Eventselectie is ongeldig.')
        return
    selected = df_overview.loc[pd.to_numeric(df_overview['Year'], errors='coerce').eq(selected_year) & df_overview['gp_slug'].astype(str).str.strip().eq(slug)]
    if len(selected) != 1:
        st.error('Het geselecteerde event is niet eenduidig gevonden.')
        return
    row = selected.iloc[0]

    def text(field):
        value = row.get(field)
        if value is None or pd.isna(value):
            return ''
        return str(value).strip()

    def aware_time(value):
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Tijd ontbreekt.')
        parsed = dt.datetime.fromisoformat(value.strip().replace('Z', '+00:00'))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError('Tijdzone ontbreekt.')
        return parsed.astimezone(dt.timezone.utc)
    event_key = f'{selected_year}_{slug}'
    st.subheader('Externe sessiecontext')
    st.caption('Gratis toegang · handmatig ophalen · geen achtergrondpolling. Projectwaarden komen uit de huidige dashboardweergave.')
    st.table([{'Veld': 'Event', 'Waarde': event_key}, {'Veld': 'Circuit', 'Waarde': text('Circuit')}, {'Veld': 'Layout', 'Waarde': text('CircuitLayoutSlug')}, {'Veld': 'Opgeslagen workflowstatus', 'Waarde': text('RaceStatus')}, {'Veld': 'Aangekondigde start UTC', 'Waarde': text('AnnouncedRaceStartUTC') or text('EventDateTimeUTC')}])
    context_findings = _context_quality_findings(row.to_dict())
    invalid_findings = [item for item in context_findings if item.get('Resultaat') == 'Ongeldig']
    different_findings = [item for item in context_findings if item.get('Resultaat') == 'Verschil']
    missing_findings = [item for item in context_findings if item.get('Resultaat') == 'Ontbreekt']
    if invalid_findings or different_findings:
        st.warning(f'Contextcontrole: {len(invalid_findings)} ongeldige waarden en {len(different_findings)} verschillen gevonden. Open de controletabel voor details.')
    else:
        st.info('Binnen de uitgevoerde contextchecks zijn geen ongeldige formaten of verschillen gevonden. Dit is geen bevestiging dat alle projectgegevens compleet zijn.')
    if missing_findings:
        st.caption(f'{len(missing_findings)} ontbrekende waarden binnen deze contextchecks. Ontbrekende historische context kan verwacht zijn; de canonical pipeline bepaalt welke inputs verplicht zijn.')
    st.caption('Deze controle wijzigt geen waarden of workflowstatus. Daadwerkelijke racestart, exacte layout en canonical readiness worden hiermee niet bewezen.')
    with st.expander('Projectgegevens controleren', expanded=False):
        st.caption('Formaat- en consistentiechecks op de projectwaarden uit de huidige dashboardweergave.')
        st.dataframe(pd.DataFrame(context_findings), hide_index=True)
    from f1_external_context_mappings import get_context_mapping
    mapping, mapping_reason = get_context_mapping(event_key, text('Circuit'), text('CircuitLayoutSlug'))
    if mapping is None:
        st.info(mapping_reason)
        return
    planned_text = text('AnnouncedRaceStartUTC') or text('EventDateTimeUTC')
    try:
        planned = aware_time(planned_text)
    except ValueError:
        planned = None
        if planned_text:
            st.warning('Projectstarttijd is ongeldig; ophalen blijft geblokkeerd.')
            return
        st.info('Projectplanning ontbreekt. Sessiecontext kan worden opgehaald; starttijden zijn niet vergelijkbaar.')
    if text('AnnouncedRaceStartUTC') and text('EventDateTimeUTC'):
        try:
            if aware_time(text('EventDateTimeUTC')) != planned:
                st.warning('Projectstarttijden verschillen. De vergelijking gebruikt AnnouncedRaceStartUTC.')
        except ValueError:
            st.warning('EventDateTimeUTC is ongeldig; ophalen geblokkeerd.')
            return
    source_url = mapping['source_url']
    state_key = 'f1_external_context_' + event_key + '_' + str(mapping['expected']['session_key']) + '_v2'
    if state_key not in st.session_state:
        st.session_state[state_key] = {'last_good': None, 'error': None, 'last_attempt': 0.0}
    state = dict(st.session_state[state_key])
    if st.button('Haal sessiecontext op / ververs', key=state_key + '_fetch'):
        elapsed = time.time() - state['last_attempt']
        if elapsed < 30:
            st.warning(f'Wacht nog {math.ceil(30 - elapsed)} seconden voordat je opnieuw handmatig ophaalt.')
        else:
            state['last_attempt'] = time.time()
            state['error'] = None
            from f1_external_context_adapter import fetch_session_context
            adapter_result = fetch_session_context(mapping)
            state['error'] = adapter_result['error']
            if adapter_result['observation'] is not None:
                state['last_good'] = adapter_result['observation']
            st.session_state[state_key] = state
    if state['error']:
        st.warning(state['error'])
    context = state['last_good']
    if context is None:
        st.info('Er is in deze dashboardsessie nog geen context opgehaald.')
    else:
        if state['error']:
            st.info('Hieronder staan eerder opgehaalde gegevens. De laatste aanvraag is mislukt; deze gegevens zijn niet vernieuwd.')
        external_start = aware_time(context['scheduled_start_utc'])
        difference = (external_start - planned).total_seconds() if planned is not None else None
        age_minutes = (dt.datetime.now(dt.timezone.utc) - aware_time(context['retrieved_at_utc'])).total_seconds() / 60
        if age_minutes >= 15:
            st.warning(f'Deze context is {int(age_minutes)} minuten geleden opgehaald.')
        st.table([{'Veld': 'Geplande start volgens OpenF1', 'Waarde': context['scheduled_start_utc']}, {'Veld': 'Verschil met projectplanning', 'Waarde': f'{difference:g} seconden' if difference is not None else 'Niet vergelijkbaar: projectplanning ontbreekt'}, {'Veld': 'Geannuleerd volgens OpenF1', 'Waarde': 'Ja' if context['is_cancelled'] else 'Nee'}, {'Veld': 'Opgehaald UTC', 'Waarde': context['retrieved_at_utc']}])
        with st.expander('Technische contextgegevens', expanded=False):
            st.json({**context, 'event_key': event_key, 'session_key': mapping['expected']['session_key'], 'source_url': source_url, 'source_updated_at_utc': None, 'layout_verified': False, 'actual_race_start_verified': False})
    st.markdown(f'[Open bronpagina]({source_url})')
    st.caption('Ophaaltijd is geen bronwijzigingstijd. Geplande sessiestart bewijst geen daadwerkelijke racestart of exacte circuitlayout. Deze context wijzigt geen projectdata of workflowstatus.')

def _context_quality_findings(record):
    import datetime as dt
    import math
    findings = []

    def text(field):
        value = record.get(field)
        if value is None:
            return ''
        result = str(value).strip()
        return '' if result.lower() in {'nan', 'nat', '<na>', 'none'} else result

    def add(field, result, explanation):
        findings.append({'Veld / controle': field, 'Resultaat': result, 'Toelichting': explanation})
    for field in ('Circuit', 'CircuitLayoutSlug'):
        add(field, 'Aanwezig' if text(field) else 'Ontbreekt', 'Aanwezigheid bewijst geen externe layoutverificatie.')
    times = {}
    for field in ('AnnouncedRaceStartUTC', 'EventDateTimeUTC', 'FeatureSnapshotTimestampUTC'):
        value = text(field)
        if not value:
            add(field, 'Ontbreekt', 'Geen waarde in de weergegeven overviewrij.')
            continue
        try:
            parsed = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
            if parsed.tzinfo is None or parsed.utcoffset() is None:
                raise ValueError('tijdzone ontbreekt')
            times[field] = parsed.astimezone(dt.timezone.utc)
            add(field, 'Geldig formaat', 'Tijdzone aanwezig; omgerekend naar UTC.')
        except (ValueError, TypeError, OverflowError):
            add(field, 'Ongeldig', 'Tijdstempel ongeldig of zonder tijdzone.')
    if all((field in times for field in ('AnnouncedRaceStartUTC', 'EventDateTimeUTC'))):
        match = times['AnnouncedRaceStartUTC'] == times['EventDateTimeUTC']
        add('Vergelijking geplande projecttijden', 'Gelijk' if match else 'Verschil', 'Vergelijking van twee projectvelden; geen bewijs van daadwerkelijke start.')
    else:
        add('Vergelijking geplande projecttijden', 'Niet vergelijkbaar', 'Beide geplande projecttijden moeten geldig en aanwezig zijn.')
    flag_text = text('IsSprintWeekend')
    flag = None
    if not flag_text:
        add('IsSprintWeekend', 'Ontbreekt', 'Sprintcontext niet vastgesteld.')
    else:
        try:
            value = float(flag_text)
            if not math.isfinite(value) or value not in (0.0, 1.0):
                raise ValueError()
            flag = int(value)
            add('IsSprintWeekend', 'Geldig formaat', 'Sprintweekend.' if flag == 1 else 'Geen sprintweekend.')
        except (ValueError, TypeError):
            add('IsSprintWeekend', 'Ongeldig', 'Verwacht 0 of 1.')
    sprint = text('SprintFastestLap_sec')
    if flag == 0:
        add('SprintFastestLap_sec', 'Niet van toepassing', 'IsSprintWeekend is 0.')
    elif flag == 1:
        if not sprint:
            add('SprintFastestLap_sec', 'Ontbreekt', 'Nog geen sprinttijd in deze overviewrij. Dit is geen beoordeling van de canonical qualifying-gate.')
        else:
            try:
                value = float(sprint)
                if not math.isfinite(value) or value <= 0:
                    raise ValueError()
                add('SprintFastestLap_sec', 'Geldig formaat', 'Positief aantal seconden; officiële bron niet gecontroleerd.')
            except (ValueError, TypeError):
                add('SprintFastestLap_sec', 'Ongeldig', 'Verwacht een eindig, positief aantal seconden.')
    add('Daadwerkelijke racestart', 'Niet bewezen', 'Geplande tijden zijn geen ActualRaceStartUTC-bewijs.')
    return findings
