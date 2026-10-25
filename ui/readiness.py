"""Streamlit presentation helpers for workflow readiness/status."""

import streamlit as st

_WORKFLOW_UI_STAGES = (
    "row_created",
    "circuit_regime_ready",
    "qualifying_ready",
    "race_weather_ready",
    "race_context_plausibility_ready",
    "prediction_ready",
    "predicted",
    "evaluated",
)

_WORKFLOW_UI_LABELS = {
    "row_created": "Event",
    "circuit_regime_ready": "Circuit",
    "qualifying_ready": "Qualifying",
    "race_weather_ready": "Weather",
    "race_context_plausibility_ready": "Plausibility",
    "prediction_ready": "Ready",
    "predicted": "Predicted",
    "evaluated": "Evaluated",
}

_WORKFLOW_UI_NEXT_ACTION = {
    "row_created": (
        "Verrijk circuit- en regimefeatures."
    ),
    "circuit_regime_ready": (
        "Vul en valideer qualifyinggegevens."
    ),
    "qualifying_ready": (
        "Vul en valideer race-weathergegevens."
    ),
    "race_weather_ready": (
        "Voer race-context plausibility uit en "
        "bevries daarna de definitieve feature snapshot."
    ),
    "race_context_plausibility_ready": (
        "Voltooi de canonical prediction-readiness gate."
    ),
    "prediction_ready": (
        "Draai de finale GBR + XGB + CAT prediction."
    ),
    "predicted": (
        "Na de race: koppel de actual via de "
        "canonical evaluatieflow."
    ),
    "evaluated": (
        "Deze raceworkflow is volledig geëvalueerd."
    ),
}

_WORKFLOW_UI_BLOCKER = {
    "row_created": (
        "Circuit- en regimegate is nog niet afgerond.",
        "Open 'Pre-race · Circuit' en voltooi de canonical circuit/regime stap.",
    ),
    "circuit_regime_ready": (
        "Qualifying gate is nog niet afgerond.",
        "Open 'Pre-race · Qualifying' en sla de vereiste qualifyingdata canonical op.",
    ),
    "qualifying_ready": (
        "Race-weather gate is nog niet afgerond.",
        "Open 'Pre-race · Race weather' en voltooi de canonical race-weather update.",
    ),
    "race_weather_ready": (
        "Race-context plausibility / snapshot readiness is nog niet afgerond.",
        "Open 'Prediction · Readiness & run' en rond plausibility, snapshot en registry-controls af.",
    ),
    "race_context_plausibility_ready": (
        "De finale prediction-readiness gate moet nog canonical worden bevestigd.",
        "Controleer de readiness checklist en bevestig daarna prediction_ready.",
    ),
    "prediction_ready": (
        None,
        "De race is prediction-ready. Draai de finale GBR + XGB + CAT prediction.",
    ),
    "predicted": (
        None,
        "Prediction is vastgelegd. Na de race kan de canonical evaluatieflow worden uitgevoerd.",
    ),
    "evaluated": (
        None,
        "De lifecycle is volledig geëvalueerd.",
    ),
}

def render_workflow_status_header(
    *,
    year,
    gp_slug,
    race_status,
):
    """
    Pure UI-render helper.

    Geen:
    - lifecycle mutation;
    - readiness-validatie;
    - dataset persistence;
    - businesslogica.

    RaceStatus wordt uitsluitend gepresenteerd.
    """

    status = (
        str(race_status).strip()
        if race_status is not None
        else ""
    )

    known = status in _WORKFLOW_UI_STAGES

    st.subheader(
        f"{year} — {gp_slug}"
    )

    col_race, col_status, col_next = st.columns(
        [1.4, 1.5, 2.0]
    )

    with col_race:
        st.metric(
            "Geselecteerde race",
            f"{year} — {gp_slug}",
        )

    with col_status:
        st.metric(
            "RaceStatus",
            status or "niet aangemaakt",
        )

    if known:
        current_idx = _WORKFLOW_UI_STAGES.index(
            status
        )

        if current_idx + 1 < len(
            _WORKFLOW_UI_STAGES
        ):
            next_status = _WORKFLOW_UI_STAGES[
                current_idx + 1
            ]
            next_label = _WORKFLOW_UI_LABELS[
                next_status
            ]
        else:
            next_label = "Afgerond"

        with col_next:
            st.metric(
                "Volgende fase",
                next_label,
            )

        # ---------------------------------------------
        # 8-stage lifecycle stepper
        # ---------------------------------------------
        step_columns = st.columns(
            len(_WORKFLOW_UI_STAGES)
        )

        for idx, stage in enumerate(
            _WORKFLOW_UI_STAGES
        ):
            label = _WORKFLOW_UI_LABELS[stage]

            # WORKFLOW_MILESTONE_STEPPER_V1: RaceStatus is an achieved milestone.
            if idx <= current_idx:
                symbol = "✅"
            elif idx == current_idx + 1:
                symbol = "🔵"
            else:
                symbol = "⬜"

            step_columns[idx].markdown(
                f"<div style='text-align:center'>"
                f"<div style='font-size:1.25rem'>"
                f"{symbol}"
                f"</div>"
                f"<div style='font-size:0.78rem'>"
                f"{label}"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        progress_value = (
            (current_idx + 1)
            / len(_WORKFLOW_UI_STAGES)
        )

        st.progress(progress_value)

        if status == "evaluated":
            st.success(
                "Workflow compleet: RaceStatus = evaluated."
            )

        elif status == "prediction_ready":
            st.success(
                "De race heeft de canonical status "
                "prediction_ready."
            )

            st.info(
                "**Aanbevolen volgende actie:** "
                + _WORKFLOW_UI_NEXT_ACTION[status]
            )

        else:
            st.info(
                "**Aanbevolen volgende actie:** "
                + _WORKFLOW_UI_NEXT_ACTION[status]
            )

            st.caption(
                "Eventuele blockers en ontbrekende gates "
                "worden door de bestaande canonical "
                "readiness-sectie bepaald."
            )

    else:
        with col_next:
            st.metric(
                "Volgende fase",
                (
                    "Nieuwe GP aanmaken"
                    if not status
                    else "Status controleren"
                ),
            )

        if not status:
            st.info(
                "Deze race is nog niet aangemaakt. "
                "Start met de tab 'Nieuwe GP'."
            )
        else:
            st.warning(
                "Onbekende RaceStatus aangetroffen. "
                "Gebruik canonical diagnostics voordat "
                "een workflowactie wordt uitgevoerd."
            )

    st.divider()


def render_readiness_action_panel(*, race_status):
    """
    Pure UI-presentatie.

    RaceStatus is authoritative.
    Deze helper voert geen validator uit en muteert niets.
    """

    status = (
        str(race_status).strip()
        if race_status is not None
        else ""
    )

    st.markdown("#### Readiness & volgende actie")

    if not status:
        st.warning(
            "Blocker: deze race heeft nog geen canonical RaceStatus."
        )
        st.info(
            "**Herstelactie:** maak of synchroniseer eerst het event "
            "via 'Event setup'."
        )
        return

    config = _WORKFLOW_UI_BLOCKER.get(status)

    if config is None:
        st.error(
            f"Onbekende RaceStatus: {status!r}. "
            "Gebruik Audit / diagnostics voordat een workflowactie wordt gestart."
        )
        return

    blocker, action = config

    c1, c2 = st.columns([1.15, 2.85])

    with c1:
        if blocker is None:
            st.success("Geen lifecycle-blocker")
        else:
            st.warning("Lifecycle-blocker")

    with c2:
        if blocker is not None:
            st.write(f"**Blocker:** {blocker}")

        st.write(f"**Aanbevolen actie:** {action}")

    if status == "race_context_plausibility_ready":
        st.caption(
            "De button voor prediction_ready is alleen actief vanuit "
            "race_context_plausibility_ready. De canonical "
            "prepare_prediction_ready-gate blijft daarna authoritative."
        )
    elif status not in {
        "prediction_ready",
        "predicted",
        "evaluated",
    }:
        st.caption(
            "Dit panel vertaalt alleen de authoritative RaceStatus naar UX. "
            "Exacte ontbrekende velden/gates blijven afkomstig uit de "
            "bestaande canonical readiness-checks."
        )
