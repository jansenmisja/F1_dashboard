# Verificatie

Opslagtests met nagebootste GitHub-API geslaagd:

1. Initialisatie van branch en eerste snapshot.
2. Canonieke servicewijziging wordt gepubliceerd.
3. Een koude herstart herstelt opgeslagen projectbestanden.
4. Een gelijktijdig opslagconflict herstelt de lokale transactie.
5. Een netwerkfout herstelt de lokale transactie.
6. Een verloren succesantwoord wordt gecontroleerd zonder nieuwe workflowrun.
7. Een afgewezen finale gate publiceert geen gewijzigde projectstaat.
8. Een archief met paden buiten het project wordt geweigerd.

Ook gecontroleerd: syntax, bytegelijkheid van de 185 brondata-/auditbestanden,
bestaande Austin-voorspelling en integriteit van het installatiearchief.

Niet live getest: GitHub met jouw token, Streamlit Community Cloud,
het installeren van de vastgezette modeldependencies en de volledige interactieve app.
De bestaande operationele voorspelling is niet opnieuw uitgevoerd.
