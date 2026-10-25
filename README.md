# F1 dashboard — publieke volledige workflow

Dit pakket bevat de actuele Streamlit-app en een eerste projectkopie uit Drive,
verzameld op 25 oktober 2026. Austin staat op `predicted`; de bestaande voorspelling
en bevroren snapshot worden byte voor byte meegenomen. Er wordt geen modelrun uitgevoerd.
De publiek gedeelde app heeft geen wachtwoord en biedt de volledige workflow.

## 1. Publiceren in GitHub

Repository: https://github.com/jansenmisja/F1_dashboard

Maak in GitHub Settings → Developer settings → Personal access tokens → Fine-grained tokens
één token voor uitsluitend deze repository, met Repository permissions → Contents: Read and write.
Gebruik dit token alleen in de verborgen invoer van de Colab-cel en in Streamlit Secrets.
Het installatieprogramma bewaart het token niet en uploadt geen credentials.

Upload dit ZIP-pakket met de meegeleverde Colab-cel. De publisher schrijft naar `main`,
bewaart andere bestaande repositorybestanden en stopt bij gelijktijdige wijzigingen.
Er wordt niets naar de bestaande Drive-projectmap geschreven.

## 2. Eenmalig deployen

Open https://share.streamlit.io en kies Create app.

- Repository: `jansenmisja/F1_dashboard`
- Branch: `main`
- Main file path: `cloud_app.py`
- Advanced settings → Python version: **3.13**
- Advanced settings → Secrets:

```toml
F1_GITHUB_TOKEN = "plak-hier-je-GitHub-token"
```

Klik Deploy en zet de app bij de sharing-instellingen op publiek / everyone.
Bezoekers voeren geen token of wachtwoord in. Het token is uitsluitend serverconfiguratie.
De echte URL verschijnt na het deployen; er is nog geen URL vooraf gereserveerd.

## 3. Opslag en ingebruikname

Bij de eerste start wordt branch `f1-data` aangemaakt met `f1-state.zip`.
Dit bevat datasets, ledger, modellog en auditbestanden. Canonieke servicewijzigingen
worden daar opgeslagen voordat de service succes teruggeeft. Een herstart haalt die
staat opnieuw op; `seed_state.zip` wordt alleen gebruikt als er nog geen cloudstaat is.
Gelijktijdige updates worden via de GitHub-bestandshash gecontroleerd.
Bij een onbereikbare opslag stopt een wijziging en herstelt de lokale servicetransactie.
Bij een onduidelijke netwerkuitkomst eerst de app verversen en de status controleren.

De cloudapp wordt na ingebruikname de werkende projectkopie. Wijzigingen in Colab/Drive
worden NIET automatisch heen en weer gesynchroniseerd. Voer verdere workflowacties
in de cloudapp uit. De snapshot in dit pakket is de staat van 25 oktober, vóór de race.
Verwijder of vervang de `f1-data`-branch niet om een app te herstarten.

De bestaande modelversies zijn vastgezet op de versies uit de actuele voorspelling:
numpy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1, xgboost 3.4.1 en catboost 1.2.10.
Er is hier geen nieuwe voorspelling of hertraining uitgevoerd. Live GitHub-opslag en
installatie van deze dependencies moeten tijdens de eerste deployment nog bevestigd worden.

Controleer na starten bij Austin 2026: RaceStatus `predicted`, snapshot
`2026-10-25T15:56:22.782705Z`, GBR `98.800`, XGB `98.807`, CAT `98.018` seconden.
Een normale herstart van de app moet deze waarden behouden. Voer geen final prediction opnieuw uit.

## Lokale verificatie

De opslagadapter is getest met een nagebootste GitHub-API: eerste initialisatie,
canonieke opslag, herstart, conflicten, netwerkfout, verloren succesantwoord,
afwijzende finale gate en ongeldige ZIP-paden. Zie `VALIDATION.md`.

Officiële deploymentdocumentatie:
https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
https://docs.github.com/en/rest/repos/contents
