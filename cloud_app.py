"""Public Streamlit entrypoint; project storage credentials are server-side only."""
from pathlib import Path
import os
import runpy
import streamlit as st

ROOT = Path(__file__).resolve().parent
os.environ['F1_PROJECT_ROOT'] = str(ROOT)
os.environ['F1_CLOUD_PERSISTENCE'] = 'github_v1'
try:
    token = st.secrets.get('F1_GITHUB_TOKEN', '')
except Exception:
    token = ''
if not token:
    st.error('Eenmalige hostingconfiguratie ontbreekt: voeg F1_GITHUB_TOKEN toe aan Streamlit Secrets.')
    st.stop()
os.environ['F1_GITHUB_TOKEN'] = str(token)
try:
    from f1_pipeline.cloud_state import bootstrap
    bootstrap()
except Exception as exc:
    st.error('Projectopslag niet beschikbaar. De app start niet met een lege of verouderde dataset.')
    st.code(str(exc))
    st.stop()

_app_namespace = runpy.run_path(str(ROOT / 'app_streamlit.py'), run_name='__main__')
