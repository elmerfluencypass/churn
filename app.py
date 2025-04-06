import streamlit as st
from utils import (
    carregar_dados_google_drive,
    tela_dataviz,
    tela_churn_score,
    tela_pov,
    tela_politica_churn,
    tela_perfis_churn
)

st.set_page_config(page_title="Churn Prediction", layout="wide")

st.sidebar.image("fluencypass_logo_converted.png", width=180)
pagina = st.sidebar.radio("Menu", ["Dataviz", "Churn Score", "POV", "Política de Churn", "Perfis de Churn"])

st.title("Fluencypass")

dfs = carregar_dados_google_drive()

if pagina == "Dataviz":
    tela_dataviz(dfs)
elif pagina == "Churn Score":
    tela_churn_score(dfs)
elif pagina == "POV":
    tela_pov(dfs)
elif pagina == "Política de Churn":
    tela_politica_churn(dfs)
elif pagina == "Perfis de Churn":
    tela_perfis_churn(dfs)
