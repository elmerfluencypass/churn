import streamlit as st
from utils_original import (
    carregar_dados,
    adicionar_logo,
    barra_progresso_mensagem,
    tela_login,
    tela_dataviz,
    tela_score_churn,
    tela_pov,
    tela_politica_churn,
    tela_perfis_churn
)

st.set_page_config(page_title="Fluencypass Churn", layout="wide")

if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    tela_login()
    st.stop()

dfs = carregar_dados()

st.sidebar.image("fluencypass_logo_converted.png", width=150)
menu = st.sidebar.radio("Menu", [
    "Dataviz",
    "Score de Churn",
    "POV",
    "Política de Churn",
    "Perfis de Churn"
])

if menu == "Dataviz":
    tela_dataviz(dfs)
elif menu == "Score de Churn":
    tela_score_churn(dfs)
elif menu == "POV":
    tela_pov(dfs)
elif menu == "Política de Churn":
    tela_politica_churn(dfs)
elif menu == "Perfis de Churn":
    tela_perfis_churn(dfs)
