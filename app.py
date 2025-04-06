
import streamlit as st
from utils_CORRIGIDO_v2 import (
    autenticar_usuario,
    carregar_dados_locais,
    tela_dataviz,
    tela_churn_score,
    tela_pov,
    tela_politica_churn,
    tela_perfis_churn
)

st.set_page_config(page_title="Churn Prediction", layout="wide")

# Header com logo
col1, col2 = st.columns([0.1, 0.9])
with col1:
    st.image("fluencypass_logo_converted.png", width=80)
with col2:
    st.title("Churn Prediction")

if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    usuario = st.text_input("Usuário")
    senha = st.text_input("Senha", type="password")
    if st.button("Login"):
        if autenticar_usuario(usuario, senha):
            st.session_state.autenticado = True
            st.experimental_rerun()
        else:
            st.error("Usuário ou senha incorretos")
else:
    menu = st.sidebar.radio("Menu", [
        "Dataviz",
        "Churn Score",
        "POV",
        "Política de Churn",
        "Perfis de Churn"
    ])
    st.sidebar.image("fluencypass_logo_converted.png", width=100)

    if "dados" not in st.session_state:
        with st.spinner("Carregando dados..."):
            st.session_state.dados = carregar_dados_locais()

    dfs = st.session_state.dados

    if dfs:
        if menu == "Dataviz":
            tela_dataviz(dfs)
        elif menu == "Churn Score":
            tela_churn_score(dfs)
        elif menu == "POV":
            tela_pov(dfs)
        elif menu == "Política de Churn":
            tela_politica_churn(dfs)
        elif menu == "Perfis de Churn":
            tela_perfis_churn(dfs)
