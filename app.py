import streamlit as st
from utils import (
    autenticar_usuario,
    carregar_dados,
    tela_dataviz,
    tela_churn_score,
    tela_pov,
    tela_politica_churn
)

st.set_page_config(page_title="Churn Prediction", layout="wide")

# Sidebar logo e título fixos
st.sidebar.image("fluencypass_logo_converted.png", width=150)
menu = st.sidebar.radio("Menu", ["Dataviz", "Churn Score", "POV", "Política de Churn"])

# Sessão de autenticação
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    st.image("fluencypass_logo_converted.png", width=100)
    st.title("Churn Prediction")
    usuario = st.text_input("Usuário")
    senha = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if autenticar_usuario(usuario, senha):
            st.session_state.autenticado = True
            st.experimental_rerun()
        else:
            st.error("Credenciais inválidas.")
else:
    # Carregamento de dados
    dfs = carregar_dados()
    
    # Navegação por menu
    if menu == "Dataviz":
        tela_dataviz(dfs)
    elif menu == "Churn Score":
        tela_churn_score(dfs)
    elif menu == "POV":
        tela_pov(dfs)
    elif menu == "Política de Churn":
        tela_politica_churn(dfs)
