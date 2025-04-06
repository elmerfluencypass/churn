import streamlit as st
from utils import (
    autenticar_usuario,
    carregar_dados_locais,
    tela_dataviz,
    tela_churn_score,
    tela_pov,
    tela_politica_churn
)

st.set_page_config(page_title="Churn Prediction", layout="wide")

# Logo visível em todas as telas
logo_path = "fluencypass_logo_converted.png"
st.sidebar.image(logo_path, width=150)
menu = st.sidebar.radio("Menu", ["Dataviz", "Churn Score", "POV", "Política de Churn"])

if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    st.image(logo_path, width=100)
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
    dfs = carregar_dados_locais()

    st.image(logo_path, width=100)

    if menu == "Dataviz":
        tela_dataviz(dfs)
    elif menu == "Churn Score":
        tela_churn_score(dfs)
    elif menu == "POV":
        tela_pov(dfs)
    elif menu == "Política de Churn":
        tela_politica_churn(dfs)
