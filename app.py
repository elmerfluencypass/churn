import streamlit as st
from utils import (
    carregar_dados,
    tela_dataviz,
    tela_churn_score,
    tela_pov,
    tela_politica_churn,
    tela_perfis_churn
)

st.set_page_config(page_title="Churn Prediction", layout="wide")

# Logo no canto superior direito
col1, col2 = st.columns([0.8, 0.2])
with col2:
    st.image("fluencypass_logo_converted.png", width=100)

if "autenticado" not in st.session_state:
    st.session_state["autenticado"] = False

def login():
    st.title("Login - Fluencypass")
    user = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if user == "fluencypass123" and password == "fluencypass123":
            st.session_state["autenticado"] = True
            st.experimental_rerun()
        else:
            st.error("Usuário ou senha incorretos.")

if not st.session_state["autenticado"]:
    login()
    st.stop()

# Menu lateral (aparece após autenticação)
menu = st.sidebar.radio("Menu", [
    "Dataviz",
    "Score de Churn",
    "POV",
    "Política de Churn",
    "Perfis de Churn"
])
st.sidebar.image("fluencypass_logo_converted.png", width=120)

# Carregar dados automaticamente após login
if "dados" not in st.session_state:
    with st.spinner("Carregando dados..."):
        st.session_state["dados"] = carregar_dados()

dfs = st.session_state["dados"]

# Roteamento entre menus
if menu == "Dataviz":
    tela_dataviz(dfs)
elif menu == "Score de Churn":
    tela_churn_score(dfs)
elif menu == "POV":
    tela_pov(dfs)
elif menu == "Política de Churn":
    tela_politica_churn(dfs)
elif menu == "Perfis de Churn":
    tela_perfis_churn(dfs)
