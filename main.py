import streamlit as st
import os
from auth import login
import dataviz
import pov
import churn_score

# ⚠️ DEVE ser a primeira chamada Streamlit
st.set_page_config(
    page_title="Fluency Churn Dashboard",
    layout="wide",
    page_icon="📊"
)

# Controle de sessão para login
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Acesso Restrito")
    st.markdown("Faça login para acessar o sistema.")
    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if username == "fluencypass123" and password == "fluencypass123":
            st.session_state.authenticated = True
            st.experimental_rerun()
        else:
            st.error("Credenciais inválidas.")
else:
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=False)

    st.sidebar.title("Menu")
    menu_opcao = st.sidebar.radio("Ir para", ["DataViz", "POV", "Churn Score"])

    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, use_container_width=True)

    if menu_opcao == "DataViz":
        dataviz.render()
    elif menu_opcao == "POV":
        pov.render()
    elif menu_opcao == "Churn Score":
        churn_score.render()
