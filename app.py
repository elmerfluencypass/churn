import streamlit as st
from utils import (
    carregar_dados_google_drive,
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

# Autenticação do usuário
if "autenticado" not in st.session_state:
    st.session_state.autenticado = False

if not st.session_state.autenticado:
    tela_login()
    st.stop()

# Carregamento dos dados após login bem-sucedido
dfs = carregar_dados_google_drive()

# Barra lateral com menu de navegação e logo
st.sidebar.image("fluencypass_logo_converted.png", width=150)
menu = st.sidebar.radio("Menu", ["Dataviz", "Score de Churn", "POV", "Política de Churn", "Perfis de Churn"])

# Renderização das páginas conforme o menu
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
