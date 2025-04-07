import streamlit as st
import os

# ⚠️ DEVE SER O PRIMEIRO COMANDO Streamlit
st.set_page_config(
    page_title="Fluency Churn Dashboard",
    layout="wide",
    page_icon="📊"
)

from auth import login
import dataviz
import pov
import churn_score

# Autenticação
if login():
    logo_path = "logo.png"
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, use_container_width=True)
    else:
        st.sidebar.warning("Logo não encontrado.")

    st.sidebar.title("Menu")
    menu_opcao = st.sidebar.radio("Ir para", ["DataViz", "POV", "Churn Score"])

    if menu_opcao == "DataViz":
        dataviz.render()

    elif menu_opcao == "POV":
        pov.render()

    elif menu_opcao == "Churn Score":
        churn_score.render()
