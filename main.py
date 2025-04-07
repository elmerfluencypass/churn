import streamlit as st

# ⚠️ Page config deve ser o primeiro comando Streamlit
st.set_page_config(
    page_title="Fluency Churn Dashboard",
    layout="wide",
    page_icon="📊"
)

from auth import login
import dataviz
import pov
import churn_score

# Login e navegação
if login():
    st.sidebar.image("logo.png", use_column_width=True)
    st.sidebar.title("Menu")
    menu_opcao = st.sidebar.radio("Ir para", ["DataViz", "POV", "Churn Score"])

    if menu_opcao == "DataViz":
        dataviz.render()

    elif menu_opcao == "POV":
        pov.render()

    elif menu_opcao == "Churn Score":
        churn_score.render()
