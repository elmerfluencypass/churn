import streamlit as st
from auth import login
import dataviz
import pov
import churn_score

st.set_page_config(page_title="Análise de Churn", layout="wide")

if login():
    st.sidebar.image("logo.png", use_column_width=True)
    st.sidebar.title("Menu")
    option = st.sidebar.radio("Ir para", ["DataViz", "POV", "Churn Score"])

    if option == "DataViz":
        dataviz.render()
    elif option == "POV":
        pov.render()
    elif option == "Churn Score":
        churn_score.render()
