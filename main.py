import streamlit as st
from auth import login
from data_loader import load_all_data
from plots import (
    show_churn_dashboard,
    show_churn_profile,
    calcular_variaveis_score,
    calcular_score_de_churn
)

st.set_page_config(page_title="Churn Analysis", layout="wide")

def main():
    if not login():
        return

    st.image("logo.webp", width=150)

    menu = st.sidebar.selectbox("Menu", [
        "Dataviz",
        "Perfil de Churn",
        "Churn Score",
        "Score"
    ])

    if menu == "Dataviz":
        with st.spinner("Carregando dados..."):
            progress = st.progress(0)
            data = load_all_data()
            progress.progress(100)
        show_churn_dashboard(data)

    elif menu == "Perfil de Churn":
        with st.spinner("Carregando dados..."):
            progress = st.progress(0)
            data = load_all_data()
            progress.progress(100)
        show_churn_profile(data)

    elif menu == "Churn Score":
        with st.spinner("Calculando variáveis para modelagem..."):
            progress = st.progress(0)
            data = load_all_data()
            progress.progress(50)
            calcular_variaveis_score(data["df_modelagem"])
            progress.progress(100)

    elif menu == "Score":
        with st.spinner("Calculando Score de Propensão à Desistência..."):
            progress = st.progress(0)
            data = load_all_data()
            progress.progress(50)
            calcular_score_de_churn(data["df_modelagem"])
            progress.progress(100)

if __name__ == "__main__":
    main()
