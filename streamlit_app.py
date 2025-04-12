import streamlit as st
import receita_perdida
import score_churn
import clusters
import dashboard

st.set_page_config(
    page_title="Churn de Alunos",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Análise de Churn de Alunos")
st.markdown("Selecione uma das páginas à esquerda para explorar:")

menu = st.sidebar.radio("📂 Escolha a análise", [
    "Receita Perdida",
    "Score de Churn",
    "Clusters de Perfis",
    "Dashboard Executivo"
])

if menu == "Receita Perdida":
    receita_perdida.run()
elif menu == "Score de Churn":
    score_churn.run()
elif menu == "Clusters de Perfis":
    clusters.run()
elif menu == "Dashboard Executivo":
    dashboard.run()
