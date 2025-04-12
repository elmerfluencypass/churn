import streamlit as st
import receita_perdida
import score_churn

st.set_page_config(page_title="Churn Analytics", layout="wide")
st.title("🎯 Análise de Churn de Alunos")

pagina = st.sidebar.radio("📂 Escolha a análise", ["📊 Receita Perdida", "📈 Score de Churn"])

if pagina == "📊 Receita Perdida":
    receita_perdida.run()
elif pagina == "📈 Score de Churn":
    score_churn.run()
