import streamlit as st
import pandas as pd
import gdown
import receita_perdida
import score_churn
import clusters
import dashboard

st.set_page_config(
    page_title="Churn de Alunos",
    page_icon="🎯",
    layout="wide"
)

@st.cache_data
def baixar_csv(nome, file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    arq = f"{nome}.csv"
    gdown.download(url, arq, quiet=True)
    df = pd.read_csv(arq)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df

# 🔗 Download de todos os arquivos centralizado
dfs = {
    "iugu_invoices": baixar_csv("iugu_invoices", "1M16LUHoXHUf_o4JJcE4EuZvkkzwawol2"),
    "iugu_subscription": baixar_csv("iugu_subscription", "1kMq0ydf7TL92wz_60wpfTf-obu0IJXtK"),
    "cadastro_clientes": baixar_csv("cadastro_clientes", "1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT"),
    "life_cycle": baixar_csv("life_cycle", "1kk5PZpfJPuvPFEYx9jO32xHcMdMLpDc8"),
}

st.title("🎯 Análise de Churn de Alunos")
st.markdown("Selecione uma das páginas à esquerda para explorar:")

menu = st.sidebar.radio("📂 Escolha a análise", [
    "Receita Perdida",
    "Score de Churn",
    "Clusters de Perfis",
    "Dashboard Executivo"
])

if menu == "Receita Perdida":
    receita_perdida.run(dfs)
elif menu == "Score de Churn":
    score_churn.run(dfs)
elif menu == "Clusters de Perfis":
    clusters.run(dfs)
elif menu == "Dashboard Executivo":
    dashboard.run(dfs)
