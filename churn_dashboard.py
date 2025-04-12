import streamlit as st
import pandas as pd
import gdown
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Links dos arquivos no Google Drive (ids extraídos dos links compartilhados)
FILE_IDS = {
    "cadastro_clientes": "1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT",
    "customer_profile_table": "1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT",
    "churn_detectado": "1kkdfrCjTjyzjqYfX7C9vDSdfod5HBYQS",
    "faq": "14QVhlWyCmoum-6NS6Vke1d-l3_2ci7na",
    "historico_pagamentos": "1j2RW-ryt4H3iX8nkaw371yBI0_EGA8MY",
    "iugu_invoices": "1M16LUHoXHUf_o4JJcE4EuZvkkzwawol2",
    "iugu_subscription": "1kMq0ydf7TL92wz_60wpfTf-obu0IJXtK",
    "life_cycle": "1kk5PZpfJPuvPFEYx9jO32xHcMdMLpDc8",
    "tbl_insider": "1tPqnQWmowQKNAx2M_4rLW5axH9DZ5TyW"
}

@st.cache_data
def download_and_load_csv(name, file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    output = f"{name}.csv"
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

st.set_page_config(page_title="Dashboard Churn e Receita", layout="wide")
st.title("📊 Dashboard de Churn e Receita Perdida")

# Carregar todos os arquivos
data = {name: download_and_load_csv(name, file_id) for name, file_id in FILE_IDS.items()}

# Interface: Seleção de dados e visualização
st.sidebar.title("📂 Visualizar Dados")
selected_df = st.sidebar.selectbox("Escolha uma base para visualizar", list(data.keys()))
st.subheader(f"🔍 Visualização da base: {selected_df}")
st.dataframe(data[selected_df].head())

# KPI simples com base em churn_detectado + cadastro
if "churn_detectado" in data and "cadastro_clientes" in data:
    churn_df = data["churn_detectado"]
    cad_df = data["cadastro_clientes"]

    if "user_id" in churn_df.columns:
        st.subheader("📉 Métricas de Churn e Receita")
        total_alunos = cad_df["user_id"].nunique()
        churnados = churn_df["user_id"].nunique()
        churn_rate = churnados / total_alunos if total_alunos else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("Total de Alunos", total_alunos)
        col2.metric("Total com Churn", churnados)
        col3.metric("Churn Rate (%)", f"{churn_rate*100:.1f}%")

# Visualização de receita perdida (se aplicável)
if "iugu_subscription" in data and "iugu_invoices" in data:
    st.subheader("💸 Receita Perdida Estimada por Churn")

    subs = data["iugu_subscription"].copy()
    invoices = data["iugu_invoices"].copy()
    invoices['paid_at'] = pd.to_datetime(invoices['paid_at'], errors='coerce')

    subs['preco_mensal'] = subs['price_cents'] / 100
    subs['customer_name_lower'] = subs['customer_name'].str.lower().str.strip()
    invoices['customer_name_lower'] = invoices['customer_name'].str.lower().str.strip()

    pagas = invoices[invoices['status'] == 'paid']
    pag_por_aluno = pagas.groupby('customer_name_lower').agg(pagas=('paid_value', 'count'), ultimo_pagamento=('paid_at', 'max')).reset_index()
    df = subs.merge(pag_por_aluno, on='customer_name_lower', how='left')
    df['meses_restantes'] = (df['max_cycles'].fillna(12) - df['pagas'].fillna(0)).clip(lower=0)
    df['receita_perdida'] = df['meses_restantes'] * df['preco_mensal']
    df['mes_desistencia'] = df['ultimo_pagamento'].dt.month_name()
    df['periodo_curso'] = df['pagas'].fillna(0).astype(int)

    # Matriz e gráfico
    matriz = df.pivot_table(index='mes_desistencia', columns='periodo_curso', values='receita_perdida', aggfunc='sum', fill_value=0)
    st.dataframe(matriz.round(2))

    fig = px.imshow(matriz, labels=dict(x="Período do Curso", y="Mês Desistência", color="R$ Perdido"),
                    title="Heatmap: Receita Perdida por Período x Mês", aspect="auto", color_continuous_scale='Reds')
    st.plotly_chart(fig)

    # Exportação
    st.download_button("📥 Baixar Matriz CSV", matriz.to_csv().encode('utf-8'), "matriz_receita.csv", "text/csv")
