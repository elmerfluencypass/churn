import streamlit as st
import pandas as pd
import gdown
import plotly.express as px

def run():
    st.header("📋 Dashboard Executivo de Churn")

    url = "https://drive.google.com/uc?id=1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT"
    gdown.download(url, "cadastro_clientes.csv", quiet=True)
    df = pd.read_csv("cadastro_clientes.csv")
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    col_pag = next((c for c in df.columns if "data" in c and "pagamento" in c), None)
    col_ini = next((c for c in df.columns if "inicio" in c and "curso" in c), None)
    if col_pag and col_ini:
        df[col_pag] = pd.to_datetime(df[col_pag], errors='coerce')
        df[col_ini] = pd.to_datetime(df[col_ini], errors='coerce')
        df['periodo_curso'] = ((df[col_pag] - df[col_ini]) / pd.Timedelta(days=30)).fillna(0).astype(int)
        df['mes_desistencia'] = df[col_pag].dt.month_name()

    st.metric("Total Alunos", len(df))
    st.metric("Ticket Médio (Estimado)", f"R$ {df.get('valor_total', pd.Series([0])).mean():,.2f}")

    if 'mes_desistencia' in df:
        fig1 = px.histogram(df, x='mes_desistencia', title="Desistências por Mês")
        st.plotly_chart(fig1, use_container_width=True)

    if 'uf' in df:
        fig2 = px.histogram(df, x='uf', title="Distribuição Geográfica")
        st.plotly_chart(fig2, use_container_width=True)

    if 'periodo_curso' in df:
        fig3 = px.box(df, x='periodo_curso', title="Período de Curso até Desistência")
        st.plotly_chart(fig3, use_container_width=True)
