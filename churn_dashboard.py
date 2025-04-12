import streamlit as st
import pandas as pd
import gdown
import gc
import plotly.express as px

st.set_page_config(page_title="Churn Dashboard Light", layout="wide")
st.title("⚡ Churn & Receita Dashboard (Versão Otimizada)")

# IDs no Google Drive
FILE_IDS = {
    "iugu_invoices": "1M16LUHoXHUf_o4JJcE4EuZvkkzwawol2",
    "iugu_subscription": "1kMq0ydf7TL92wz_60wpfTf-obu0IJXtK"
}

@st.cache_data
def download_and_sample_csv(name, file_id, date_column=None, sample_n=10000):
    url = f"https://drive.google.com/uc?id={file_id}"
    output = f"{name}.csv"
    gdown.download(url, output, quiet=True)
    df = pd.read_csv(output, low_memory=False, parse_dates=[date_column] if date_column else None)
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=42)
    return df

# MENU lateral
st.sidebar.title("🔎 Selecione o Dataset")
dataset = st.sidebar.radio("Escolha um arquivo para análise", ["Receita Perdida"])

# Receita Perdida (iugu)
if dataset == "Receita Perdida":
    st.subheader("💸 Receita Perdida por Aluno (Amostra)")
    invoices = download_and_sample_csv("iugu_invoices", FILE_IDS["iugu_invoices"], "paid_at")
    subs = download_and_sample_csv("iugu_subscription", FILE_IDS["iugu_subscription"])

    # Preparo leve
    invoices['customer_name_lower'] = invoices['customer_name'].str.strip().str.lower()
    subs['customer_name_lower'] = subs['customer_name'].str.strip().str.lower()
    subs['preco_mensal'] = subs['price_cents'] / 100
    subs['max_cycles'] = subs['max_cycles'].fillna(12)

    pagas = invoices[invoices['status'] == 'paid']
    pag_por_aluno = pagas.groupby('customer_name_lower')\
        .agg(pagas=('paid_value', 'count'), ultimo_pagamento=('paid_at', 'max')).reset_index()

    df = subs.merge(pag_por_aluno, on='customer_name_lower', how='inner')
    df['meses_restantes'] = (df['max_cycles'] - df['pagas']).clip(lower=0)
    df['receita_perdida'] = df['meses_restantes'] * df['preco_mensal']
    df['mes_desistencia'] = df['ultimo_pagamento'].dt.month_name()
    df['periodo_curso'] = df['pagas']

    st.write(f"🔢 Amostra carregada: {len(df):,} registros")

    col1, col2 = st.columns(2)
    col1.metric("Clientes únicos", df['customer_name_lower'].nunique())
    col2.metric("Receita Perdida Total (R$)", f"{df['receita_perdida'].sum():,.2f}")

    matriz = df.pivot_table(index='mes_desistencia', columns='periodo_curso', values='receita_perdida', aggfunc='sum', fill_value=0)
    st.dataframe(matriz.round(2))

    fig = px.imshow(matriz, text_auto=True, color_continuous_scale='Reds', aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

    # Exportar
    st.download_button("📥 Baixar Matriz CSV", matriz.to_csv().encode('utf-8'), "matriz_receita.csv", "text/csv")

    # Limpeza de memória
    if st.button("🧹 Limpar Memória"):
        del invoices, subs, pagas, pag_por_aluno, df
        gc.collect()
        st.success("Memória liberada com sucesso!")
