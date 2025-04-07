import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
import gdown

# ======================================
# 🔽 Carregamento de Dados do Google Drive
# ======================================

CSV_URLS = {
    "cadastro_clientes": "1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT",
    "churn_detectado": "1kkdfrCjTjyzjqYfX7C9vDSdfod5HBYQS",
    "historico_pagamentos": "1j2RW-ryt4H3iX8nkaw371yBI0_EGA8MY",
    "tbl_insider": "1tPqnQWmowQKNAx2M_4rLW5axH9DZ5TyW",
    "iugu_invoices": "1eNcobHn8QJKduVRcs79LsbzT_2L7hK88"
}

def baixar_dados():
    os.makedirs("data", exist_ok=True)
    arquivos = []
    for nome, file_id in CSV_URLS.items():
        path = f"data/{nome}.csv"
        url = f"https://drive.google.com/uc?id={file_id}"
        if not os.path.exists(path):
            gdown.download(url, path, quiet=False)
        arquivos.append(path)
    return arquivos

def carregar_dados():
    arquivos = baixar_dados()
    dfs = []

    for f in arquivos:
        df = pd.read_csv(f, parse_dates=True, dayfirst=True)
        
        # Se a tabela for iugu_invoices, vamos usar due_date como referência temporal
        if "iugu_invoices" in f and "due_date" in df.columns:
            df["due_date"] = pd.to_datetime(df["due_date"], errors='coerce')

        dfs.append(df)

    df_geral = pd.concat(dfs, ignore_index=True)
    return df_geral

# ======================================
# 📅 Filtro por Data (com base em due_date)
# ======================================

def filtro_data(df):
    if 'due_date' not in df.columns:
        st.error("Coluna 'due_date' não encontrada nos dados.")
        return df

    df_valid = df[df['due_date'].notna()]

    if df_valid.empty:
        st.warning("Não há datas válidas na coluna 'due_date' para aplicar o filtro.")
        return df

    min_date = df_valid['due_date'].min()
    max_date = df_valid['due_date'].max()
    data_ini, data_fim = st.date_input("Filtrar por data de vencimento:", [min_date, max_date])

    return df[(df['due_date'] >= pd.to_datetime(data_ini)) & (df['due_date'] <= pd.to_datetime(data_fim))]

# ======================================
# 📊 Gráficos
# ======================================

def histogramas(df, coluna):
    fig = px.histogram(df, x=coluna, title=f"Histograma de {coluna}")
    st.plotly_chart(fig, use_container_width=True)

def boxplot(df, coluna):
    fig = px.box(df, y=coluna, title=f"Boxplot de {coluna}")
    st.plotly_chart(fig, use_container_width=True)

def grafico_pizza(df, coluna):
    contagem = df[coluna].value_counts().reset_index()
    contagem.columns = [coluna, 'total']
    fig = px.pie(contagem, names=coluna, values='total', title=f"Distribuição de {coluna}")
    st.plotly_chart(fig, use_container_width=True)

def correlacoes(df):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

# ======================================
# 🔥 Matriz Temporal
# ======================================

def gerar_matriz_temporal(df):
    if 'due_date' not in df.columns:
        st.warning("Coluna 'due_date' não encontrada.")
        return

    df['ano'] = pd.to_datetime(df['due_date'], errors='coerce').dt.year
    df['mes'] = pd.to_datetime(df['due_date'], errors='coerce').dt.month

    if 'status' in df.columns:
        matriz = df[df['status'] == 'desistente'].pivot_table(
            index='ano', columns='mes', values='id_aluno', aggfunc='count', fill_value=0
        )
        st.write("### Matriz de Desistências por Mês (com base em due_date)")
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(matriz, cmap="YlOrRd", annot=True, fmt="d", linewidths=0.5, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Coluna 'status' não encontrada para gerar a matriz de desistências.")

# ======================================
# 🧠 Identificação de Padrões
# ======================================

def identificar_padroes(df):
    if 'status' not in df.columns:
        st.warning("Coluna 'status' ausente — não é possível identificar padrões de desistência.")
        return

    df_model = df.copy()
    df_model['desistencia'] = df_model['status'] == 'desistente'

    drop_cols = ['status', 'due_date', 'id_aluno', 'desistencia']
    X = df_model.drop(columns=[col for col in drop_cols if col in df_model.columns], errors='ignore')
    y = df_model['desistencia']

    X = pd.get_dummies(X, drop_first=True)
    if X.empty:
        st.warning("Dados insuficientes para análise de padrão.")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    importances = pd.Series(clf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(10)

    st.write("### Principais Variáveis antes da Desistência")
    st.bar_chart(top_features)

    preds = clf.predict(X_test)
    st.text("Relatório de Classificação:")
    st.text(classification_report(y_test, preds))

# ======================================
# 🚀 Execução Principal
# ======================================

st.set_page_config(layout="wide")
st.title("📉 Painel de Análise de Desistências")

df = carregar_dados()
df_filtrado = filtro_data(df)

col1, col2 = st.columns(2)
with col1:
    if 'idade' in df_filtrado.columns:
        histogramas(df_filtrado, 'idade')
    if 'tempo_uso' in df_filtrado.columns:
        boxplot(df_filtrado, 'tempo_uso')

with col2:
    if 'sexo' in df_filtrado.columns:
        grafico_pizza(df_filtrado, 'sexo')
    correlacoes(df_filtrado)

st.divider()
gerar_matriz_temporal(df_filtrado)

st.divider()
identificar_padroes(df_filtrado)
