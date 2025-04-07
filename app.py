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
# 🔽 Funções de carregamento dos dados
# ======================================

CSV_URLS = {
    "cadastro_clientes": "1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT",
    "churn_detectado": "1kkdfrCjTjyzjqYfX7C9vDSdfod5HBYQS",
    "historico_pagamentos": "1j2RW-ryt4H3iX8nkaw371yBI0_EGA8MY",
    "tbl_insider": "1tPqnQWmowQKNAx2M_4rLW5axH9DZ5TyW",
    "temp_life_cycle": "1eNcobHn8QJKduVRcs79LsbzT_2L7hK88"
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
    dfs = [pd.read_csv(f, parse_dates=True, dayfirst=True) for f in arquivos]
    return pd.concat(dfs, ignore_index=True)

# ======================================
# 🔍 Filtro por data
# ======================================

def filtro_data(df):
    df['data_evento'] = pd.to_datetime(df['data_evento'], errors='coerce')
    min_date = df['data_evento'].min()
    max_date = df['data_evento'].max()
    data_ini, data_fim = st.date_input("Filtrar por data:", [min_date, max_date])
    return df[(df['data_evento'] >= pd.to_datetime(data_ini)) & (df['data_evento'] <= pd.to_datetime(data_fim))]

# ======================================
# 📊 Visualizações
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
    df['ano'] = df['data_evento'].dt.year
    df['mes'] = df['data_evento'].dt.month
    matriz = df[df['status'] == 'desistente'].pivot_table(index='ano', columns='mes', values='id_aluno', aggfunc='count', fill_value=0)
    st.write("### Matriz de Desistências por Mês")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(matriz, cmap="YlOrRd", annot=True, fmt="d", linewidths=0.5, ax=ax)
    st.pyplot(fig)

# ======================================
# 🧠 Padrões de Desistência
# ======================================

def identificar_padroes(df):
    df_model = df.copy()
    df_model['desistencia'] = df_model['status'] == 'desistente'
    X = df_model.drop(columns=['status', 'data_evento', 'id_aluno', 'desistencia'], errors='ignore')
    y = df_model['desistencia']
    X = pd.get_dummies(X, drop_first=True)
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
# 🚀 Streamlit App
# ======================================

st.set_page_config(layout="wide")
st.title("📉 Painel de Análise de Desistências")

df = carregar_dados()
df_filtrado = filtro_data(df)

col1, col2 = st.columns(2)
with col1:
    histogramas(df_filtrado, 'idade')
    boxplot(df_filtrado, 'tempo_uso')

with col2:
    grafico_pizza(df_filtrado, 'sexo')
    correlacoes(df_filtrado)

st.divider()
gerar_matriz_temporal(df_filtrado)

st.divider()
identificar_padroes(df_filtrado)
