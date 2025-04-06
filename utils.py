import streamlit as st
import pandas as pd
import numpy as np
import calendar
import time
import requests
from io import BytesIO
import plotly.express as px
from vizro import Vizro
import plotly.graph_objects as go

@st.cache_data
def carregar_dados():
    def baixar_csv(gdrive_url):
        file_id = gdrive_url.split("id=")[-1]
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        response.raise_for_status()
        return pd.read_csv(BytesIO(response.content))

    clientes = baixar_csv("https://drive.google.com/uc?id=1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT")
    churn = baixar_csv("https://drive.google.com/uc?id=1kkdfrCjTjyzjqYfX7C9vDSdfod5HBYQS")
    pagamentos = baixar_csv("https://drive.google.com/uc?id=1j2RW-ryt4H3iX8nkaw371yBI0_EGA8MY")
    return {"clientes": clientes, "churn": churn, "pagamentos": pagamentos}

def barra_progresso(texto):
    barra = st.progress(0, text=texto)
    for i in range(1, 101):
        barra.progress(i, text=texto)
        time.sleep(0.005)
    barra.empty()

def tela_dataviz(dfs):
    st.title("📊 Visão Geral de Churn")
    col1, col2 = st.columns([0.8, 0.2])
    with col2:
        st.image("fluencypass_logo_converted.png", width=100)

    barra_progresso("Processando churn por mês...")

    churn = dfs["churn"]
    clientes = dfs["clientes"]

    churn["mes_churn_nome"] = pd.to_datetime(churn["data_churn"], errors="coerce").dt.month
    churn = churn[churn["mes_churn_nome"].notna()]
    churn["mes_churn_nome"] = churn["mes_churn_nome"].astype(int).apply(lambda x: calendar.month_name[x])
    churn["mes_churn_nome"] = pd.Categorical(churn["mes_churn_nome"], categories=list(calendar.month_name)[1:], ordered=True)

    churn_por_mes = churn.groupby("mes_churn_nome").size().reset_index(name="qtde")

    fig = px.bar(churn_por_mes, x="mes_churn_nome", y="qtde", color="qtde", color_continuous_scale="greens")
    fig.update_layout(title="Desistências por Mês", xaxis_title="Mês", yaxis_title="Qtd de Alunos", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    barra_progresso("Processando matriz de churn...")

    if "mes_plano_cancelado" not in churn.columns:
        st.error("Coluna 'mes_plano_cancelado' não encontrada nos dados de churn.")
        return

    churn["mes_plano_cancelado"] = churn["mes_plano_cancelado"].fillna(-1).astype(int)
    churn = churn[churn["mes_plano_cancelado"] >= 0]
    matriz = churn.pivot_table(index="mes_churn_nome", columns="mes_plano_cancelado", values="user_id", aggfunc="count", fill_value=0)

    st.subheader("📉 Matriz de Churn por Mês e Período do Plano")
    st.dataframe(matriz.style.background_gradient(axis=None, cmap="Greens"))
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

def tela_churn_score(dfs):
    st.title("📈 Score de Propensão ao Churn")
    col1, col2 = st.columns([0.8, 0.2])
    with col2:
        st.image("fluencypass_logo_converted.png", width=100)

    barra_progresso("Preparando dados para modelagem...")

    clientes = dfs["clientes"].copy()
    churn = dfs["churn"].copy()
    pagamentos = dfs["pagamentos"].copy()

    clientes["data_nascimento"] = pd.to_datetime(clientes["data_nascimento"], errors="coerce")
    clientes["idade"] = pd.to_datetime("today").year - clientes["data_nascimento"].dt.year
    clientes = clientes.dropna(subset=["idade"])

    # Base de treino
    churn["target"] = 1
    ativos = clientes[~clientes["user_id"].isin(churn["user_id"])]
    ativos["target"] = 0

    df_full = pd.concat([churn.merge(clientes, on="user_id", how="left"), ativos], ignore_index=True)
    df_full = df_full.dropna(subset=["idade", "plano_duracao_meses", "valor_mensalidade", "estado", "tipo_plano"])

    X = pd.get_dummies(df_full[["idade", "plano_duracao_meses", "valor_mensalidade", "estado", "tipo_plano"]])
    y = df_full["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    barra_progresso("Treinando modelo LightGBM...")

    modelo = lgb.LGBMClassifier()
    param_grid = {"num_leaves": [15, 31], "n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
    gs = GridSearchCV(modelo, param_grid, cv=3, scoring="roc_auc")
    gs.fit(X_train, y_train)

    y_score = gs.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_score)

    st.success(f"AUC no conjunto de teste: {auc:.3f}")

    barra_progresso("Calculando scores para alunos ativos...")

    base_ativos = ativos.dropna(subset=["idade", "plano_duracao_meses", "valor_mensalidade", "estado", "tipo_plano"])
    X_ativos = pd.get_dummies(base_ativos[["idade", "plano_duracao_meses", "valor_mensalidade", "estado", "tipo_plano"]])
    X_ativos = X_ativos.reindex(columns=X.columns, fill_value=0)

    base_ativos["score"] = gs.predict_proba(X_ativos)[:, 1]
    base_ativos["score"] = MinMaxScaler().fit_transform(base_ativos[["score"]])
    base_ativos["score"] = base_ativos["score"].round(4)
    base_ativos["mes_previsto"] = pd.to_datetime("today").month + 1

    st.subheader("📋 Score dos Alunos Ativos")
    st.dataframe(base_ativos[["user_id", "nome", "score", "mes_previsto"]].sort_values(by="score", ascending=False))

    csv = base_ativos[["user_id", "nome", "score", "mes_previsto"]].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar Score em CSV", csv, "score_churn.csv", "text/csv")
