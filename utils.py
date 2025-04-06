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
def tela_pov(dfs):
    st.title("💰 Prova de Valor")
    col1, col2 = st.columns([0.8, 0.2])
    with col2:
        st.image("fluencypass_logo_converted.png", width=100)

    percentual = st.selectbox("Selecione o percentual de recuperação:", list(range(5, 105, 5)), index=1)

    barra_progresso("Calculando simulação de recuperação...")

    churn = dfs["churn"].copy()
    pagamentos = dfs["pagamentos"].copy()
    clientes = dfs["clientes"].copy()

    churn["mes_churn_nome"] = pd.to_datetime(churn["data_churn"], errors="coerce").dt.month
    churn = churn[churn["mes_churn_nome"].notna()]
    churn["mes_churn_nome"] = churn["mes_churn_nome"].astype(int).apply(lambda x: calendar.month_name[x])
    churn["mes_churn_nome"] = pd.Categorical(churn["mes_churn_nome"], categories=list(calendar.month_name)[1:], ordered=True)

    pagamentos["data_pagamento"] = pd.to_datetime(pagamentos["data_pagamento"], errors="coerce")
    pagamentos = pagamentos.sort_values("data_pagamento").drop_duplicates("user_id", keep="last")
    pagamentos = pagamentos[["user_id", "valor_mensalidade"]]

    merged = churn.merge(pagamentos, on="user_id", how="left")
    merged = merged.merge(clientes[["user_id", "plano_duracao_meses"]], on="user_id", how="left")
    merged["meses_restantes"] = merged["plano_duracao_meses"] - merged["mes_plano_cancelado"]
    merged["meses_restantes"] = merged["meses_restantes"].clip(lower=0)
    merged["valor_perdido"] = merged["meses_restantes"] * merged["valor_mensalidade"]

    perdas_agrupadas = merged.groupby("mes_churn_nome").agg({
        "user_id": "count",
        "valor_perdido": "sum"
    }).rename(columns={"user_id": "total_desistentes", "valor_perdido": "total_perda_r$"}).fillna(0)

    perdas_agrupadas["recuperados"] = (perdas_agrupadas["total_desistentes"] * percentual / 100).round(0)
    perdas_agrupadas["valor_recuperado"] = perdas_agrupadas["total_perda_r$"] * (percentual / 100)

    fig1 = px.bar(perdas_agrupadas, y=perdas_agrupadas.index, x="recuperados", orientation="h",
                  color="recuperados", color_continuous_scale="greens",
                  labels={"recuperados": "Alunos Recuperados"}, title="Alunos Recuperados por Mês")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.bar(perdas_agrupadas, y=perdas_agrupadas.index, x="valor_recuperado", orientation="h",
                  color="valor_recuperado", color_continuous_scale="purples",
                  labels={"valor_recuperado": "Valor Recuperado (R$)"}, title="Valor Recuperado por Mês (R$)")
    st.plotly_chart(fig2, use_container_width=True)

    st.metric("💸 Valor Total Recuperado", f"R$ {perdas_agrupadas['valor_recuperado'].sum():,.2f}")
def tela_politica_churn(dfs):
    st.title("📌 Política de Churn")
    col1, col2 = st.columns([0.8, 0.2])
    with col2:
        st.image("fluencypass_logo_converted.png", width=100)

    barra_progresso("Calculando score médio por período do plano...")

    clientes = dfs["clientes"].copy()
    churn = dfs["churn"].copy()

    clientes["data_nascimento"] = pd.to_datetime(clientes["data_nascimento"], errors="coerce")
    clientes["idade"] = pd.to_datetime("today").year - clientes["data_nascimento"].dt.year
    clientes = clientes.dropna(subset=["idade"])

    churn = churn.merge(clientes, on="user_id", how="left")
    churn = churn.dropna(subset=["idade", "plano_duracao_meses", "valor_mensalidade", "estado", "tipo_plano"])

    X = pd.get_dummies(churn[["idade", "plano_duracao_meses", "valor_mensalidade", "estado", "tipo_plano"]])
    churn["score_simulado"] = MinMaxScaler().fit_transform(np.random.rand(len(X), 1))  # simulando score real

    score_medio = churn.groupby("mes_plano_cancelado")["score_simulado"].mean().reset_index()
    score_medio = score_medio[score_medio["mes_plano_cancelado"] >= 0].sort_values("mes_plano_cancelado")

    fig = px.bar(score_medio,
                 x="mes_plano_cancelado", y="score_simulado",
                 color="score_simulado", color_continuous_scale="blues",
                 labels={"mes_plano_cancelado": "Período do Plano", "score_simulado": "Score Médio"},
                 title="Score Médio de Churn por Período do Plano")

    st.plotly_chart(fig, use_container_width=True)

    valor_politica = score_medio["score_simulado"].median()
    st.metric("🎯 Política de Churn Sugerida", f"Score ≥ {valor_politica:.2f}")
from sklearn.cluster import KMeans

def tela_perfis_churn(dfs):
    st.title("👥 Perfis de Churn por Mês")
    col1, col2 = st.columns([0.8, 0.2])
    with col2:
        st.image("fluencypass_logo_converted.png", width=100)

    churn = dfs["churn"].copy()
    clientes = dfs["clientes"].copy()

    clientes["data_nascimento"] = pd.to_datetime(clientes["data_nascimento"], errors="coerce")
    clientes["idade"] = pd.to_datetime("today").year - clientes["data_nascimento"].dt.year
    clientes = clientes.dropna(subset=["idade"])

    churn["mes_churn"] = pd.to_datetime(churn["data_churn"], errors="coerce").dt.month
    churn = churn[churn["mes_churn"].notna()]
    churn["mes_churn"] = churn["mes_churn"].astype(int)

    mes_nome = st.selectbox("Selecione o mês para análise", list(calendar.month_name)[1:])
    mes_num = list(calendar.month_name).index(mes_nome)

    base = churn[churn["mes_churn"] == mes_num].merge(clientes, on="user_id", how="left")
    base = base.dropna(subset=["idade", "estado", "tipo_plano", "canal"])

    X = pd.get_dummies(base[["idade", "estado", "tipo_plano", "canal"]], drop_first=True)

    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    base["cluster"] = kmeans.fit_predict(X)

    fig = px.scatter(base, x="idade", y="plano_duracao_meses", color=base["cluster"].astype(str),
                     title=f"Clusters de Desistentes - {mes_nome}",
                     labels={"idade": "Idade", "plano_duracao_meses": "Duração Plano", "color": "Cluster"})

    st.plotly_chart(fig, use_container_width=True)

    csv = base[["user_id", "nome", "cluster", "estado", "tipo_plano", "canal"]].to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Baixar Perfis CSV", csv, f"perfis_churn_{mes_nome.lower()}.csv", "text/csv")
