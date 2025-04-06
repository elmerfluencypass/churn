import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

@st.cache_data
def carregar_dados_google_drive():
    urls = {
        "clientes": "https://drive.google.com/uc?id=1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT",
        "churn": "https://drive.google.com/uc?id=1kkdfrCjTjyzjqYfX7C9vDSdfod5HBYQS",
        "pagamentos": "https://drive.google.com/uc?id=1j2RW-ryt4H3iX8nkaw371yBI0_EGA8MY"
    }
    clientes = pd.read_csv(urls["clientes"])
    churn = pd.read_csv(urls["churn"])
    pagamentos = pd.read_csv(urls["pagamentos"])
    return {"clientes": clientes, "churn": churn, "pagamentos": pagamentos}

def preprocessar_clientes(clientes):
    clientes["data_nascimento"] = pd.to_datetime(clientes["data_nascimento"], errors='coerce')
    clientes["idade"] = pd.to_datetime("today").year - clientes["data_nascimento"].dt.year
    return clientes

def tela_dataviz(dfs):
    st.title("📊 Histórico de Churn")
    clientes = preprocessar_clientes(dfs["clientes"])
    churn = dfs["churn"]
    pagamentos = dfs["pagamentos"]

    churn["mes_nome"] = pd.to_datetime(churn["data_churn"]).dt.month_name()
    churn["mes_churn"] = churn["mes_plano_cancelado"]
    churn = churn.merge(clientes[["user_id", "plano_duracao_meses"]], on="user_id", how="left")
    churn["meses_restantes"] = churn["plano_duracao_meses"] - churn["mes_churn"]
    churn = churn[churn["mes_churn"] >= 0]

    st.subheader("Matriz de Quantidade de Alunos")
    matriz_qtd = churn.pivot_table(index="mes_nome", columns="mes_churn", values="user_id", aggfunc="count", fill_value=0)
    st.dataframe(matriz_qtd.style.background_gradient(axis=None, cmap="Reds"))

    st.subheader("Matriz de Receita Perdida por Mês e Período do Plano (R$)")
    pagamentos_ult = pagamentos.sort_values("data_pagamento").drop_duplicates("user_id", keep="last")
    merged = churn.merge(pagamentos_ult[["user_id", "valor_mensalidade"]], on="user_id", how="left")
    matriz_receita = merged.pivot_table(index="mes_nome", columns="mes_churn", values="valor_mensalidade", aggfunc="sum", fill_value=0)
    st.dataframe(matriz_receita.style.background_gradient(axis=None, cmap="Oranges"))

def tela_churn_score(dfs):
    st.title("📉 Score de Propensão ao Churn Mensal")
    clientes = preprocessar_clientes(dfs["clientes"])
    pagamentos = dfs["pagamentos"]
    churn = dfs["churn"]

    st.button("Processar Score")

    df = clientes.merge(pagamentos.sort_values("data_pagamento").drop_duplicates("user_id", keep="last"), on="user_id", how="left")
    df = df[df["user_id"].isin(pagamentos["user_id"])]

    df = df.dropna(subset=["idade", "estado", "valor_mensalidade"])
    X = pd.get_dummies(df[["idade", "estado", "valor_mensalidade"]], drop_first=True)

    model = RandomForestClassifier()
    y = np.random.choice([0, 1], size=len(X))  # Simulação de rótulo
    model.fit(X, y)
    score = model.predict_proba(X)[:, 1]

    df["score"] = MinMaxScaler().fit_transform(score.reshape(-1, 1))
    st.download_button("⬇️ Baixar Scores CSV", df[["user_id", "score"]].to_csv(index=False), "scores_churn.csv", "text/csv")
    st.bar_chart(df["score"])

def tela_pov(dfs):
    st.title("🧪 Prova de Valor")
    st.write("Em desenvolvimento")

def tela_politica_churn(dfs):
    st.title("📌 Política de Churn")
    st.write("Em desenvolvimento")

def tela_perfis_churn(dfs):
    st.title("🧬 Perfis de Churn por Mês")
    st.write("Em desenvolvimento")
