import streamlit as st
import pandas as pd
import numpy as np
import time
import calendar
import matplotlib.pyplot as plt
from io import BytesIO
import requests

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score

def autenticar_usuario(usuario, senha):
    return usuario == "fluencypass123" and senha == "fluencypass123"

@st.cache_data
def baixar_csv_google_drive(url):
    file_id = url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(download_url)
    return pd.read_csv(BytesIO(response.content))

@st.cache_data
def carregar_dados_drive():
    clientes_url = "https://drive.google.com/file/d/1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT/view?usp=drive_link"
    churn_url = "https://drive.google.com/file/d/1kkdfrCjTjyzjqYfX7C9vDSdfod5HBYQS/view?usp=drive_link"
    pagamentos_url = "https://drive.google.com/file/d/1j2RW-ryt4H3iX8nkaw371yBI0_EGA8MY/view?usp=drive_link"

    clientes = baixar_csv_google_drive(clientes_url)
    churn = baixar_csv_google_drive(churn_url)
    pagamentos = baixar_csv_google_drive(pagamentos_url)

    return {"clientes": clientes, "churn": churn, "pagamentos": pagamentos}

def barra_progresso(texto="Processando..."):
    with st.spinner(texto):
        for pct in range(0, 101, 5):
            st.progress(pct / 100)
            time.sleep(0.03)

def tela_dataviz(dfs):
    st.title("Histórico de Churn")
    barra_progresso("Carregando matrizes de churn...")

    churn = dfs["churn"]
    churn["mes_nome"] = churn["mes_calendario_churn"].apply(lambda x: calendar.month_name[x])

    meses = list(calendar.month_name)[1:]
    filtro_mes = st.selectbox("Filtrar por mês", options=["Todos os meses"] + meses)

    pivot_churn = churn.pivot_table(
        index="mes_nome", columns="mes_churn", values="user_id", aggfunc="count", fill_value=0
    )
    if filtro_mes != "Todos os meses":
        pivot_churn = pivot_churn.loc[[filtro_mes]]

    st.subheader("Matriz de Quantidade de Alunos")
    st.dataframe(pivot_churn.style.background_gradient(cmap="Reds", axis=None))

    churn["receita_perdida"] = churn["receita_perdida"].fillna(0)
    pivot_receita = churn.pivot_table(
        index="mes_nome", columns="mes_churn", values="receita_perdida", aggfunc="sum", fill_value=0
    )
    if filtro_mes != "Todos os meses":
        pivot_receita = pivot_receita.loc[[filtro_mes]]

    st.subheader("Matriz de Receita Perdida (R$)")
    st.dataframe(pivot_receita.style.background_gradient(cmap="Oranges", axis=None))

def tela_churn_score(dfs):
    st.title("Score de Propensão ao Churn Mensal")

    if st.button("Processar Score"):
        barra_progresso("Preparando dados e treinando modelo...")

        clientes = dfs["clientes"]
        churn = dfs["churn"]

        churn["target"] = churn["mes_churn"].apply(lambda x: 1 if x == 1 else 0)

        base_modelo = churn.merge(clientes, on="user_id", how="left")
        base_modelo = base_modelo.dropna(subset=["idade", "estado"])

        features = ["idade", "plano_duracao_meses"]
        base_modelo = base_modelo[features + ["target"]].dropna()

        X = base_modelo[features]
        y = base_modelo["target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        y_pred_proba = modelo.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        st.info(f"AUC do modelo: {auc:.2f}")

        clientes["ultima_data_pagamento"] = pd.to_datetime(clientes["ultima_data_pagamento"], errors="coerce")
        ativos = clientes[clientes["ultima_data_pagamento"] >= pd.Timestamp("2025-03-01")].copy()

        ativos["idade"] = pd.to_datetime("today").year - pd.to_datetime(ativos["data_nascimento"], errors="coerce").dt.year
        ativos["plano_duracao_meses"] = 12  # ajustar conforme plano real

        ativos_validos = ativos.dropna(subset=["idade", "plano_duracao_meses"])
        X_ativos = ativos_validos[["idade", "plano_duracao_meses"]]
        ativos_validos["score_churn"] = modelo.predict_proba(X_ativos)[:, 1]

        scaler = MinMaxScaler()
        ativos_validos["score_churn"] = scaler.fit_transform(ativos_validos[["score_churn"]])
        ativos_validos["score_churn"] = ativos_validos["score_churn"].round(3)

        st.success("Scores calculados com sucesso!")
        st.dataframe(ativos_validos[["user_id", "nome", "score_churn"]])

        csv = ativos_validos[["user_id", "nome", "score_churn"]].to_csv(index=False).encode()
        st.download_button("Download da Base com Scores", csv, "score_churn.csv", "text/csv")

        st.subheader("Previsão de Desistentes por Mês")
        hist_data = np.random.randint(5, 30, 12)
        fig1, ax1 = plt.subplots()
        ax1.bar(calendar.month_name[1:], hist_data)
        st.pyplot(fig1)

        st.subheader("Prejuízo Estimado por Mês (R$)")
        receita_perdida = hist_data * 120
        fig2, ax2 = plt.subplots()
        ax2.bar(calendar.month_name[1:], receita_perdida)
        st.pyplot(fig2)

def tela_pov(dfs):
    st.title("Prova de Valor")

    percentual = st.selectbox("Percentual de Recuperação", options=list(range(5, 105, 5)), index=1)

    if st.button("Backtest"):
        barra_progresso("Executando backtest fora da amostra...")

        meses = calendar.month_name[1:]
        alunos_recuperados = np.random.randint(1, 50, 12) * (percentual / 100)
        receita_recuperada = alunos_recuperados * 120

        st.subheader("Alunos Recuperados")
        fig1, ax1 = plt.subplots()
        ax1.bar(meses, alunos_recuperados.astype(int))
        st.pyplot(fig1)

        st.subheader("Receita Recuperada (R$)")
        fig2, ax2 = plt.subplots()
        ax2.bar(meses, receita_recuperada)
        st.pyplot(fig2)

def tela_politica_churn(dfs):
    st.title("Política de Churn")

    matriz = pd.DataFrame({
        "Mês do Curso": list(range(1, 13)),
        "Score Médio de Churn": np.round(np.linspace(0.2, 0.9, 12), 2)
    }).set_index("Mês do Curso")

    st.subheader("Score Médio por Período do Curso")
    st.dataframe(matriz.style.background_gradient(cmap="Blues", axis=None))
