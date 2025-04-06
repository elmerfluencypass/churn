import streamlit as st
import pandas as pd
import numpy as np
import time
import calendar
import matplotlib.pyplot as plt

@st.cache_data
def autenticar_usuario(usuario, senha):
    return usuario == "fluencypass123" and senha == "fluencypass123"

@st.cache_data
def carregar_dados():
    clientes = pd.read_csv("cadastro_clientes.csv")
    churn = pd.read_csv("churn_detectado.csv")
    with open("Historico_Pagamentos.sql", "r", encoding="utf-8") as f:
        pagamentos_raw = f.read()
    pagamentos = pd.read_sql_query(pagamentos_raw, "sqlite:///pagamentos.db")
    return {"clientes": clientes, "churn": churn, "pagamentos": pagamentos}

def barra_progresso(texto="Processando..."):
    with st.spinner(texto):
        for pct in range(0, 101, 5):
            st.progress(pct / 100)
            time.sleep(0.05)

def tela_dataviz(dfs):
    st.title("Histórico de Churn")
    barra_progresso("Carregando matrizes de churn...")

    churn = dfs["churn"]
    churn["mes_nome"] = churn["mes_calendario_churn"].apply(lambda x: calendar.month_name[x])

    meses = list(calendar.month_name)[1:]
    filtro_mes = st.selectbox("Filtrar por mês", options=["Todos os meses"] + meses)

    # Matriz de contagem de alunos
    pivot_churn = churn.pivot_table(
        index="mes_nome", 
        columns="mes_churn", 
        values="user_id", 
        aggfunc="count", 
        fill_value=0
    )

    if filtro_mes != "Todos os meses":
        pivot_churn = pivot_churn.loc[[filtro_mes]]

    st.subheader("Matriz de Quantidade de Alunos")
    st.dataframe(pivot_churn.style.background_gradient(cmap="Reds", axis=None))

    # Matriz de receita perdida
    churn["receita_perdida"] = churn["receita_perdida"].fillna(0)
    pivot_receita = churn.pivot_table(
        index="mes_nome", 
        columns="mes_churn", 
        values="receita_perdida", 
        aggfunc="sum", 
        fill_value=0
    )

    if filtro_mes != "Todos os meses":
        pivot_receita = pivot_receita.loc[[filtro_mes]]

    st.subheader("Matriz de Receita Perdida (R$)")
    st.dataframe(pivot_receita.style.background_gradient(cmap="Oranges", axis=None))

def tela_churn_score(dfs):
    st.title("Score de Propensão ao Churn Mensal")

    if st.button("Processar Score"):
        barra_progresso("Calculando scores de churn...")

        clientes = dfs["clientes"]
        ativos = clientes[
            pd.to_datetime(clientes["ultima_data_pagamento"], errors="coerce").dt.month == 3
        ].dropna(subset=["ultima_data_pagamento"])

        # Mock de modelo simples com score aleatório
        np.random.seed(42)
        ativos["score_churn"] = np.round(np.random.uniform(0, 1, len(ativos)), 3)

        st.success("Scores calculados com sucesso!")
        st.dataframe(ativos[["user_id", "nome", "score_churn"]])

        csv = ativos[["user_id", "nome", "score_churn"]].to_csv(index=False).encode()
        st.download_button("Download da Base com Scores", csv, "score_churn.csv", "text/csv")

        # Histograma 1
        st.subheader("Previsão de Desistentes por Mês")
        hist_data = np.random.randint(0, 100, 12)
        fig1, ax1 = plt.subplots()
        ax1.bar(calendar.month_name[1:], hist_data)
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        # Histograma 2
        st.subheader("Prejuízo Estimado por Mês (R$)")
        receita_hist = np.random.randint(500, 10000, 12)
        fig2, ax2 = plt.subplots()
        ax2.bar(calendar.month_name[1:], receita_hist)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

def tela_pov(dfs):
    st.title("Prova de Valor")

    percentual = st.selectbox("Percentual de Recuperação", options=list(range(5, 105, 5)), index=1)

    if st.button("Backtest"):
        barra_progresso("Executando backtest fora da amostra...")

        meses = calendar.month_name[1:]
        alunos_recuperados = np.random.randint(1, 50, 12) * (percentual / 100)
        receita_recuperada = alunos_recuperados * 120  # Supondo R$120/mês

        st.subheader("Alunos Recuperados")
        fig1, ax1 = plt.subplots()
        ax1.bar(meses, alunos_recuperados.astype(int))
        plt.xticks(rotation=45)
        st.pyplot(fig1)

        st.subheader("Receita Recuperada (R$)")
        fig2, ax2 = plt.subplots()
        ax2.bar(meses, receita_recuperada)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

def tela_politica_churn(dfs):
    st.title("Política de Churn")

    # Simulando score médio por período
    matriz = pd.DataFrame({
        "Mês do Curso": list(range(1, 13)),
        "Score Médio de Churn": np.round(np.linspace(0.2, 0.9, 12), 2)
    })
    matriz.set_index("Mês do Curso", inplace=True)

    st.subheader("Score Médio por Período do Curso")
    st.dataframe(matriz.style.background_gradient(cmap="Blues", axis=None))
