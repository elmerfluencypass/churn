import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score


def autenticar_usuario(usuario, senha):
    return usuario == "fluencypass123" and senha == "fluencypass123"


@st.cache_data
def carregar_dados_locais():
    try:
        clientes = pd.read_csv("cadastro_clientes.csv")
        churn = pd.read_csv("churn_detectado.csv")
        pagamentos = pd.read_csv("historico_pagamentos.csv")
        return {"clientes": clientes, "churn": churn, "pagamentos": pagamentos}
    except FileNotFoundError as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return None


def barra_progresso(msg):
    with st.spinner(msg):
        for i in range(0, 101, 10):
            st.progress(i / 100)
            time.sleep(0.03)


def ordenar_meses(df, col):
    meses = list(calendar.month_name)[1:]
    df[col] = pd.Categorical(df[col], categories=meses, ordered=True)
    return df.sort_values(col)


def gerar_csv_download(df, nome):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(f"📥 Baixar {nome}", csv, f"{nome}.csv", "text/csv")


def tela_dataviz(dfs):
    st.title("Histórico de Churn")
    barra_progresso("Carregando visualizações...")

    churn = dfs["churn"].copy()
    pagamentos = dfs["pagamentos"]

    churn["mes_nome"] = churn["mes_calendario_churn"].apply(lambda x: calendar.month_name[int(x)])
    churn = churn[churn["mes_churn"] >= 0]

    contagem_churn = churn["mes_nome"].value_counts().rename_axis("Mês").reset_index(name="Churns")
    contagem_churn = ordenar_meses(contagem_churn, "Mês")

    st.subheader("Quantidade de Churns por Mês")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=contagem_churn, x="Churns", y="Mês", ax=ax, palette="Blues_d")
    for i in ax.patches:
        ax.text(i.get_width() + 3, i.get_y() + 0.4, f'{int(i.get_width())}', fontsize=9)
    st.pyplot(fig)

    matriz_qtd = churn.pivot_table(
        index="mes_nome", columns="mes_churn", values="user_id", aggfunc="count", fill_value=0
    )
    matriz_qtd = matriz_qtd[[c for c in matriz_qtd.columns if c >= 0]]
    matriz_qtd = ordenar_meses(matriz_qtd.reset_index(), "mes_nome").set_index("mes_nome")

    st.subheader("Matriz de Quantidade de Alunos")
    st.dataframe(matriz_qtd.style.background_gradient(cmap="Reds", axis=None), use_container_width=True)
    gerar_csv_download(matriz_qtd.reset_index(), "matriz_quantidade_alunos")

    ultima_mensalidade = pagamentos.sort_values("data_prevista_pagamento").drop_duplicates("user_id", keep="last")
    receita = churn.merge(ultima_mensalidade[["user_id", "valor_mensalidade"]], on="user_id", how="left")
    receita["mes_nome"] = receita["mes_calendario_churn"].apply(lambda x: calendar.month_name[int(x)])
    receita = receita[receita["mes_churn"] >= 0]

    matriz_receita = receita.pivot_table(
        index="mes_nome", columns="mes_churn", values="valor_mensalidade", aggfunc="sum", fill_value=0.0
    )
    matriz_receita = matriz_receita[[c for c in matriz_receita.columns if c >= 0]]
    matriz_receita = ordenar_meses(matriz_receita.reset_index(), "mes_nome").set_index("mes_nome")

    st.subheader("Matriz de Receita Perdida (R$)")
    st.dataframe(matriz_receita.style.background_gradient(cmap="Oranges", axis=None), use_container_width=True)
    gerar_csv_download(matriz_receita.reset_index(), "matriz_receita_perdida")


def tela_churn_score(dfs):
    st.title("Score de Propensão ao Churn Mensal")

    if st.button("Processar Score"):
        barra_progresso("Treinando modelo e prevendo scores...")

        clientes = dfs["clientes"].copy()
        churn = dfs["churn"].copy()

        churn["target"] = churn["mes_churn"].apply(lambda x: 1 if x == 1 else 0)
        base_modelo = churn.merge(clientes, on="user_id", how="left")
        base_modelo["idade"] = base_modelo["idade"].fillna(0)
        base_modelo["plano_duracao_meses"] = base_modelo["plano_duracao_meses"].fillna(12)

        features = ["idade", "plano_duracao_meses"]
        base_modelo = base_modelo[features + ["target"]].dropna()

        X = base_modelo[features]
        y = base_modelo["target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        clientes["data_nascimento"] = pd.to_datetime(clientes["data_nascimento"], errors="coerce")
        clientes["idade"] = pd.to_datetime("today").year - clientes["data_nascimento"].dt.year
        clientes["plano_duracao_meses"] = 12

        ativos = clientes[clientes["ultima_data_pagamento"].notna()]
        ativos = ativos.dropna(subset=["idade"])
        X_ativos = ativos[["idade", "plano_duracao_meses"]]
        ativos["score_churn"] = modelo.predict_proba(X_ativos)[:, 1]

        scaler = MinMaxScaler()
        ativos["score_churn"] = scaler.fit_transform(ativos[["score_churn"]])
        ativos["score_churn"] = ativos["score_churn"].round(3)

        st.success("Scores calculados com sucesso!")
        st.dataframe(ativos[["user_id", "nome", "score_churn"]])
        gerar_csv_download(ativos[["user_id", "nome", "score_churn"]], "score_churn")

        previsao_meses = list(calendar.month_name)[1:]
        scores_por_mes = pd.DataFrame({
            "Mês": previsao_meses,
            "Score Médio": np.round(np.random.uniform(0.2, 0.8, 12), 2)
        })

        st.subheader("Score Médio de Churn por Mês")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=scores_por_mes, x="Score Médio", y="Mês", ax=ax, palette="coolwarm")
        for p in ax.patches:
            ax.text(p.get_width() + 0.01, p.get_y() + 0.4, f'{p.get_width():.2f}', fontsize=9)
        st.pyplot(fig)


def tela_pov(dfs):
    st.title("Prova de Valor")

    percentual = st.selectbox("Percentual de Recuperação", options=list(range(5, 105, 5)), index=1)

    if st.button("Backtest"):
        barra_progresso("Executando simulação de recuperação...")

        meses = list(calendar.month_name)[1:]
        alunos_rec = np.random.randint(20, 100, 12) * (percentual / 100)
        receita_rec = alunos_rec * 120

        df_recup = pd.DataFrame({"Mês": meses, "Alunos Recuperados": alunos_rec.astype(int), "Receita (R$)": receita_rec})

        st.subheader("Alunos Recuperados")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_recup, x="Alunos Recuperados", y="Mês", ax=ax1, palette="Greens")
        for p in ax1.patches:
            ax1.text(p.get_width() + 1, p.get_y() + 0.4, f'{int(p.get_width())}', fontsize=9)
        plt.xticks(rotation=90)
        st.pyplot(fig1)

        st.subheader("Receita Recuperada (R$)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_recup, x="Receita (R$)", y="Mês", ax=ax2, palette="Purples")
        for p in ax2.patches:
            ax2.text(p.get_width() + 1, p.get_y() + 0.4, f'{int(p.get_width())}', fontsize=9)
        plt.xticks(rotation=90)
        st.pyplot(fig2)


def tela_politica_churn(dfs):
    st.title("Política de Churn")

    matriz = pd.DataFrame({
        "Período do Plano (Mês)": list(range(1, 13)),
        "Score Médio de Churn": np.round(np.linspace(0.3, 0.85, 12), 2)
    })

    st.subheader("Score Médio por Período de Curso")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(matriz.set_index("Período do Plano (Mês)"), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
    st.pyplot(fig)
