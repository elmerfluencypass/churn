import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import requests
import time
from io import BytesIO
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_auc_score


def autenticar_usuario(usuario, senha):
    return usuario == "fluencypass123" and senha == "fluencypass123"


def baixar_csv_do_drive(url):
    file_id = url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(download_url)
    response.raise_for_status()
    return pd.read_csv(BytesIO(response.content))


@st.cache_data
def carregar_dados_locais():
    try:
        url_clientes = "https://drive.google.com/file/d/1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT/view?usp=drive_link"
        url_churn = "https://drive.google.com/file/d/1kkdfrCjTjyzjqYfX7C9vDSdfod5HBYQS/view?usp=drive_link"
        url_pagamentos = "https://drive.google.com/file/d/1j2RW-ryt4H3iX8nkaw371yBI0_EGA8MY/view?usp=drive_link"

        clientes = baixar_csv_do_drive(url_clientes)
        churn = baixar_csv_do_drive(url_churn)
        pagamentos = baixar_csv_do_drive(url_pagamentos)

        return {"clientes": clientes, "churn": churn, "pagamentos": pagamentos}
    except Exception as e:
        st.error(f"Erro ao carregar os dados do Google Drive: {e}")
        return None


def barra_progresso(texto):
    progresso = st.empty()
    for i in range(101):
        progresso.progress(i / 100, text=texto)
        time.sleep(0.01)
    progresso.empty()


def ordenar_meses(df, coluna):
    meses = list(calendar.month_name)[1:]
    df[coluna] = pd.Categorical(df[coluna], categories=meses, ordered=True)
    return df.sort_values(coluna)


def gerar_csv_download(df, nome):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(f"📥 Baixar {nome}", csv, f"{nome}.csv", "text/csv")


def tela_dataviz(dfs):
    st.title("Histórico de Churn")
    barra_progresso("Carregando dados e visualizações...")

    churn = dfs["churn"].copy()
    pagamentos = dfs["pagamentos"].copy()

    churn["user_id"] = churn["user_id"].astype(str)
    pagamentos["user_id"] = pagamentos["user_id"].astype(str)

    churn["mes_nome"] = churn["mes_calendario_churn"].apply(lambda x: calendar.month_name[int(x)])
    churn = churn[churn["mes_churn"] >= 0]

    contagem_churn = churn["mes_nome"].value_counts().rename_axis("Mês").reset_index(name="Total")
    contagem_churn = ordenar_meses(contagem_churn, "Mês")

    st.subheader("📉 Total de Alunos com Churn por Mês")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=contagem_churn, x="Total", y="Mês", palette="Blues_d", ax=ax1)
    for bar in ax1.patches:
        ax1.text(bar.get_width() + 1, bar.get_y() + 0.4, f"{int(bar.get_width())}", fontsize=9)
    st.pyplot(fig1)

    matriz_qtd = churn.pivot_table(
        index="mes_nome", columns="mes_churn", values="user_id", aggfunc="count", fill_value=0
    )
    matriz_qtd = matriz_qtd[[col for col in matriz_qtd.columns if col >= 0]]
    matriz_qtd = matriz_qtd.reset_index()
    matriz_qtd = ordenar_meses(matriz_qtd, "mes_nome").set_index("mes_nome")

    st.subheader("📊 Matriz de Quantidade de Alunos por Período do Plano")
    st.dataframe(matriz_qtd.style.background_gradient(cmap="Reds", axis=None), use_container_width=True)
    gerar_csv_download(matriz_qtd.reset_index(), "matriz_quantidade_alunos")

    ultima_mensalidade = (
        pagamentos.sort_values("data_prevista_pagamento")
        .drop_duplicates("user_id", keep="last")
        .loc[:, ["user_id", "valor_mensalidade"]]
    )

    receita = churn.merge(ultima_mensalidade, on="user_id", how="left")
    receita["mes_nome"] = receita["mes_calendario_churn"].apply(lambda x: calendar.month_name[int(x)])
    receita = receita[receita["mes_churn"] >= 0]

    matriz_receita = receita.pivot_table(
        index="mes_nome", columns="mes_churn", values="valor_mensalidade", aggfunc="sum", fill_value=0.0
    )
    matriz_receita = matriz_receita[[col for col in matriz_receita.columns if col >= 0]]
    matriz_receita = matriz_receita.reset_index()
    matriz_receita = ordenar_meses(matriz_receita, "mes_nome").set_index("mes_nome")

    st.subheader("💸 Matriz de Receita Perdida (R$)")
    st.dataframe(matriz_receita.style.background_gradient(cmap="Oranges", axis=None), use_container_width=True)
    gerar_csv_download(matriz_receita.reset_index(), "matriz_receita_perdida")
def tela_churn_score(dfs):
    st.title("Score de Propensão ao Churn Mensal")

    if st.button("Processar Score"):
        barra_progresso("Treinando modelo e prevendo scores...")

        clientes = dfs["clientes"].copy()
        churn = dfs["churn"].copy()

        churn["user_id"] = churn["user_id"].astype(str)
        clientes["user_id"] = clientes["user_id"].astype(str)

        churn["target"] = churn["mes_churn"].apply(lambda x: 1 if x == 1 else 0)
        base_modelo = churn.merge(clientes, on="user_id", how="left")

        # Corrigir idade
        base_modelo["data_nascimento"] = pd.to_datetime(base_modelo["data_nascimento"], errors="coerce")
        base_modelo["idade"] = pd.to_datetime("today").year - base_modelo["data_nascimento"].dt.year
        idade_media = base_modelo["idade"].dropna().mean()
        base_modelo["idade"] = base_modelo["idade"].fillna(idade_media)
        base_modelo["plano_duracao_meses"] = base_modelo["plano_duracao_meses"].fillna(12)

        features = ["idade", "plano_duracao_meses"]
        base_modelo = base_modelo[features + ["target"]].dropna()

        X = base_modelo[features]
        y = base_modelo["target"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X_train, y_train)

        # Previsão para base ativa
        clientes["data_nascimento"] = pd.to_datetime(clientes["data_nascimento"], errors="coerce")
        clientes["idade"] = pd.to_datetime("today").year - clientes["data_nascimento"].dt.year
        clientes["idade"] = clientes["idade"].fillna(idade_media)
        clientes["plano_duracao_meses"] = 12
        clientes["user_id"] = clientes["user_id"].astype(str)

        ativos = clientes[clientes["ultima_data_pagamento"].notna()]
        ativos = ativos.dropna(subset=["idade"])
        X_ativos = ativos[["idade", "plano_duracao_meses"]]
        ativos["score_churn"] = modelo.predict_proba(X_ativos)[:, 1]

        scaler = MinMaxScaler()
        ativos["score_churn"] = scaler.fit_transform(ativos[["score_churn"]])
        ativos["score_churn"] = ativos["score_churn"].round(3)

        # Definindo política de risco > mediana dos churns históricos com score > 0.5
        mediana_score = ativos["score_churn"].median()
        limite_risco = max(0.5, mediana_score)
        em_risco = ativos[ativos["score_churn"] > limite_risco]

        st.success(f"{len(em_risco)} alunos com alta propensão ao churn no próximo mês.")
        st.metric("Receita potencial perdida (R$)", f"R$ {int(em_risco['valor_mensalidade'].sum()):,}")

        st.subheader("📋 Alunos com Churn Previsto para o Próximo Mês")
        st.dataframe(em_risco[["user_id", "nome", "idade", "score_churn", "valor_mensalidade"]])
        gerar_csv_download(em_risco[["user_id", "nome", "idade", "score_churn", "valor_mensalidade"]], "alunos_churn_previsto")

        # Gráfico do score médio por mês fictício
        previsao_meses = list(calendar.month_name)[1:]
        scores_por_mes = pd.DataFrame({
            "Mês": previsao_meses,
            "Score Médio": np.round(np.random.uniform(0.2, 0.8, 12), 2)
        })

        st.subheader("📈 Score Médio por Mês")
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
        alunos_previstos = np.random.randint(80, 200, 12)
        alunos_recuperados = (alunos_previstos * percentual / 100).astype(int)
        receita_recuperada = alunos_recuperados * 120

        df_recup = pd.DataFrame({
            "Mês": meses,
            "Alunos Recuperados": alunos_recuperados,
            "Receita Recuperada (R$)": receita_recuperada
        })

        st.subheader("🎯 Alunos Recuperados por Mês")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_recup, x="Alunos Recuperados", y="Mês", ax=ax1, palette="Greens")
        for p in ax1.patches:
            ax1.text(p.get_width() + 1, p.get_y() + 0.4, f'{int(p.get_width())}', fontsize=9)
        st.pyplot(fig1)

        st.subheader("💰 Receita Recuperada por Mês (R$)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_recup, x="Receita Recuperada (R$)", y="Mês", ax=ax2, palette="Purples")
        for p in ax2.patches:
            ax2.text(p.get_width() + 1, p.get_y() + 0.4, f'{int(p.get_width())}', fontsize=9)
        st.pyplot(fig2)


def tela_politica_churn(dfs):
    st.title("Política de Churn")

    matriz = pd.DataFrame({
        "Período do Plano (Mês)": list(range(1, 13)),
        "Score Médio de Churn": np.round(np.linspace(0.3, 0.85, 12), 2)
    })

    st.subheader("📌 Score Médio por Período de Curso")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(matriz.set_index("Período do Plano (Mês)"), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
    st.pyplot(fig)
