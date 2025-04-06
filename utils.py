
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
import requests
from io import BytesIO
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


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
    url_clientes = "https://drive.google.com/file/d/1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT/view?usp=drive_link"
    url_churn = "https://drive.google.com/file/d/1kkdfrCjTjyzjqYfX7C9vDSdfod5HBYQS/view?usp=drive_link"
    url_pagamentos = "https://drive.google.com/file/d/1j2RW-ryt4H3iX8nkaw371yBI0_EGA8MY/view?usp=drive_link"
    clientes = baixar_csv_do_drive(url_clientes)
    churn = baixar_csv_do_drive(url_churn)
    pagamentos = baixar_csv_do_drive(url_pagamentos)
    return {"clientes": clientes, "churn": churn, "pagamentos": pagamentos}


def barra_progresso(texto):
    barra = st.progress(0, text=texto)
    for i in range(1, 101):
        barra.progress(i, text=texto)
        time.sleep(0.005)
    barra.empty()


def gerar_csv_download(df, nome):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Baixar CSV", csv, f"{nome}.csv", "text/csv", use_container_width=True)


def tela_dataviz(dfs):
    st.title("Histórico de Churn")
    clientes = dfs["clientes"]
    churn = dfs["churn"]

    churn["mes_nome"] = churn["mes_calendario_churn"].apply(lambda x: calendar.month_name[int(x)])
    churn["mes_nome"] = pd.Categorical(churn["mes_nome"], categories=list(calendar.month_name)[1:], ordered=True)

    st.subheader("📊 Quantidade de Alunos por Mês e Período do Plano")
    filtro_mes = st.selectbox("Filtrar por mês", ["Todos os meses"] + list(calendar.month_name)[1:])
    if filtro_mes != "Todos os meses":
        churn = churn[churn["mes_nome"] == filtro_mes]

    churn["mes_churn"] = churn["mes_churn"].astype(int)
    matriz_qtd = churn.groupby(["mes_nome", "mes_churn"]).size().unstack(fill_value=0).sort_index()
    st.dataframe(matriz_qtd)

    st.subheader("💸 Receita Perdida por Mês e Período do Plano (R$)")
    pagamentos = dfs["pagamentos"]
    pagamentos["data_prevista_pagamento"] = pd.to_datetime(pagamentos["data_prevista_pagamento"], errors="coerce")
    ultima = pagamentos.dropna().sort_values("data_prevista_pagamento").drop_duplicates("user_id", keep="last")
    ultima = ultima[["user_id", "valor_mensalidade"]]

    merged = churn.merge(ultima, on="user_id", how="left")
    merged["perda"] = merged["valor_mensalidade"]
    matriz_receita = merged.groupby(["mes_nome", "mes_churn"])["perda"].sum().unstack(fill_value=0).sort_index()
    st.dataframe(matriz_receita)


def tela_churn_score(dfs):
    st.title("Score de Propensão ao Churn Mensal")

    if st.button("Processar Score"):
        barra_progresso("Processando score de alunos...")

        clientes = dfs["clientes"].copy()
        churn = dfs["churn"].copy()

        clientes["user_id"] = clientes["user_id"].astype(str)
        churn["user_id"] = churn["user_id"].astype(str)

        clientes["data_nascimento"] = pd.to_datetime(clientes["data_nascimento"], errors="coerce")
        clientes = clientes.dropna(subset=["data_nascimento"])
        clientes["idade"] = pd.to_datetime("today").year - clientes["data_nascimento"].dt.year

        churn["target"] = churn["mes_churn"].apply(lambda x: 1 if x == 1 else 0)
        base_modelo = churn.merge(clientes, on="user_id", how="left")
        base_modelo = base_modelo.dropna(subset=["idade", "plano_duracao_meses"])

        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(base_modelo[["idade", "plano_duracao_meses"]], base_modelo["target"])

        ativos = clientes.copy()
        ativos = ativos.dropna(subset=["idade", "plano_duracao_meses", "valor_mensalidade"])
        X_ativos = ativos[["idade", "plano_duracao_meses"]]

        ativos["score_churn"] = modelo.predict_proba(X_ativos)[:, 1]
        ativos["score_churn"] = MinMaxScaler().fit_transform(ativos[["score_churn"]])
        ativos["score_churn"] = ativos["score_churn"].round(3)

        score_median = ativos["score_churn"].median()
        em_risco = ativos[ativos["score_churn"] >= score_median]

        st.metric("Alunos em Risco", len(em_risco))
        st.metric("Receita em Risco (R$)", f"{em_risco['valor_mensalidade'].sum():,.2f}")
        st.dataframe(em_risco[["user_id", "nome", "idade", "score_churn", "valor_mensalidade"]])
        gerar_csv_download(em_risco, "alunos_com_risco_churn")


def tela_pov(dfs):
    st.title("Prova de Valor")
    percentual = st.selectbox("Percentual de Recuperação", list(range(5, 105, 5)), index=19)

    if st.button("Backtest"):
        barra_progresso("Executando backtest de recuperação...")

        churn = dfs["churn"].copy()
        pagamentos = dfs["pagamentos"].copy()

        pagamentos["user_id"] = pagamentos["user_id"].astype(str)
        churn["user_id"] = churn["user_id"].astype(str)

        pagamentos["data_prevista_pagamento"] = pd.to_datetime(pagamentos["data_prevista_pagamento"], errors="coerce")
        pagamentos = pagamentos.dropna(subset=["data_prevista_pagamento", "valor_mensalidade"])
        ultima = pagamentos.sort_values("data_prevista_pagamento").drop_duplicates("user_id", keep="last")
        ultima = ultima[["user_id", "valor_mensalidade"]]

        merged = churn.merge(ultima, on="user_id", how="left")
        merged = merged.dropna(subset=["valor_mensalidade", "mes_churn", "plano_duracao_meses", "mes_calendario_churn"])
        merged["meses_restantes"] = merged["plano_duracao_meses"] - merged["mes_churn"]
        merged["meses_restantes"] = merged["meses_restantes"].clip(lower=0)
        merged["recuperavel"] = merged["meses_restantes"] * merged["valor_mensalidade"]
        merged["mes_nome"] = merged["mes_calendario_churn"].apply(lambda x: calendar.month_name[int(x)])

        agrupado = merged.groupby("mes_nome")["recuperavel"].sum().reindex(list(calendar.month_name)[1:], fill_value=0)
        recuperado = agrupado * (percentual / 100)

        df_plot = pd.DataFrame({"Mês": recuperado.index, "Recuperado": recuperado.values})
        st.subheader("💰 Receita Recuperada por Mês (R$)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_plot, x="Recuperado", y="Mês", palette="YlGnBu", ax=ax)
        for i, val in enumerate(recuperado):
            ax.text(val, i, f"R$ {val:,.0f}", va="center")
        st.pyplot(fig)
        st.metric("Total de Receita Recuperável", f"R$ {recuperado.sum():,.2f}")


def tela_politica_churn(dfs):
    st.title("Política de Churn")

    churn = dfs["churn"].copy()
    churn["score_simulado"] = np.random.uniform(0.3, 0.9, size=len(churn))

    media = churn.groupby("mes_churn")["score_simulado"].mean().reset_index()

    st.subheader("📌 Score Médio por Período de Curso")
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(data=media, x="mes_churn", y="score_simulado", palette="coolwarm", ax=ax)
    ax.set_ylabel("Score Médio de Churn")
    ax.set_xlabel("Mês do Plano")
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 0.01, f"{p.get_height():.2f}", ha="center", fontsize=8)
    st.pyplot(fig)


def tela_perfis_churn(dfs):
    st.title("Perfis de Churn por Mês")

    churn = dfs["churn"].copy()
    clientes = dfs["clientes"].copy()

    clientes["data_nascimento"] = pd.to_datetime(clientes["data_nascimento"], errors="coerce")
    clientes = clientes.dropna(subset=["data_nascimento"])
    clientes["idade"] = pd.to_datetime("today").year - clientes["data_nascimento"].dt.year

    meses = list(calendar.month_name)[1:]
    mes_nome = st.selectbox("Selecione o mês para análise", meses)
    mes_num = meses.index(mes_nome) + 1

    base = churn[churn["mes_calendario_churn"] == mes_num].merge(clientes, on="user_id", how="left")
    base = base.dropna(subset=["idade", "estado", "tipo_plano", "canal"])

    X = pd.get_dummies(base[["idade", "estado", "tipo_plano", "canal"]])
    modelo = KMeans(n_clusters=3, random_state=42, n_init=10)
    base["cluster"] = modelo.fit_predict(X)

    st.subheader("🔍 Clusters de Alunos Desistentes")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=base, x="idade", y="cluster", hue="estado", palette="tab10", ax=ax)
    st.pyplot(fig)

    gerar_csv_download(base, f"perfis_churn_{mes_nome.lower()}")
