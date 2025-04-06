
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
    pagamentos = dfs["pagamentos"]

    churn["mes_nome"] = churn["mes_calendario_churn"].apply(lambda x: calendar.month_name[int(x)])
    churn["mes_nome"] = pd.Categorical(churn["mes_nome"], categories=list(calendar.month_name)[1:], ordered=True)

    st.subheader("📊 Matriz de Quantidade de Alunos")
    filtro_mes = st.selectbox("Filtrar por mês", ["Todos os meses"] + list(calendar.month_name)[1:])
    if filtro_mes != "Todos os meses":
        churn = churn[churn["mes_nome"] == filtro_mes]

    churn = churn[churn["mes_churn"] >= 0]  # remover negativos
    matriz_qtd = churn.groupby(["mes_nome", "mes_churn"]).size().unstack(fill_value=0).sort_index()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(matriz_qtd, cmap="Reds", annot=True, fmt="d", ax=ax)
    ax.set_title("Quantidade de Alunos por Mês e Período")
    st.pyplot(fig)

    st.subheader("💸 Receita Perdida por Mês e Período do Plano (R$)")
    pagamentos["data_prevista_pagamento"] = pd.to_datetime(pagamentos["data_prevista_pagamento"], errors="coerce")
    ultima = pagamentos.dropna().sort_values("data_prevista_pagamento").drop_duplicates("user_id", keep="last")
    ultima = ultima[["user_id", "valor_mensalidade"]]

    merged = churn.merge(clientes[["user_id", "plano_duracao_meses"]], on="user_id", how="left")
    merged = merged.merge(ultima, on="user_id", how="left")
    merged["meses_restantes"] = merged["plano_duracao_meses"] - merged["mes_churn"]
    merged["meses_restantes"] = merged["meses_restantes"].clip(lower=0)
    merged["perda"] = merged["meses_restantes"] * merged["valor_mensalidade"]
    merged = merged[merged["mes_churn"] >= 0]

    receita_matriz = merged.groupby(["mes_nome", "mes_churn"])["perda"].sum().unstack(fill_value=0).sort_index()
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    sns.heatmap(receita_matriz, cmap="Oranges", annot=True, fmt=".0f", ax=ax2)
    ax2.set_title("Receita Perdida (R$) por Mês e Período do Plano")
    st.pyplot(fig2)


def tela_churn_score(dfs):
    st.title("Score de Propensão ao Churn Mensal")

    if st.button("Processar Score"):
        barra_progresso("Gerando Score para Alunos Ativos...")

        clientes = dfs["clientes"].copy()
        pagamentos = dfs["pagamentos"]
        churn = dfs["churn"]

        clientes["data_nascimento"] = pd.to_datetime(clientes["data_nascimento"], errors="coerce")
        clientes = clientes.dropna(subset=["data_nascimento"])
        clientes["idade"] = pd.to_datetime("today").year - clientes["data_nascimento"].dt.year

        pagamentos["data_prevista_pagamento"] = pd.to_datetime(pagamentos["data_prevista_pagamento"], errors="coerce")
        ultima_pg = pagamentos.sort_values("data_prevista_pagamento").drop_duplicates("user_id", keep="last")
        ativos = clientes[clientes["user_id"].isin(ultima_pg["user_id"])]

        churn["target"] = churn["mes_churn"].apply(lambda x: 1 if x == 1 else 0)
        base_treino = churn.merge(clientes[["user_id", "idade", "plano_duracao_meses"]], on="user_id", how="left")
        base_treino = base_treino.dropna(subset=["idade", "plano_duracao_meses"])

        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(base_treino[["idade", "plano_duracao_meses"]], base_treino["target"])

        ativos = ativos.dropna(subset=["idade", "plano_duracao_meses"])
        ativos["score_churn"] = modelo.predict_proba(ativos[["idade", "plano_duracao_meses"]])[:, 1]
        ativos["score_churn"] = MinMaxScaler().fit_transform(ativos[["score_churn"]])
        ativos["score_churn"] = ativos["score_churn"].round(3)
        ativos["mes_previsto"] = pd.to_datetime("today").month + 1

        st.metric("Alunos Escorados", len(ativos))
        st.dataframe(ativos[["user_id", "nome", "idade", "score_churn", "mes_previsto"]])
        gerar_csv_download(ativos[["user_id", "nome", "score_churn", "mes_previsto"]], "score_churn_alunos")


def tela_pov(dfs):
    st.title("Prova de Valor")
    percentual = st.selectbox("Percentual de Recuperação", list(range(5, 105, 5)), index=19)

    if st.button("Backtest"):
        barra_progresso("Executando backtest...")

        churn = dfs["churn"]
        pagamentos = dfs["pagamentos"]
        clientes = dfs["clientes"]

        pagamentos["data_prevista_pagamento"] = pd.to_datetime(pagamentos["data_prevista_pagamento"], errors="coerce")
        ultima_pg = pagamentos.sort_values("data_prevista_pagamento").drop_duplicates("user_id", keep="last")
        ultima_pg = ultima_pg[["user_id", "valor_mensalidade"]]

        df = churn.merge(ultima_pg, on="user_id", how="left")
        df = df.merge(clientes[["user_id", "plano_duracao_meses"]], on="user_id", how="left")
        df["meses_restantes"] = df["plano_duracao_meses"] - df["mes_churn"]
        df["meses_restantes"] = df["meses_restantes"].clip(lower=0)
        df["recuperavel"] = df["meses_restantes"] * df["valor_mensalidade"]
        df["mes_nome"] = df["mes_calendario_churn"].apply(lambda x: calendar.month_name[int(x)])

        resultado = df.groupby("mes_nome")["recuperavel"].sum().reindex(list(calendar.month_name)[1:], fill_value=0)
        recuperado = resultado * (percentual / 100)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=recuperado.values, y=recuperado.index, palette="BuGn", ax=ax)
        for i, val in enumerate(recuperado):
            ax.text(val, i, f"R$ {val:,.0f}", va="center")
        ax.set_xlabel("Receita Recuperada")
        st.pyplot(fig)
        st.metric("Total de Receita Recuperada", f"R$ {recuperado.sum():,.2f}")


def tela_perfis_churn(dfs):
    st.title("Perfis de Churn por Mês")

    churn = dfs["churn"]
    clientes = dfs["clientes"]

    clientes["data_nascimento"] = pd.to_datetime(clientes["data_nascimento"], errors="coerce")
    clientes["idade"] = pd.to_datetime("today").year - clientes["data_nascimento"].dt.year
    clientes = clientes.dropna(subset=["idade"])

    mes_nome = st.selectbox("Selecione o mês para análise", list(calendar.month_name)[1:])
    mes_num = list(calendar.month_name).index(mes_nome)

    base = churn[churn["mes_calendario_churn"] == mes_num].merge(clientes, on="user_id", how="left")
    base = base.dropna(subset=["idade", "estado", "tipo_plano", "canal"])

    X = pd.get_dummies(base[["idade", "estado", "tipo_plano", "canal"]])
    modelo = KMeans(n_clusters=3, n_init=10, random_state=42)
    base["cluster"] = modelo.fit_predict(X)

    st.subheader("🎯 Grupos de Desistência por Perfil")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=base, x="idade", y="cluster", hue="tipo_plano", ax=ax)
    st.pyplot(fig)
    gerar_csv_download(base, f"perfis_churn_{mes_nome.lower()}")
