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
        time.sleep(0.005)
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

    churn = churn[churn["mes_churn"] >= 0]
    churn["mes_nome"] = churn["mes_calendario_churn"].apply(lambda x: calendar.month_name[int(x)])

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

    # Receita perdida = meses restantes * última mensalidade
    ultima_mensalidade = (
        pagamentos.sort_values("data_prevista_pagamento")
        .drop_duplicates("user_id", keep="last")
        .loc[:, ["user_id", "valor_mensalidade"]]
    )
    merged = churn.merge(ultima_mensalidade, on="user_id", how="left")

    merged["meses_restantes"] = merged["plano_duracao_meses"] - merged["mes_churn"]
    merged["meses_restantes"] = merged["meses_restantes"].clip(lower=0)
    merged["receita_perdida_calc"] = merged["meses_restantes"] * merged["valor_mensalidade"]
    merged["mes_nome"] = merged["mes_calendario_churn"].apply(lambda x: calendar.month_name[int(x)])

    matriz_receita = merged.pivot_table(
        index="mes_nome", columns="mes_churn", values="receita_perdida_calc", aggfunc="sum", fill_value=0
    )
    matriz_receita = matriz_receita.reset_index()
    matriz_receita = ordenar_meses(matriz_receita, "mes_nome").set_index("mes_nome")

    st.subheader("💸 Matriz de Receita Perdida Corrigida (R$)")
    st.dataframe(matriz_receita.style.background_gradient(cmap="Oranges"), use_container_width=True)
    gerar_csv_download(matriz_receita.reset_index(), "matriz_receita_perdida_corrigida")
def tela_churn_score(dfs):
    st.title("Score de Propensão ao Churn Mensal")

    if st.button("Processar Score"):
        barra_progresso("Treinando modelo e prevendo scores...")

        clientes = dfs["clientes"].copy()
        churn = dfs["churn"].copy()

        clientes["user_id"] = clientes["user_id"].astype(str)
        churn["user_id"] = churn["user_id"].astype(str)

        churn["target"] = churn["mes_churn"].apply(lambda x: 1 if x == 1 else 0)
        base_modelo = churn.merge(clientes, on="user_id", how="left")

        base_modelo["data_nascimento"] = pd.to_datetime(base_modelo["data_nascimento"], errors="coerce")
        base_modelo = base_modelo.dropna(subset=["data_nascimento"])
        base_modelo["idade"] = pd.to_datetime("today").year - base_modelo["data_nascimento"].dt.year
        idade_media = base_modelo["idade"].mean()

        X = base_modelo[["idade", "plano_duracao_meses"]]
        y = base_modelo["target"]

        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X, y)

        clientes["data_nascimento"] = pd.to_datetime(clientes["data_nascimento"], errors="coerce")
        clientes = clientes.dropna(subset=["data_nascimento"])
        clientes["idade"] = pd.to_datetime("today").year - clientes["data_nascimento"].dt.year

        ativos = clientes[clientes["ultima_data_pagamento"].notna()].copy()
        ativos["plano_duracao_meses"] = ativos["plano_duracao_meses"].fillna(12)

        X_ativos = ativos[["idade", "plano_duracao_meses"]]
        ativos["score_churn"] = modelo.predict_proba(X_ativos)[:, 1]
        ativos["score_churn"] = MinMaxScaler().fit_transform(ativos[["score_churn"]])
        ativos["score_churn"] = ativos["score_churn"].round(3)

        score_limite = ativos["score_churn"].median()
        em_risco = ativos[ativos["score_churn"] > score_limite]

        st.success(f"{len(em_risco)} alunos com alta propensão ao churn.")
        receita_total = em_risco["valor_mensalidade"].sum()
        st.metric("Receita prevista perdida (R$)", f"R$ {receita_total:,.2f}")
        st.dataframe(em_risco[["user_id", "nome", "idade", "score_churn", "valor_mensalidade"]])
        gerar_csv_download(em_risco, "alunos_churn_previsto")


def tela_pov(dfs):
    st.title("Prova de Valor")

    percentual = st.selectbox("Percentual de Recuperação", options=list(range(5, 105, 5)), index=1)

    if st.button("Backtest"):
        barra_progresso("Executando simulação de recuperação...")

        churn = dfs["churn"].copy()
        pagamentos = dfs["pagamentos"].copy()

        pagamentos["user_id"] = pagamentos["user_id"].astype(str)
        churn["user_id"] = churn["user_id"].astype(str)

        ultima_mensalidade = pagamentos.sort_values("data_prevista_pagamento") \
                                       .drop_duplicates("user_id", keep="last") \
                                       [["user_id", "valor_mensalidade"]]

        merged = churn.merge(ultima_mensalidade, on="user_id", how="left")

        merged["meses_restantes"] = merged["plano_duracao_meses"] - merged["mes_churn"]
        merged["meses_restantes"] = merged["meses_restantes"].clip(lower=0)
        merged["receita_total_faltante"] = merged["meses_restantes"] * merged["valor_mensalidade"]
        merged["mes_nome"] = merged["mes_calendario_churn"].apply(lambda x: calendar.month_name[int(x)])

        meses = list(calendar.month_name)[1:]
        receita_prev = merged.groupby("mes_nome")["receita_total_faltante"].sum().reindex(meses, fill_value=0)

        receita_rec = receita_prev * (percentual / 100)
        df_recup = pd.DataFrame({
            "Mês": meses,
            "Receita Recuperada (R$)": receita_rec.values
        })

        st.subheader("💰 Receita Recuperada por Mês (R$)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_recup, x="Receita Recuperada (R$)", y="Mês", palette="Purples", ax=ax2)
        for p in ax2.patches:
            ax2.text(p.get_width() + 1, p.get_y() + 0.4, f'{int(p.get_width())}', fontsize=9)
        st.pyplot(fig2)

        st.metric("Total de Receita Recuperável (R$)", f"R$ {df_recup['Receita Recuperada (R$)'].sum():,.2f}")


def tela_politica_churn(dfs):
    st.title("Política de Churn")

    churn = dfs["churn"].copy()
    churn["score_fake"] = np.random.uniform(0.3, 0.9, len(churn))

    matriz = churn.groupby("mes_churn")["score_fake"].mean().reset_index()
    matriz.columns = ["Período do Plano (Mês)", "Score Médio de Churn"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(matriz.set_index("Período do Plano (Mês)").T, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
    st.subheader("📌 Score Médio por Período de Curso")
    st.pyplot(fig)


def tela_perfis_churn(dfs):
    st.title("Perfis de Churn por Mês")

    churn = dfs["churn"].copy()
    clientes = dfs["clientes"].copy()
    churn["user_id"] = churn["user_id"].astype(str)
    clientes["user_id"] = clientes["user_id"].astype(str)

    meses_opcoes = list(calendar.month_name)[1:]
    mes_nome = st.selectbox("Selecione o mês para análise", meses_opcoes)
    mes_num = meses_opcoes.index(mes_nome) + 1

    base = churn[churn["mes_calendario_churn"] == mes_num].merge(clientes, on="user_id", how="left")
    base["data_nascimento"] = pd.to_datetime(base["data_nascimento"], errors="coerce")
    base = base.dropna(subset=["data_nascimento"])
    base["idade"] = pd.to_datetime("today").year - base["data_nascimento"].dt.year

    base = base[["user_id", "nome", "idade", "estado", "tipo_plano", "canal"]].dropna()

    base_modelo = pd.get_dummies(base[["idade", "estado", "tipo_plano", "canal"]])
    modelo_kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    base["cluster"] = modelo_kmeans.fit_predict(base_modelo)

    st.subheader("🔍 Grupos de Alunos Desistentes")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=base, x="idade", y="cluster", hue="estado", palette="tab10", ax=ax)
    st.pyplot(fig)

    st.dataframe(base)
    gerar_csv_download(base, "perfis_churn_mes_" + mes_nome.lower())
