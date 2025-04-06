
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
        time.sleep(0.002)
    progresso.empty()


def gerar_csv_download(df, nome):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(f"📥 Baixar {nome}", csv, f"{nome}.csv", "text/csv", use_container_width=True)


def tela_churn_score(dfs):
    st.title("Score de Propensão ao Churn Mensal")

    if st.button("Processar Score"):
        barra_progresso("Treinando modelo e prevendo scores...")

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

        X = base_modelo[["idade", "plano_duracao_meses"]]
        y = base_modelo["target"]

        modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        modelo.fit(X, y)

        ativos = clientes.copy()
        ativos = ativos.dropna(subset=["idade", "plano_duracao_meses", "valor_mensalidade"])
        X_ativos = ativos[["idade", "plano_duracao_meses"]]

        ativos["score_churn"] = modelo.predict_proba(X_ativos)[:, 1]
        ativos["score_churn"] = MinMaxScaler().fit_transform(ativos[["score_churn"]])
        ativos["score_churn"] = ativos["score_churn"].round(3)

        score_median = ativos["score_churn"].median()
        em_risco = ativos[ativos["score_churn"] >= score_median]

        st.metric("Total Alunos em Risco", len(em_risco))
        st.metric("Receita Possivelmente Perdida", f"R$ {em_risco['valor_mensalidade'].sum():,.2f}")
        st.dataframe(em_risco[["user_id", "nome", "idade", "score_churn", "valor_mensalidade"]])
        gerar_csv_download(em_risco, "alunos_risco_churn")


def tela_pov(dfs):
    st.title("Prova de Valor")

    percentual = st.selectbox("Percentual de Recuperação", list(range(5, 105, 5)), index=19)

    if st.button("Backtest"):
        barra_progresso("Simulando recuperação de receita...")

        churn = dfs["churn"].copy()
        pagamentos = dfs["pagamentos"].copy()

        pagamentos["user_id"] = pagamentos["user_id"].astype(str)
        churn["user_id"] = churn["user_id"].astype(str)

        pagamentos_validos = pagamentos.dropna(subset=["valor_mensalidade"])
        ultima = pagamentos_validos.sort_values("data_prevista_pagamento")                                    .drop_duplicates("user_id", keep="last")[["user_id", "valor_mensalidade"]]

        merged = churn.merge(ultima, on="user_id", how="left")
        merged = merged.dropna(subset=["valor_mensalidade", "plano_duracao_meses", "mes_churn", "mes_calendario_churn"])

        merged["meses_restantes"] = merged["plano_duracao_meses"] - merged["mes_churn"]
        merged["meses_restantes"] = merged["meses_restantes"].clip(lower=0)
        merged["recuperavel"] = merged["meses_restantes"] * merged["valor_mensalidade"]
        merged["mes_nome"] = merged["mes_calendario_churn"].apply(lambda x: calendar.month_name[int(x)])

        agrupado = merged.groupby("mes_nome")["recuperavel"].sum().reindex(list(calendar.month_name)[1:], fill_value=0)
        recuperado = agrupado * (percentual / 100)

        df_plot = pd.DataFrame({"Mês": agrupado.index, "Receita Recuperada": recuperado.values})

        st.subheader("💰 Receita Recuperada por Mês (R$)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_plot, y="Mês", x="Receita Recuperada", palette="YlGnBu", ax=ax)
        for bar in ax.patches:
            ax.text(bar.get_width(), bar.get_y() + 0.4, f"R$ {bar.get_width():,.0f}", fontsize=9)
        st.pyplot(fig)

        st.metric("Total de Receita Recuperada", f"R$ {recuperado.sum():,.2f}")


def tela_politica_churn(dfs):
    st.title("Política de Churn")

    churn = dfs["churn"].copy()
    churn["score_simulado"] = np.random.uniform(0.3, 0.9, size=len(churn))

    media_scores = churn.groupby("mes_churn")["score_simulado"].mean().reset_index()

    st.subheader("📊 Score Médio por Período do Plano (Simulado)")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(data=media_scores, x="mes_churn", y="score_simulado", palette="coolwarm", ax=ax)
    ax.set_ylabel("Score Médio")
    ax.set_xlabel("Mês do Plano")
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 0.01, f"{p.get_height():.2f}", ha='center')
    st.pyplot(fig)


def tela_perfis_churn(dfs):
    st.title("Perfis de Churn por Mês")

    churn = dfs["churn"].copy()
    clientes = dfs["clientes"].copy()

    churn["user_id"] = churn["user_id"].astype(str)
    clientes["user_id"] = clientes["user_id"].astype(str)

    clientes["data_nascimento"] = pd.to_datetime(clientes["data_nascimento"], errors="coerce")
    clientes["idade"] = pd.to_datetime("today").year - clientes["data_nascimento"].dt.year

    meses_opcoes = list(calendar.month_name)[1:]
    mes_nome = st.selectbox("Selecione o mês para análise", meses_opcoes)
    mes_num = meses_opcoes.index(mes_nome) + 1

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
    st.dataframe(base)
