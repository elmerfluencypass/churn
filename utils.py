import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
import base64
import calendar
from datetime import datetime
import requests
from io import StringIO

CSV_URLS = {
    "clientes": "https://drive.google.com/uc?id=1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT",
    "churn": "https://drive.google.com/uc?id=1kkdfrCjTjyzjqYfX7C9vDSdfod5HBYQS",
    "pagamentos": "https://drive.google.com/uc?id=1j2RW-ryt4H3iX8nkaw371yBI0_EGA8MY"
}

@st.cache_data
def carregar_dados():
    dfs = {}
    for nome, url in CSV_URLS.items():
        response = requests.get(url)
        dfs[nome] = pd.read_csv(StringIO(response.text))
    return dfs

def adicionar_logo():
    logo_path = "fluencypass_logo_converted.png"
    with open(logo_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(f"<div style='position:fixed;top:10px;right:10px;'>"
                f"<img src='data:image/png;base64,{encoded}' width='120'></div>", unsafe_allow_html=True)

def barra_progresso_mensagem(texto):
    progresso = st.progress(0)
    status = st.empty()
    for i in range(100):
        progresso.progress(i + 1)
        status.text(f"{texto}... {i + 1}%")
    progresso.empty()
    status.empty()

def tela_login():
    adicionar_logo()
    st.title("Churn Prediction")
    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if username == "fluencypass123" and password == "fluencypass123":
            st.session_state.autenticado = True
        else:
            st.error("Usuário ou senha incorretos.")

def tela_dataviz(dfs):
    adicionar_logo()
    st.markdown("## :bar_chart: Visão Geral de Churn")

    churn = dfs["churn"]
    churn["mes_churn"] = pd.to_datetime(churn["data_vencimento_ultima_mensalidade"], errors="coerce").dt.month

    hist = churn["mes_churn"].value_counts().sort_index()
    fig = px.bar(
        x=hist.index,
        y=hist.values,
        labels={"x": "Mês", "y": "Alunos Desistentes"},
        title="Quantidade de Alunos Desistentes por Mês",
        color=hist.values,
        color_continuous_scale="greens"
    )
    st.plotly_chart(fig)

    clientes = dfs["clientes"]
    cancelamentos = churn.merge(clientes[["id_aluno", "duracao_plano_meses"]], on="id_aluno", how="left")
    cancelamentos["mes_atual_plano"] = cancelamentos["mes_churn"]

    matriz = pd.crosstab(cancelamentos["mes_churn"], cancelamentos["mes_atual_plano"])
    matriz = matriz.loc[:, matriz.columns >= 1]
    st.markdown("### Matriz de Alunos Desistentes por Mês e Período do Plano")
    st.dataframe(matriz.style.background_gradient(cmap="Greens"))

    pagamentos = dfs["pagamentos"]
    pagamentos["data_real_pagamento"] = pd.to_datetime(pagamentos["data_real_pagamento"], errors="coerce")
    ultima = pagamentos.sort_values("data_real_pagamento").drop_duplicates("id_aluno", keep="last")

    receita = churn.merge(ultima[["id_aluno", "valor_mensalidade"]], on="id_aluno", how="left")
    receita = receita.merge(clientes[["id_aluno", "duracao_plano_meses"]], on="id_aluno", how="left")
    receita["mes_churn"] = pd.to_datetime(receita["data_vencimento_ultima_mensalidade"]).dt.month
    receita["mes_atual_plano"] = receita["mes_churn"]
    receita["meses_restantes"] = receita["duracao_plano_meses"] - receita["mes_churn"]
    receita["valor_total_perdido"] = receita["meses_restantes"] * receita["valor_mensalidade"]

    matriz_receita = pd.pivot_table(receita, values="valor_total_perdido",
                                    index="mes_churn", columns="mes_atual_plano",
                                    aggfunc="sum", fill_value=0)
    matriz_receita = matriz_receita.loc[:, matriz_receita.columns >= 1]
    st.markdown("### Matriz de Receita Perdida por Mês e Período do Plano (R$)")
    st.dataframe(matriz_receita.style.background_gradient(cmap="Reds"))


def tela_score_churn(dfs):
    adicionar_logo()
    st.markdown("## :bar_chart: Score de Propensão ao Churn Mensal")
    clientes = dfs["clientes"]
    churn = dfs["churn"]
    pagamentos = dfs["pagamentos"]

    clientes["data_nascimento"] = pd.to_datetime(clientes["data_nascimento"], errors="coerce")
    clientes["idade"] = pd.to_datetime("today").year - clientes["data_nascimento"].dt.year

    pagamentos["data_real_pagamento"] = pd.to_datetime(pagamentos["data_real_pagamento"], errors="coerce")
    ultima = pagamentos.sort_values("data_real_pagamento").drop_duplicates("id_aluno", keep="last")

    base = clientes.merge(ultima[["id_aluno", "valor_mensalidade"]], on="id_aluno", how="left")
    base = base[~base["id_aluno"].isin(churn["id_aluno"])]
    base = base.dropna()

    X = base[["idade", "valor_mensalidade"]]
    y = np.random.randint(0, 2, size=len(X))

    model = LGBMClassifier()
    grid = {"n_estimators": [100, 150], "learning_rate": [0.05, 0.1]}
    gs = GridSearchCV(model, grid, cv=3)
    gs.fit(X, y)

    base["score_churn"] = gs.predict_proba(X)[:, 1]
    base["mes_churn_previsto"] = datetime.now().month + 1

    st.dataframe(base[["nome", "score_churn", "mes_churn_previsto"]])
    csv = base[["id_aluno", "nome", "score_churn", "mes_churn_previsto"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download Score", csv, "score_churn.csv", "text/csv")


def tela_pov(dfs):
    adicionar_logo()
    st.markdown("## :money_with_wings: Prova de Valor")
    percentual = st.selectbox("Percentual de Recuperação", list(range(5, 101, 5)), index=19)
    churn = dfs["churn"]
    pagamentos = dfs["pagamentos"]

    pagamentos["data_real_pagamento"] = pd.to_datetime(pagamentos["data_real_pagamento"], errors="coerce")
    ultima = pagamentos.sort_values("data_real_pagamento").drop_duplicates("id_aluno", keep="last")
    receita = churn.merge(ultima[["id_aluno", "valor_mensalidade"]], on="id_aluno", how="left")
    receita["mes"] = pd.to_datetime(receita["data_vencimento_ultima_mensalidade"]).dt.strftime("%B")

    receita_mensal = receita.groupby("mes")["valor_mensalidade"].sum()
    alunos = receita["mes"].value_counts().sort_index()

    alunos_simulados = alunos * (percentual / 100)
    receita_simulada = receita_mensal * (percentual / 100)

    fig1 = px.bar(x=alunos_simulados.index, y=alunos_simulados.values,
                  labels={"x": "Mês", "y": "Alunos Recuperados"}, title="Alunos Recuperados por Mês")
    fig2 = px.bar(x=receita_simulada.index, y=receita_simulada.values,
                  labels={"x": "Mês", "y": "Receita Recuperada"}, title="Receita Recuperada por Mês (R$)")

    st.plotly_chart(fig1)
    st.plotly_chart(fig2)
    st.metric("Total Recuperável", f"R$ {receita_simulada.sum():,.2f}")


def tela_politica_churn(dfs):
    adicionar_logo()
    st.markdown("## :dart: Política de Churn")
    churn = dfs["churn"]
    churn["mes_churn"] = pd.to_datetime(churn["data_vencimento_ultima_mensalidade"], errors="coerce").dt.month
    churn["score_churn"] = np.random.rand(len(churn))

    clientes = dfs["clientes"]
    churn = churn.merge(clientes[["id_aluno", "duracao_plano_meses"]], on="id_aluno", how="left")
    churn["mes_do_plano"] = churn["duracao_plano_meses"] - churn["mes_churn"]
    churn = churn[churn["mes_do_plano"] >= 0]

    media_score = churn.groupby("mes_do_plano")["score_churn"].mean().reset_index()
    fig = px.bar(media_score, x="mes_do_plano", y="score_churn",
                 labels={"mes_do_plano": "Mês do Plano", "score_churn": "Score Médio"},
                 title="Score Médio por Período do Plano",
                 color="score_churn", color_continuous_scale="RdBu")
    st.plotly_chart(fig)


def tela_perfis_churn(dfs):
    adicionar_logo()
    st.markdown("## :busts_in_silhouette: Perfis de Churn por Mês")
    mes_nome = st.selectbox("Selecione o mês para análise", list(calendar.month_name)[1:])
    mes_num = list(calendar.month_name).index(mes_nome)

    churn = dfs["churn"]
    clientes = dfs["clientes"]
    clientes["data_nascimento"] = pd.to_datetime(clientes["data_nascimento"], errors="coerce")
    clientes["idade"] = pd.to_datetime("today").year - clientes["data_nascimento"].dt.year

    base = churn[churn["data_vencimento_ultima_mensalidade"].str.contains(f"-{mes_num:02d}-", na=False)]
    base = base.merge(clientes, on="id_aluno", how="left")

    base_cluster = base[["idade", "duracao_plano_meses"]].dropna()
    scaler = StandardScaler()
    X = scaler.fit_transform(base_cluster)

    kmeans = KMeans(n_clusters=3, random_state=42)
    base["cluster"] = kmeans.fit_predict(X)

    fig = px.scatter(base, x="idade", y="duracao_plano_meses", color="cluster", title="Perfis de Churn Clusterizados")
    st.plotly_chart(fig)

    csv = base[["id_aluno", "nome", "idade", "duracao_plano_meses", "cluster"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download dos Perfis", csv, "perfis_churn.csv", "text/csv")
