
import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import base64
from datetime import datetime
import calendar
import vizro.plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
import plotly.express as px_native

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
    st.markdown(f"<div style='position:fixed;top:10px;right:10px;z-index:1000;'>"
                f"<img src='data:image/png;base64,{encoded}' width='130'></div>", unsafe_allow_html=True)

def barra_progresso_mensagem(texto):
    progresso = st.progress(0)
    status = st.empty()
    for i in range(101):
        progresso.progress(i)
        status.text(f"{texto}... {i}%")
    progresso.empty()
    status.empty()

def tela_login():
    adicionar_logo()
    st.title("Fluencypass")
    st.header("Login")
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

    churn = dfs.get("churn")
    clientes = dfs.get("clientes")

    if churn is None or clientes is None:
        st.error("Erro ao carregar os dados.")
        return

    if "data_vencimento_ultima_mensalidade" in churn.columns:
        churn["data_vencimento_ultima_mensalidade"] = pd.to_datetime(churn["data_vencimento_ultima_mensalidade"], errors="coerce")
        churn["mes_churn"] = churn["data_vencimento_ultima_mensalidade"].dt.month

        hist = churn["mes_churn"].value_counts().sort_index()
        fig = px.bar(
            x=hist.index,
            y=hist.values,
            labels={"x": "Mês", "y": "Alunos Desistentes"},
            title="Quantidade de Alunos Desistentes por Mês",
            color=hist.values,
            color_continuous_scale="greens"
        )
        st.plotly_chart(fig, use_container_width=True)

        if "id_aluno" in churn.columns and "id_aluno" in clientes.columns and "duracao_plano_meses" in clientes.columns:
            cancelamentos = churn.merge(clientes[["id_aluno", "duracao_plano_meses"]], on="id_aluno", how="left")
            cancelamentos["mes_atual_plano"] = cancelamentos["mes_churn"]
            matriz = pd.crosstab(cancelamentos["mes_churn"], cancelamentos["mes_atual_plano"])
            matriz = matriz.loc[:, matriz.columns >= 1]
            if not matriz.empty:
                st.markdown("### Matriz de Alunos Desistentes por Mês e Período do Plano")
                st.dataframe(matriz.style.background_gradient(cmap="Greens"), use_container_width=True)
            else:
                st.warning("Matriz de alunos desistentes está vazia.")
