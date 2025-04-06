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
