import streamlit as st
import pandas as pd
import gdown
from datetime import datetime, timedelta
from PIL import Image
import vizro.plotting as vz

st.set_page_config(page_title="Fluencypass Churn", layout="wide")

# ---- Login Section ----
def login():
    logo = Image.open("logo.webp")  # ← agora busca o logo na raiz
    st.image(logo, width=150)
    st.title("Login Fluencypass")
    
    username = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    
    if st.button("Entrar"):
        if username == "fluencypass123" and password == "fluencypass123":
            st.session_state.logged_in = True
        else:
            st.error("Usuário ou senha inválidos")

# ---- Load Data from Google Drive ----
@st.cache_data
def load_data():
    urls = {
        "cadastro_clientes": "https://drive.google.com/uc?id=1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT",
        "churn_detectado": "https://drive.google.com/uc?id=1kkdfrCjTjyzjqYfX7C9vDSdfod5HBYQS",
        "historico_pagamentos": "https://drive.google.com/uc?id=1j2RW-ryt4H3iX8nkaw371yBI0_EGA8MY",
        "tbl_insider": "https://drive.google.com/uc?id=1tPqnQWmowQKNAx2M_4rLW5axH9DZ5TyW",
        "iugu_invoices": "https://drive.google.com/uc?id=1eNcobHn8QJKduVRcs79LsbzT_2L7hK88"
    }
    dataframes = {}
    for name, url in urls.items():
        output = f"/tmp/{name}.csv"
        gdown.download(url, output, quiet=False)
        dataframes[name] = pd.read_csv(output)
    return dataframes

# ---- Main Dataviz Screen ----
def dataviz():
    logo = Image.open("logo.webp")  # ← também na tela principal
    st.image(logo, width=150)
    st.title("📊 Análise de Churn - Fluencypass")

    data = load_data()
    cadastro = data["cadastro_clientes"]
    historico = data["historico_pagamentos"]

    # Pré-processamento
    cadastro['ultima data pagamento'] = pd.to_datetime(cadastro['ultima data pagamento'], errors='coerce')
    cadastro['idade'] = cadastro['data nascimento'].apply(lambda x: datetime.now().year - pd.to_datetime(x, errors='coerce').year)
    hoje = datetime.today()
    cadastro['desistente'] = cadastro['ultima data pagamento'] < (hoje - timedelta(days=30))
    desistentes = cadastro[cadastro['desistente'] == True]
    desistentes['mes_desistencia'] = desistentes['ultima data pagamento'].dt.strftime('%B')

    # -- 1. Distribuição da Idade
    st.subheader("Distribuição da Idade dos Alunos")
    vz.histogram(data=cadastro, x='idade', title="Distribuição da Idade dos Alunos").show()

    # -- 2. Quantidade de Desistentes por Mês (Verde)
    st.subheader("Quantidade de Desistentes por Mês")
    vz.histogram(
        data=desistentes,
        x='mes_desistencia',
        title="Quantidade de Desistentes por Mês",
        color_discrete_sequence=["#e0f2f1", "#66bb6a"]
    ).show()

    # -- 3. Mensalidades Não Pagas por Mês (Azul)
    st.subheader("Mensalidades Não Pagas por Mês")
    historico['data prevista pagamento'] = pd.to_datetime(historico['data prevista pagamento'], errors='coerce')
    historico['mes_pagamento'] = historico['data prevista pagamento'].dt.strftime('%B')
    inadimplentes = historico[historico['status pagamento'] == "Em aberto"]
    inadimplentes_grouped = inadimplentes.groupby('mes_pagamento')['valor mensalidade'].sum().reset_index()

    vz.histogram(
        data=inadimplentes_grouped,
        x='mes_pagamento', y='valor mensalidade',
        title="Mensalidades Não Pagas por Mês (R$)",
        color_discrete_sequence=["#e3f2fd", "#2196f3"]
    ).show()

    # -- 4. Matriz Mês x Duração Plano
    st.subheader("Matriz de Desistência por Mês e Duração do Plano")
    desistentes['mes'] = desistentes['ultima data pagamento'].dt.month
    matriz = pd.pivot_table(
        data=desistentes,
        index='mes',
        columns='plano duracao meses',
        values='user id',
        aggfunc='count',
        fill_value=0
    )
    vz.heatmap(data=matriz, title="Matriz de Desistência por Mês e Duração do Plano").show()

    # -- 5. Bolhas: Sexo por Mês
    st.subheader("Distribuição de Desistência por Sexo")
    sexo_mes = desistentes.groupby(['mes_desistencia', 'sexo'])['user id'].count().reset_index()
    vz.scatter(
        data=sexo_mes,
        x='mes_desistencia', y='sexo', size='user id',
        title="Desistência por Sexo"
    ).show()

    # -- 6. Bolhas: Cidade
    st.subheader("Desistentes por Cidade")
    cidade_mes = desistentes.groupby(['mes_desistencia', 'cidade'])['user id'].count().reset_index()
    vz.scatter(
        data=cidade_mes,
        x='mes_desistencia', y='cidade', size='user id',
        title="Desistência por Cidade"
    ).show()

# ---- App Body ----
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
else:
    menu = st.sidebar.selectbox("Menu", ["Dataviz"])
    if menu == "Dataviz":
        dataviz()
