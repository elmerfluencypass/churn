import streamlit as st
import pandas as pd
import gdown
from datetime import datetime, timedelta
from PIL import Image
import plotly.express as px

st.set_page_config(page_title="Fluencypass Churn", layout="wide")

# ---- Login Section ----
def login():
    logo = Image.open("logo.webp")
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
        "customer_profile_table": "https://drive.google.com/uc?id=1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT",
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
    logo = Image.open("logo.webp")
    st.image(logo, width=150)
    st.title("📊 Análise de Churn - Fluencypass")

    data = load_data()
    cadastro = data["customer_profile_table"]

    # 🔍 Debug: Mostrar colunas para confirmar nomes
    st.subheader("Debug: Colunas disponíveis na tabela de clientes")
    st.write(cadastro.columns.tolist())

    historico = data["historico_pagamentos"]

    # ✅ Conversão segura da data e cálculo da idade
    cadastro['data_nascimento'] = pd.to_datetime(cadastro['data_nascimento'], errors='coerce')
    cadastro['idade'] = ((pd.Timestamp.now() - cadastro['data_nascimento']).dt.days // 365).astype('Int64')

    cadastro['ultima_data_pagamento'] = pd.to_datetime(cadastro['ultima_data_pagamento'], errors='coerce')
    hoje = datetime.today()
    cadastro['desistente'] = cadastro['ultima_data_pagamento'] < (hoje - timedelta(days=30))
    desistentes = cadastro[cadastro['desistente'] == True].copy()
    desistentes['mes_desistencia'] = desistentes['ultima_data_pagamento'].dt.strftime('%B')

    # -- 1. Distribuição da Idade
    st.subheader("Distribuição da Idade dos Alunos")
    fig_idade = px.histogram(cadastro, x='idade', nbins=20, title="Distribuição da Idade dos Alunos")
    st.plotly_chart(fig_idade, use_container_width=True)

    # -- 2. Quantidade de Desistentes por Mês (Verde)
    st.subheader("Quantidade de Desistentes por Mês")
    fig_desist = px.histogram(
        desistentes,
        x='mes_desistencia',
        color_discrete_sequence=px.colors.sequential.Greens,
        title="Quantidade de Desistentes por Mês"
    )
    st.plotly_chart(fig_desist, use_container_width=True)

    # -- 3. Mensalidades Não Pagas por Mês (Azul)
    st.subheader("Mensalidades Não Pagas por Mês")
    historico['data prevista pagamento'] = pd.to_datetime(historico['data prevista pagamento'], errors='coerce')
    historico['mes_pagamento'] = historico['data prevista pagamento'].dt.strftime('%B')
    inadimplentes = historico[historico['status pagamento'] == "Em aberto"]
    inadimplentes_grouped = inadimplentes.groupby('mes_pagamento')['valor mensalidade'].sum().reset_index()

    fig_inadimplencia = px.bar(
        inadimplentes_grouped,
        x='mes_pagamento',
        y='valor mensalidade',
        title="Mensalidades Não Pagas por Mês (R$)",
        color_discrete_sequence=px.colors.sequential.Blues
    )
    st.plotly_chart(fig_inadimplencia, use_container_width=True)

    # -- 4. Matriz Mês x Duração Plano
    st.subheader("Matriz de Desistência por Mês e Duração do Plano")
    desistentes['mes'] = desistentes['ultima_data_pagamento'].dt.month
    matriz = pd.pivot_table(
        data=desistentes,
        index='mes',
        columns='tipo_plano',
        values='user_id',
        aggfunc='count',
        fill_value=0
    )
    fig_matriz = px.imshow(
        matriz,
        labels=dict(x="Tipo do Plano", y="Mês", color="Desistentes"),
        color_continuous_scale=px.colors.sequential.Reds
    )
    st.plotly_chart(fig_matriz, use_container_width=True)

    # -- 5. Bolhas: Sexo por Mês
    st.subheader("Distribuição de Desistência por Sexo")
    if 'sexo' in cadastro.columns:
        sexo_mes = desistentes.groupby(['mes_desistencia', 'sexo'])['user_id'].count().reset_index()
        fig_sexo = px.scatter(
            sexo_mes,
            x='mes_desistencia',
            y='sexo',
            size='user_id',
            color='sexo',
            title="Desistência por Sexo",
            size_max=40
        )
        st.plotly_chart(fig_sexo, use_container_width=True)

    # -- 6. Bolhas: Cidade
    if 'cidade' in cadastro.columns:
        st.subheader("Desistentes por Cidade")
        cidade_mes = desistentes.groupby(['mes_desistencia', 'cidade'])['user_id'].count().reset_index()
        fig_cidade = px.scatter(
            cidade_mes,
            x='mes_desistencia',
            y='cidade',
            size='user_id',
            color='cidade',
            title="Desistência por Cidade",
            size_max=40
        )
        st.plotly_chart(fig_cidade, use_container_width=True)

# ---- App Body ----
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
else:
    menu = st.sidebar.selectbox("Menu", ["Dataviz"])
    if menu == "Dataviz":
        dataviz()
