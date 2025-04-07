import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import LabelEncoder
import vizro.plotly.express as vpx

st.set_page_config(layout="wide")
st.title("📉 Painel de Análise de Desistências")

# ----------------------
# 🔽 Baixar dados do Google Drive
# ----------------------

CSV_URLS = {
    "cadastro_clientes": "1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT",
    "churn_detectado": "1kkdfrCjTjyzjqYfX7C9vDSdfod5HBYQS",
    "historico_pagamentos": "1j2RW-ryt4H3iX8nkaw371yBI0_EGA8MY",
    "tbl_insider": "1tPqnQWmowQKNAx2M_4rLW5axH9DZ5TyW",
    "iugu_invoices": "1eNcobHn8QJKduVRcs79LsbzT_2L7hK88"
}

def baixar_dados():
    import os
    os.makedirs("data", exist_ok=True)
    paths = []
    for nome, file_id in CSV_URLS.items():
        path = f"data/{nome}.csv"
        url = f"https://drive.google.com/uc?id={file_id}"
        if not os.path.exists(path):
            gdown.download(url, path, quiet=False)
        paths.append(path)
    return paths

# ----------------------
# 📦 Carregar arquivos
# ----------------------

def carregar_dados():
    arquivos = baixar_dados()
    dfs = {}
    for path in arquivos:
        nome = path.split("/")[-1].replace(".csv", "")
        df = pd.read_csv(path)
        dfs[nome] = df
    return dfs

# ----------------------
# 🔄 Pré-processamento
# ----------------------

def preparar_dados(dfs):
    pagamentos = dfs['historico_pagamentos'].copy()
    cadastro = dfs['cadastro_clientes'].copy()

    pagamentos['data_real_pagamento'] = pd.to_datetime(pagamentos['data_real_pagamento'], errors='coerce')
    pagamentos['mes_pagamento'] = pagamentos['data_real_pagamento'].dt.month
    pagamentos['ano_pagamento'] = pagamentos['data_real_pagamento'].dt.year

    # Conversão segura da data de nascimento
    cadastro['data_nascimento'] = pd.to_datetime(cadastro['data_nascimento'], errors='coerce')
    cadastro_valid = cadastro[cadastro['data_nascimento'].notna()].copy()

    cadastro_valid['idade'] = (
        (pd.Timestamp.now(tz=None) - cadastro_valid['data_nascimento']).dt.days // 365
    ).astype('Int64')

    cadastro_valid['faixa_etaria'] = pd.cut(
        cadastro_valid['idade'],
        bins=[0, 17, 24, 34, 44, 54, 64, 200],
        labels=["<18", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    )
    
    return pagamentos, cadastro_valid

# ----------------------
# 📗 Histograma Vizro: Desistentes por mês
# ----------------------

def grafico_desistentes_por_mes(df):
    df['atraso'] = df['dias_em_atraso'] > 30
    desistentes = df[df['atraso'] & df['mes_pagamento'].notna()]
    resumo = desistentes.groupby('mes_pagamento').size().reset_index(name='total')
    fig = vpx.bar(resumo, x='mes_pagamento', y='total', title="Total de Desistentes por Mês", color_discrete_sequence=['green'])
    st.plotly_chart(fig, use_container_width=True)

# ----------------------
# 📘 Histograma Vizro: Perda financeira por período
# ----------------------

def grafico_perda_financeira(df):
    df['perda'] = df['valor_mensalidade'] * (12 - df['mes'])
    df_filtrado = df[df['dias_em_atraso'] > 30]
    perdas = df_filtrado.groupby('mes')['perda'].sum().reset_index()
    fig = vpx.bar(perdas, x='mes', y='perda', title="Desistentes por Período do Plano", color_discrete_sequence=['blue'])
    st.plotly_chart(fig, use_container_width=True)

# ----------------------
# 🧊 Matriz mês x mês
# ----------------------

def matriz_mes_a_mes(df):
    df['mes_real'] = pd.to_datetime(df['data_real_pagamento'], errors='coerce').dt.month
    df['mes_curso'] = df['mes']
    matriz = df[df['dias_em_atraso'] > 30].pivot_table(
        index='mes_real', columns='mes_curso', values='user_id', aggfunc='count', fill_value=0
    )
    st.write("### Matriz Mês-a-Mês de Desistência")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(matriz, annot=True, fmt='d', cmap='YlGnBu', linewidths=0.5, ax=ax)
    st.pyplot(fig)

# ----------------------
# 🧠 Estatística Kappa
# ----------------------

def estatistica_kappa(cadastro_df):
    st.subheader("📊 Estatística Kappa por Variáveis Categóricas")
    resultado = []
    variaveis = ['faixa_etaria', 'estado', 'tipo_plano', 'canal_aquisicao']
    cadastro_df = cadastro_df.dropna(subset=['status_atual'])

    for var in variaveis:
        if cadastro_df[var].nunique() > 1 and cadastro_df[var].notna().sum() > 0:
            le1 = LabelEncoder().fit(cadastro_df[var].astype(str))
            le2 = LabelEncoder().fit(cadastro_df['status_atual'].astype(str))
            kappa = cohen_kappa_score(
                le1.transform(cadastro_df[var].astype(str)),
                le2.transform(cadastro_df['status_atual'].astype(str))
            )
            resultado.append((var, round(kappa, 3)))

    df_result = pd.DataFrame(resultado, columns=['Variável', 'Kappa'])
    st.dataframe(df_result)

# ----------------------
# 🚀 Execução principal
# ----------------------

dfs = carregar_dados()
pagamentos_df, cadastro_df = preparar_dados(dfs)

grafico_desistentes_por_mes(pagamentos_df)
grafico_perda_financeira(pagamentos_df)
matriz_mes_a_mes(pagamentos_df)
estatistica_kappa(cadastro_df)
