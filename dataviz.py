import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loader import load_csv

def render():
    st.title("📊 Análise de Churn - DataViz")

    # Carregar dados
    df_churn = load_csv("churn_detectado")
    df_clientes = load_csv("customer_profile_table")
    df_pagamentos = load_csv("historico_pagamentos")

    # Conversões necessárias
    df_clientes['data nascimento'] = pd.to_datetime(df_clientes['data nascimento'], errors='coerce')
    df_clientes['ultima data pagamento'] = pd.to_datetime(df_clientes['ultima data pagamento'], errors='coerce')
    
    # ==== GRÁFICO 1: Histograma de desistentes por mês ====
    df_churn['data_cancelamento'] = pd.to_datetime(df_churn['data_cancelamento'], errors='coerce')
    df_churn['mes_cancelamento'] = df_churn['data_cancelamento'].dt.month

    st.subheader("📅 Desistências por mês")
    desistencias_mes = df_churn['mes_cancelamento'].value_counts().sort_index()
    fig1, ax1 = plt.subplots()
    ax1.bar(desistencias_mes.index, desistencias_mes.values)
    ax1.set_xlabel("Mês")
    ax1.set_ylabel("Quantidade de desistências")
    ax1.set_xticks(range(1, 13))
    ax1.set_title("Total de Alunos Desistentes por Mês")
    st.pyplot(fig1)

    # ==== GRÁFICO 2: Distribuição de idade dos desistentes ====
    st.subheader("👤 Distribuição de idade dos alunos desistentes")
    current_year = pd.Timestamp.now().year
    df_merged = pd.merge(df_churn, df_clientes, on="user id", how="left")
    df_merged['idade'] = current_year - df_merged['data nascimento'].dt.year

    fig2, ax2 = plt.subplots()
    ax2.hist(df_merged['idade'].dropna(), bins=10)
    ax2.set_xlabel("Idade")
    ax2.set_ylabel("Frequência")
    ax2.set_title("Distribuição de Idade dos Alunos Desistentes")
    st.pyplot(fig2)

    # ==== MATRIZ 1: Quantidade de desistentes (mês vs período plano) ====
    st.subheader("📈 Matriz: Desistências por Mês vs. Período do Plano")
    df_merged['mes_cancelamento'] = df_merged['data_cancelamento'].dt.month
    matriz_qtd = pd.pivot_table(
        df_merged,
        index='mes_cancelamento',
        columns='mes',
        values='user id',
        aggfunc='count',
        fill_value=0
    )
    st.dataframe(matriz_qtd.style.format(precision=0), use_container_width=True)

    # ==== MATRIZ 2: Valor financeiro por célula ====
    st.subheader("💰 Matriz: Valor financeiro total perdido (por mês e período do plano)")
    df_pagamentos['data real pagamento'] = pd.to_datetime(df_pagamentos['data real pagamento'], errors='coerce')
    df_pagamentos['mes_cancelamento'] = df_pagamentos['data real pagamento'].dt.month
    df_valor = pd.merge(df_churn[['user id']], df_pagamentos, on='user id')
    df_valor = df_valor[df_valor['status pagamento'] != 'Pago']  # assumir valores não pagos como desistência

    matriz_valor = pd.pivot_table(
        df_valor,
        index='mes_cancelamento',
        columns='mes',
        values='valor mensalidade',
        aggfunc='sum',
        fill_value=0
    )
    st.dataframe(matriz_valor.style.format("R$ {:,.2f}"), use_container_width=True)
