import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loader import load_csv

def render():
    st.title("📊 Análise de Churn - DataViz")

    df_churn = load_csv("churn_detectado")
    df_clientes = load_csv("customer_profile_table")
    df_pagamentos = load_csv("historico_pagamentos")

    df_clientes['data_nascimento'] = pd.to_datetime(df_clientes['data_nascimento'], errors='coerce')
    df_clientes['ultima_data_pagamento'] = pd.to_datetime(df_clientes['ultima_data_pagamento'], errors='coerce')

    st.write("📌 Colunas do df_churn:", df_churn.columns.tolist())

    # Fallback para identificar coluna de cancelamento
    coluna_cancelamento = [col for col in df_churn.columns if 'cancelamento' in col]
    if coluna_cancelamento:
        cancelamento_col = coluna_cancelamento[0]
        df_churn[cancelamento_col] = pd.to_datetime(df_churn[cancelamento_col], errors='coerce')
        df_churn['mes_cancelamento'] = df_churn[cancelamento_col].dt.month
    else:
        st.error("❌ Coluna relacionada a cancelamento não encontrada.")
        st.stop()

    st.subheader("📅 Desistências por mês")
    desistencias_mes = df_churn['mes_cancelamento'].value_counts().sort_index()
    fig1, ax1 = plt.subplots()
    ax1.bar(desistencias_mes.index, desistencias_mes.values)
    ax1.set_xlabel("Mês")
    ax1.set_ylabel("Quantidade de desistências")
    ax1.set_xticks(range(1, 13))
    ax1.set_title("Total de Alunos Desistentes por Mês")
    st.pyplot(fig1)

    st.subheader("👤 Distribuição de idade dos alunos desistentes")
    current_year = pd.Timestamp.now().year
    df_merged = pd.merge(df_churn, df_clientes, on="user_id", how="left")
    df_merged['idade'] = current_year - df_merged['data_nascimento'].dt.year

    fig2, ax2 = plt.subplots()
    ax2.hist(df_merged['idade'].dropna(), bins=10)
    ax2.set_xlabel("Idade")
    ax2.set_ylabel("Frequência")
    ax2.set_title("Distribuição de Idade dos Alunos Desistentes")
    st.pyplot(fig2)

    st.subheader("📈 Matriz: Desistências por Mês vs. Período do Plano")
    df_merged['mes_cancelamento'] = df_merged[cancelamento_col].dt.month
    matriz_qtd = pd.pivot_table(
        df_merged,
        index='mes_cancelamento',
        columns='mes',
        values='user_id',
        aggfunc='count',
        fill_value=0
    )
    st.dataframe(matriz_qtd.style.format(precision=0), use_container_width=True)

    st.subheader("💰 Matriz: Valor financeiro total perdido (por mês e período do plano)")
    df_pagamentos['data_real_pagamento'] = pd.to_datetime(df_pagamentos['data_real_pagamento'], errors='coerce')
    df_pagamentos['mes_cancelamento'] = df_pagamentos['data_real_pagamento'].dt.month
    df_valor = pd.merge(df_churn[['user_id']], df_pagamentos, on='user_id')
    df_valor = df_valor[df_valor['status_pagamento'] != 'Pago']

    matriz_valor = pd.pivot_table(
        df_valor,
        index='mes_cancelamento',
        columns='mes',
        values='valor_mensalidade',
        aggfunc='sum',
        fill_value=0
    )
    st.dataframe(matriz_valor.style.format("R$ {:,.2f}"), use_container_width=True)
