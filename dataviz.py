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

    # Conversão da coluna de vencimento da última fatura
    df_churn['last_invoice_due_date'] = pd.to_datetime(df_churn['last_invoice_due_date'], errors='coerce')
    hoje = pd.Timestamp.now()

    # Calcular meses do plano restantes
    df_churn['data_inicio_curso'] = pd.to_datetime(df_churn['data_inicio_curso'], errors='coerce')
    df_churn['meses_ativos'] = ((hoje - df_churn['data_inicio_curso']) / np.timedelta64(1, 'M')).astype(int)
    df_churn['plano_ativo'] = df_churn['meses_ativos'] < df_churn['plano_duracao_meses']

    # Aplicar regra de churn inferido
    df_churn['churn_inferido'] = (
        (hoje - df_churn['last_invoice_due_date']).dt.days > 30
    ) & (
        df_churn['plano_ativo']
    ) & (
        df_churn['receita_perdida'] > 0
    )

    # Filtrar alunos churnados
    df_churn_filtrado = df_churn[df_churn['churn_inferido']].copy()
    df_churn_filtrado['mes_churn'] = pd.to_numeric(df_churn_filtrado['mes_churn'], errors='coerce')

    # ==== GRÁFICO 1: Desistências por mês ====
    st.subheader("📅 Desistências por mês (inferido)")
    desistencias_mes = df_churn_filtrado['mes_churn'].value_counts().sort_index()
    fig1, ax1 = plt.subplots()
    ax1.bar(desistencias_mes.index, desistencias_mes.values)
    ax1.set_xlabel("Mês")
    ax1.set_ylabel("Quantidade de desistências")
    ax1.set_title("Total de Alunos Desistentes por Mês")
    st.pyplot(fig1)

    # ==== GRÁFICO 2: Distribuição de idade ====
    st.subheader("👤 Distribuição de idade dos alunos desistentes")
    df_merge = pd.merge(df_churn_filtrado[['user_id']], df_clientes, on='user_id', how='left')
    df_merge['idade'] = hoje.year - df_merge['data_nascimento'].dt.year
    fig2, ax2 = plt.subplots()
    ax2.hist(df_merge['idade'].dropna(), bins=10)
    ax2.set_xlabel("Idade")
    ax2.set_ylabel("Frequência")
    ax2.set_title("Distribuição de Idade dos Alunos Desistentes")
    st.pyplot(fig2)

    # ==== MATRIZ 1: Quantidade por mês vs. período do plano ====
    st.subheader("📈 Matriz: Quantidade de desistentes por Mês vs. Período do Plano")
    matriz_qtd = pd.pivot_table(
        df_churn_filtrado,
        index='mes_churn',
        columns='mes_calendario_churn',
        values='user_id',
        aggfunc='count',
        fill_value=0
    )
    st.dataframe(matriz_qtd.style.format(precision=0), use_container_width=True)

    # ==== MATRIZ 2: Receita perdida por mês vs. período ====
    st.subheader("💰 Matriz: Receita perdida por Mês vs. Período do Plano")
    matriz_valor = pd.pivot_table(
        df_churn_filtrado,
        index='mes_churn',
        columns='mes_calendario_churn',
        values='receita_perdida',
        aggfunc='sum',
        fill_value=0
    )
    st.dataframe(matriz_valor.style.format("R$ {:,.2f}"), use_container_width=True)
