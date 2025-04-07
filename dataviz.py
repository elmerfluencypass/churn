import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from loader import load_csv

def render():
    st.title("📊 Análise de Churn - DataViz")

    df_churn = load_csv("churn_detectado")
    df_clientes = load_csv("customer_profile_table")
    df_pagamentos = load_csv("historico_pagamentos")

    df_clientes['data_nascimento'] = pd.to_datetime(df_clientes['data_nascimento'], errors='coerce')
    df_clientes['ultima_data_pagamento'] = pd.to_datetime(df_clientes['ultima_data_pagamento'], errors='coerce')

    df_churn['last_invoice_due_date'] = pd.to_datetime(df_churn['last_invoice_due_date'], errors='coerce')
    df_churn['data_inicio_curso'] = pd.to_datetime(df_churn['data_inicio_curso'], errors='coerce')
    hoje = pd.Timestamp.now()

    def calcular_meses_ativos(inicio):
        if pd.isnull(inicio):
            return np.nan
        delta = relativedelta(hoje, inicio)
        return delta.years * 12 + delta.months

    df_churn['meses_ativos'] = df_churn['data_inicio_curso'].apply(calcular_meses_ativos)
    df_churn['plano_ativo'] = df_churn['meses_ativos'] < df_churn['plano_duracao_meses']

    df_churn['churn_inferido'] = (
        (hoje - df_churn['last_invoice_due_date']).dt.days > 30
    ) & (
        df_churn['plano_ativo']
    ) & (
        df_churn['receita_perdida'] > 0
    )

    df_churn_filtrado = df_churn[df_churn['churn_inferido']].copy()
    df_churn_filtrado['mes_churn'] = pd.to_numeric(df_churn_filtrado['mes_churn'], errors='coerce')

    st.subheader("📅 Desistências por mês (inferido)")
    desistencias_mes = df_churn_filtrado['mes_churn'].value_counts().sort_index()
    fig1, ax1 = plt.subplots()
    ax1.bar(desistencias_mes.index, desistencias_mes.values)
    ax1.set_xlabel("Mês")
    ax1.set_ylabel("Quantidade de desistências")
    ax1.set_title("Total de Alunos Desistentes por Mês")
    st.pyplot(fig1)

    st.subheader("👤 Distribuição de idade dos alunos desistentes")
    df_merge = pd.merge(df_churn_filtrado[['user_id']], df_clientes, on='user_id', how='left')
    df_merge['idade'] = hoje.year - df_merge['data_nascimento'].dt.year
    fig2, ax2 = plt.subplots()
    ax2.hist(df_merge['idade'].dropna(), bins=10)
    ax2.set_xlabel("Idade")
    ax2.set_ylabel("Frequência")
    ax2.set_title("Distribuição de Idade dos Alunos Desistentes")
    st.pyplot(fig2)

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
