import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loader import load_csv

def render():
    st.title("🛡️ Simulação de Prevenção de Churn - POV")

    # Carregar dados
    df_churn = load_csv("churn_detectado")
    df_pagamentos = load_csv("historico_pagamentos")

    # Conversões e preparo
    df_churn['data_cancelamento'] = pd.to_datetime(df_churn['data_cancelamento'], errors='coerce')
    df_churn['mes_cancelamento'] = df_churn['data_cancelamento'].dt.month

    df_pagamentos['data_real_pagamento'] = pd.to_datetime(df_pagamentos['data_real_pagamento'], errors='coerce')
    df_pagamentos['mes_pagamento'] = df_pagamentos['data_real_pagamento'].dt.month

    # Agrupar desistências por mês
    desistencias_por_mes = df_churn['mes_cancelamento'].value_counts().sort_index()

    # INPUT: percentual de retenção desejado
    st.subheader("🎛️ Selecione o percentual de retenção")
    percentual = st.selectbox("Percentual de alunos a reter", list(range(5, 105, 5)))
    st.markdown("---")

    # Simulação de alunos evitados
    alunos_ev = (desistencias_por_mes * (percentual / 100)).astype(int)
    st.subheader("📉 Alunos retidos por mês")
    fig1, ax1 = plt.subplots()
    ax1.bar(alunos_ev.index, alunos_ev.values)
    ax1.set_xlabel("Mês")
    ax1.set_ylabel("Alunos evitados")
    ax1.set_title("Simulação: Alunos Retidos por Mês")
    st.pyplot(fig1)

    # Cálculo de valor financeiro salvo
    df_valor = pd.merge(df_churn[['user_id', 'mes_cancelamento']], df_pagamentos, on='user_id')
    df_valor = df_valor[df_valor['status_pagamento'] != 'Pago']
    df_valor_mensal = df_valor.groupby('mes_cancelamento')['valor_mensalidade'].sum().sort_index()

    valor_salvo_mes = (df_valor_mensal * (percentual / 100)).round(2)

    st.subheader("💰 Receita preservada por mês")
    fig2, ax2 = plt.subplots()
    ax2.bar(valor_salvo_mes.index, valor_salvo_mes.values)
    ax2.set_xlabel("Mês")
    ax2.set_ylabel("Valor salvo (R$)")
    ax2.set_title("Simulação: Valor Financeiro Retido por Mês")
    st.pyplot(fig2)

    # Total anual salvo
    total = valor_salvo_mes.sum()
    st.subheader(f"📊 Receita total preservada no ano: R$ {total:,.2f}")
