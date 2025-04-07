import streamlit as st
import pandas as pd
import plotly.express as px
from data_processing import identificar_desistentes, enriquecer_com_idade

def show_churn_dashboard(data):
    clientes = data["customer_profile_table"]
    pagamentos = data["historico_pagamentos"]
    insider = data["tbl_insider"]

    # Enriquecer e filtrar desistentes
    clientes = enriquecer_com_idade(clientes)
    desistentes, _ = identificar_desistentes(clientes, pagamentos)

    # Merge com insider para obter dados adicionais
    merged = desistentes.merge(insider, how="left", left_on="user_id", right_on="student_id")

    st.subheader("📍 Distribuição de Alunos Desistentes por UF")
    plot_uf_pizza(merged)

    st.subheader("🎓 Nível do Curso dos Alunos Desistentes")
    plot_last_level_pizza(merged)

    st.subheader("💬 Distribuição por Nome do Plano (Bolhas)")
    plot_plan_name_bubbles(merged)

    st.subheader("🎯 Objetivos Declarados dos Alunos Desistentes")
    plot_last_objective_bar(merged)

    st.subheader("📊 Matriz: Mês da Desistência vs Período do Curso")
    plot_matriz_periodo_mes(pagamentos, desistentes)

# 1. Gráfico pizza por UF
def plot_uf_pizza(df):
    uf_counts = df["student_uf"].value_counts().reset_index()
    uf_counts.columns = ["UF", "Quantidade"]
    fig = px.pie(uf_counts, values="Quantidade", names="UF", title="Distribuição por UF")
    st.plotly_chart(fig, use_container_width=True)

# 2. Gráfico pizza por last_level
def plot_last_level_pizza(df):
    levels = df["last_level"].dropna().value_counts().reset_index()
    levels.columns = ["Nível", "Quantidade"]
    fig = px.pie(levels, values="Quantidade", names="Nível", title="Distribuição por Nível do Curso")
    st.plotly_chart(fig, use_container_width=True)

# 3. Gráfico de bolhas por plan_name
def plot_plan_name_bubbles(df):
    plans = df["plan_name"].dropna().value_counts().reset_index()
    plans.columns = ["Plano", "Quantidade"]
    fig = px.scatter(plans, x="Plano", y="Quantidade", size="Quantidade", hover_name="Plano",
                     title="Distribuição por Plano Contratado")
    st.plotly_chart(fig, use_container_width=True)

# 4. Gráfico de barras por last_objective
def plot_last_objective_bar(df):
    objetivos = df["last_objective"].dropna().value_counts().reset_index()
    objetivos.columns = ["Objetivo", "Quantidade"]
    fig = px.bar(objetivos, x="Objetivo", y="Quantidade", title="Objetivos Declarados por Alunos Desistentes")
    st.plotly_chart(fig, use_container_width=True)

# 5. Matriz mês x parcela do curso
def plot_matriz_periodo_mes(pagamentos_df, desistentes_df):
    pagamentos_df["data_prevista_pagamento"] = pd.to_datetime(pagamentos_df["data_prevista_pagamento"], errors="coerce")
    df = pagamentos_df[pagamentos_df["user_id"].isin(desistentes_df["user_id"])]
    df["mes_ano"] = df["data_prevista_pagamento"].dt.strftime("%B")
    df["parcela"] = df["mes"]

    matriz = pd.pivot_table(df, values="user_id", index="mes_ano", columns="parcela", aggfunc="count", fill_value=0)
    fig = px.imshow(matriz, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                    title="Matriz de Desistência por Mês do Ano vs Período do Curso")
    st.plotly_chart(fig, use_container_width=True)
