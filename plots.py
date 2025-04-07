import streamlit as st
import pandas as pd
import plotly.express as px
from data_processing import identificar_desistentes, enriquecer_com_idade

def show_churn_dashboard(data):
    clientes = data["customer_profile_table"]
    pagamentos = data["historico_pagamentos"]
    insider = data["tbl_insider"]
    churn = data["churn_detectado"]

    # Enriquecer e filtrar desistentes
    clientes = enriquecer_com_idade(clientes)
    desistentes, _ = identificar_desistentes(clientes, pagamentos)

    # Merge com insider
    merged = desistentes.merge(insider, how="left", left_on="user_id", right_on="student_id")

    st.subheader("📉 Churn por Mês")
    plot_churn_mensal(desistentes)

    st.subheader("👥 Distribuição Etária (Desistentes)")
    plot_distribuicao_etaria(desistentes)

    st.subheader("📍 Distribuição de Alunos Desistentes por UF")
    plot_uf_pizza(merged)

    st.subheader("🎓 Nível do Curso dos Alunos Desistentes")
    plot_last_level_pizza(merged)

    st.subheader("💬 Distribuição por Nome do Plano (Bolhas)")
    plot_plan_name_bubbles(merged)

    st.subheader("🎯 Objetivos Declarados dos Alunos Desistentes")
    plot_last_objective_bar(merged)

    st.subheader("📊 Matriz: Desistência por Mês x Período do Curso")
    plot_matriz_periodo_mes(data["historico_pagamentos"], data["customer_profile_table"])

# 1. Gráfico churn por mês
def plot_churn_mensal(df):
    df["mes_nome"] = pd.to_datetime(df["ultima_data_pagamento"], errors="coerce").dt.strftime("%b")
    contagem = df["mes_nome"].value_counts().sort_index().reset_index()
    contagem.columns = ["mes_nome", "desistencias"]
    fig = px.bar(contagem, x="mes_nome", y="desistencias", title="Desistências por Mês")
    st.plotly_chart(fig, use_container_width=True)

# 2. Gráfico distribuição etária
def plot_distribuicao_etaria(df):
    df = df[df["idade"].notnull() & (df["idade"] > 0)]
    fig = px.histogram(df, x="idade", nbins=20, title="Distribuição de Idade dos Desistentes")
    st.plotly_chart(fig, use_container_width=True)

# 3. Pizza por UF
def plot_uf_pizza(df):
    uf_counts = df["student_uf"].value_counts().reset_index()
    uf_counts.columns = ["UF", "Quantidade"]
    fig = px.pie(uf_counts, values="Quantidade", names="UF", title="Distribuição por UF")
    st.plotly_chart(fig, use_container_width=True)

# 4. Pizza por nível do curso
def plot_last_level_pizza(df):
    levels = df["last_level"].dropna().value_counts().reset_index()
    levels.columns = ["Nível", "Quantidade"]
    fig = px.pie(levels, values="Quantidade", names="Nível", title="Distribuição por Nível do Curso")
    st.plotly_chart(fig, use_container_width=True)

# 5. Bolhas por plano
def plot_plan_name_bubbles(df):
    plans = df["plan_name"].dropna().value_counts().reset_index()
    plans.columns = ["Plano", "Quantidade"]
    fig = px.scatter(plans, x="Plano", y="Quantidade", size="Quantidade", hover_name="Plano",
                     title="Distribuição por Plano Contratado")
    st.plotly_chart(fig, use_container_width=True)

# 6. Objetivos finais
def plot_last_objective_bar(df):
    objetivos = df["last_objective"].dropna().value_counts().reset_index()
    objetivos.columns = ["Objetivo", "Quantidade"]
    fig = px.bar(objetivos, x="Objetivo", y="Quantidade", title="Objetivos Declarados por Alunos Desistentes")
    st.plotly_chart(fig, use_container_width=True)

# 7. Matriz mês do ano x período do curso (nova lógica)
def plot_matriz_periodo_mes(pagamentos_df, cadastro_df):
    pagamentos_df["user_id"] = pagamentos_df["user_id"].astype(str)
    cadastro_df["user_id"] = cadastro_df["user_id"].astype(str)

    hoje = pd.to_datetime("today").normalize()
    cadastro_df["ultima_data_pagamento"] = pd.to_datetime(cadastro_df["ultima_data_pagamento"], errors="coerce")
    cadastro_df["dias_sem_pagar"] = (hoje - cadastro_df["ultima_data_pagamento"]).dt.days
    cadastro_df["status_atual"] = cadastro_df["status_atual"].str.lower()

    desistentes = cadastro_df[
        (cadastro_df["status_atual"] != "concluído") &
        (cadastro_df["dias_sem_pagar"] > 30)
    ].copy()

    pagamentos_filtrados = pagamentos_df[pagamentos_df["user_id"].isin(desistentes["user_id"])].copy()
    pagamentos_filtrados["data_prevista_pagamento"] = pd.to_datetime(pagamentos_filtrados["data_prevista_pagamento"], errors="coerce")
    pagamentos_filtrados["mes_calendario"] = pagamentos_filtrados["data_prevista_pagamento"].dt.strftime("%B")
    pagamentos_filtrados["mes"] = pagamentos_filtrados["mes"].astype(str)

    matriz = pd.pivot_table(
        pagamentos_filtrados,
        values="user_id",
        index="mes_calendario",
        columns="mes",
        aggfunc="count",
        fill_value=0
    )

    ordem_meses = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    matriz = matriz.reindex(ordem_meses)

    fig = px.imshow(
        matriz,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Matriz de Desistência por Mês do Ano vs Período do Curso",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)
