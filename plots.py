import streamlit as st
import pandas as pd
import vizro.plotly.express as px
from data_processing import identificar_desistentes, enriquecer_com_idade, churn_por_mes

def show_churn_dashboard(data):
    clientes = data["customer_profile_table"]
    pagamentos = data["historico_pagamentos"]

    clientes = enriquecer_com_idade(clientes)
    desistentes, ativos = identificar_desistentes(clientes, pagamentos)

    st.subheader("📉 Churn por Mês")
    plot_churn_mensal(desistentes)

    st.subheader("👥 Distribuição Etária")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Ativos**")
        plot_distribuicao_etaria(ativos)
    with col2:
        st.markdown("**Desistentes**")
        plot_distribuicao_etaria(desistentes)

    st.subheader("📦 Boxplot por Sexo")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**Ativos**")
        plot_boxplot_sexo(ativos)
    with col4:
        st.markdown("**Desistentes**")
        plot_boxplot_sexo(desistentes)

    st.subheader("🌍 Distribuição Geográfica")
    col5, col6 = st.columns(2)
    with col5:
        plot_bolhas(desistentes, "estado")
    with col6:
        plot_bolhas(desistentes, "cidade")

    st.subheader("📊 Matriz de Churn (Mês do Ano x Parcela do Curso)")
    plot_matriz_churn(pagamentos, desistentes)

def plot_churn_mensal(df):
    df = df.copy()
    df["mes"] = df["ultima_data_pagamento"].dt.month
    df["mes_nome"] = df["ultima_data_pagamento"].dt.strftime("%b")
    contagem = df.groupby("mes_nome").size().reset_index(name="desistencias")
    fig = px.bar(contagem, x="mes_nome", y="desistencias", title="Desistências por Mês")
    st.plotly_chart(fig, use_container_width=True)

def plot_distribuicao_etaria(df):
    df = df[df["idade"].notnull() & (df["idade"] > 0)]
    fig = px.histogram(df, x="idade", nbins=20, title="Distribuição de Idade")
    st.plotly_chart(fig, use_container_width=True)

def plot_boxplot_sexo(df):
    df = df[df["idade"].notnull() & df["sexo"].notnull()]
    if not df.empty:
        fig = px.box(df, x="sexo", y="idade", title="Boxplot Idade x Sexo")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dados insuficientes para este gráfico.")

def plot_bolhas(df, coluna):
    df_valid = df[df[coluna].notnull()]
    grupo = df_valid.groupby(coluna).size().reset_index(name="quantidade")
    fig = px.scatter(grupo, x=coluna, y="quantidade", size="quantidade",
                     hover_name=coluna, title=f"Distribuição por {coluna.capitalize()}")
    st.plotly_chart(fig, use_container_width=True)

def plot_matriz_churn(pagamentos_df, desistentes_df):
    pagamentos_df["data_prevista_pagamento"] = pd.to_datetime(pagamentos_df["data_prevista_pagamento"], errors="coerce")
    df = pagamentos_df[pagamentos_df["user_id"].isin(desistentes_df["user_id"])]
    df["mes_ano"] = df["data_prevista_pagamento"].dt.strftime("%B")
    df["parcela"] = df["mes"]

    matriz = pd.pivot_table(df, values="user_id", index="mes_ano", columns="parcela", aggfunc="count", fill_value=0)
    fig = px.imshow(matriz, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                    title="Matriz de Desistências por Mês vs Parcela do Curso")
    st.plotly_chart(fig, use_container_width=True)
