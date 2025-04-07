import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
from data_processing import identificar_desistentes, enriquecer_com_idade, churn_por_mes

def show_churn_dashboard(data):
    clientes = data["customer_profile_table"]
    pagamentos = data["historico_pagamentos"]

    clientes = enriquecer_com_idade(clientes)
    desistentes, ativos = identificar_desistentes(clientes, pagamentos)

    st.subheader("📉 Churn por Mês")
    plot_histograma_churn(desistentes)

    st.subheader("👥 Distribuição Etária")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Ativos**")
        plot_histograma_idade(ativos)
    with col2:
        st.markdown("**Desistentes**")
        plot_histograma_idade(desistentes)

    st.subheader("📦 Boxplots por Sexo")
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
        st.markdown("**Por Estado**")
        plot_bolhas(desistentes, "estado")
    with col6:
        st.markdown("**Por Cidade**")
        plot_bolhas(desistentes, "cidade")

    st.subheader("📊 Matriz de Churn por Mês x Período")
    plot_matriz_churn(pagamentos, desistentes)

def plot_histograma_churn(df):
    contagem = churn_por_mes(df)
    meses = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
    valores = [contagem.get(i, 0) for i in range(1, 13)]
    plt.figure(figsize=(10, 4))
    plt.bar(meses, valores)
    plt.xlabel("Mês")
    plt.ylabel("Desistências")
    plt.title("Churn por Mês")
    st.pyplot(plt.gcf())
    plt.clf()

def plot_histograma_idade(df):
    plt.figure(figsize=(6, 3))
    sns.histplot(df["idade"].dropna(), bins=10, kde=True)
    st.pyplot(plt.gcf())
    plt.clf()

def plot_boxplot_sexo(df):
    plt.figure(figsize=(4, 3))
    sns.boxplot(x="sexo", y="idade", data=df)
    st.pyplot(plt.gcf())
    plt.clf()

def plot_bolhas(df, col):
    grupo = df.groupby(col).size().reset_index(name="quantidade")
    fig = px.scatter(grupo, x=col, y="quantidade", size="quantidade", hover_name=col)
    st.plotly_chart(fig, use_container_width=True)

def plot_matriz_churn(pagamentos_df, desistentes_df):
    df = pagamentos_df[pagamentos_df["user id"].isin(desistentes_df["user id"])]
    df["data prevista pagamento"] = pd.to_datetime(df["data prevista pagamento"])
    df["mes_ano"] = df["data prevista pagamento"].dt.strftime("%b")
    
