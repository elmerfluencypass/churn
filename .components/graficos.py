import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
import pandas as pd

def histogramas(df: pd.DataFrame, coluna: str):
    fig = px.histogram(df, x=coluna, title=f"Histograma de {coluna}")
    st.plotly_chart(fig, use_container_width=True)

def boxplot(df: pd.DataFrame, coluna: str):
    fig = px.box(df, y=coluna, title=f"Boxplot de {coluna}")
    st.plotly_chart(fig, use_container_width=True)

def grafico_pizza(df: pd.DataFrame, coluna: str):
    contagem = df[coluna].value_counts().reset_index()
    contagem.columns = [coluna, 'total']
    fig = px.pie(contagem, names=coluna, values='total', title=f"Distribuição de {coluna}")
    st.plotly_chart(fig, use_container_width=True)

def correlacoes(df: pd.DataFrame):
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
