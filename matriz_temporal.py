import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def gerar_matriz_temporal(df: pd.DataFrame):
    df['ano'] = df['data_evento'].dt.year
    df['mes'] = df['data_evento'].dt.month
    matriz = df[df['status'] == 'desistente'].pivot_table(index='ano', columns='mes', values='id_aluno', aggfunc='count', fill_value=0)
    st.write("### Matriz de Desistências por Mês")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(matriz, cmap="YlOrRd", annot=True, fmt="d", linewidths=0.5, ax=ax)
    st.pyplot(fig)
