import streamlit as st
from components.carregamento import carregar_dados
from components.filtros import filtro_data
from components.graficos import histogramas, boxplot, grafico_pizza, correlacoes
from components.matriz_temporal import gerar_matriz_temporal
from components.padroes import identificar_padroes

st.set_page_config(layout="wide")
st.title("📉 Painel de Análise de Desistências")

df = carregar_dados()
df_filtrado = filtro_data(df)

col1, col2 = st.columns(2)
with col1:
    histogramas(df_filtrado, 'idade')
    boxplot(df_filtrado, 'tempo_uso')

with col2:
    grafico_pizza(df_filtrado, 'sexo')
    correlacoes(df_filtrado)

st.divider()
gerar_matriz_temporal(df_filtrado)

st.divider()
identificar_padroes(df_filtrado)
