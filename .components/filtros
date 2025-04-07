import streamlit as st
import pandas as pd

def filtro_data(df: pd.DataFrame) -> pd.DataFrame:
    min_date = df['data_evento'].min()
    max_date = df['data_evento'].max()
    data_ini, data_fim = st.date_input("Filtrar por data:", [min_date, max_date])
    return df[(df['data_evento'] >= pd.to_datetime(data_ini)) & (df['data_evento'] <= pd.to_datetime(data_fim))]
