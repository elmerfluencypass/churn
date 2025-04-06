import streamlit as st
import pandas as pd

def tela_upload_csv():
    st.title("Upload de Arquivo CSV")

    uploaded_file = st.file_uploader("Escolha um arquivo CSV", type="csv")

    if uploaded_file is not None:
        if uploaded_file.type == "text/csv":
            try:
                df = pd.read_csv(uploaded_file)
                st.success("Arquivo carregado com sucesso!")
                st.dataframe(df)
            except Exception as e:
                st.error(f"Erro ao ler o arquivo: {e}")
        else:
            st.error("Por favor, envie um arquivo CSV válido.")
