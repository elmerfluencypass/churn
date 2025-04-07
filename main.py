import streamlit as st
from auth import login
from data_loader import load_all_data
from plots import show_churn_dashboard

st.set_page_config(page_title="Churn Analysis", layout="wide")

def main():
    if not login():
        return

    st.image("logo.webp", width=150)
    menu = st.sidebar.selectbox("Menu", ["Dataviz"])

    if menu == "Dataviz":
        with st.spinner("Carregando dados..."):
            progress = st.progress(0)
            data = load_all_data()
            progress.progress(100)
        show_churn_dashboard(data)

if __name__ == "__main__":
    main()
