import streamlit as st
from auth import login
from data_loader import load_all_data
from plots import show_churn_dashboard

st.set_page_config(page_title="Churn Analysis", layout="wide")

def main():
    # Controle de login
    if not login():
        return

    # Logo da Fluencypass no topo
    st.image("logo.webp", width=150)

    # Menu lateral
    menu = st.sidebar.selectbox("Menu", ["Dataviz"])

    if menu == "Dataviz":
        data = load_all_data()
        show_churn_dashboard(data)

if __name__ == "__main__":
    main()
