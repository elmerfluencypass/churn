import streamlit as st
from auth import login
from data_loader import load_all_data
from plots import show_churn_dashboard

st.set_page_config(page_title="Churn Analysis", layout="wide")

def main():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.session_state.authenticated = login()
        st.rerun()  # Corrigido aqui
    else:
        st.image("logo.webp", width=150)
        menu = st.sidebar.selectbox("Menu", ["Dataviz"])

        if menu == "Dataviz":
            data = load_all_data()
            show_churn_dashboard(data)

if __name__ == "__main__":
    main()
