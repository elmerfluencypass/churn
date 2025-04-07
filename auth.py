import streamlit as st

def login():
    st.sidebar.title("Login")
    username = st.sidebar.text_input("Usuário")
    password = st.sidebar.text_input("Senha", type="password")
    if username == "fluencypass123" and password == "fluencypass123":
        return True
    elif username and password:
        st.sidebar.error("Credenciais inválidas")
    return False
