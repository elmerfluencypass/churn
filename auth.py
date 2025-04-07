import streamlit as st

def login():
    st.title("Login")
    user = st.text_input("Usuário")
    password = st.text_input("Senha", type="password")
    if st.button("Entrar"):
        if user == "fluencypass123" and password == "fluencypass123":
            return True
        else:
            st.error("Credenciais inválidas.")
    return False
