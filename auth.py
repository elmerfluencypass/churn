import streamlit as st

def login():
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("Login")
        user = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")
        login_button = st.button("Entrar")

        if login_button:
            if user == "fluencypass123" and password == "fluencypass123":
                st.session_state.authenticated = True
                # Não usar st.experimental_rerun() aqui
            else:
                st.error("Credenciais inválidas.")

    return st.session_state.authenticated
