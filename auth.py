import streamlit as st

def login():
    # Inicializa o estado se ainda não existir
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # Se não autenticado, renderiza o formulário de login
    if not st.session_state.authenticated:
        st.title("Login")
        user = st.text_input("Usuário")
        password = st.text_input("Senha", type="password")

        if st.button("Entrar"):
            if user == "fluencypass123" and password == "fluencypass123":
                st.session_state.authenticated = True
                st.experimental_rerun()  # ou st.rerun() se usar Streamlit 1.25+

        # Logo abaixo do botão
        st.image("logo.webp", width=120)

    return st.session_state.authenticated
