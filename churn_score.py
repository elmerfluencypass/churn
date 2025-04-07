import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from loader import load_csv

def render():
    st.title("📈 Cálculo de Churn Score")

    # Carregar dados
    df_clientes = load_csv("customer_profile_table")
    df_pagamentos = load_csv("historico_pagamentos")

    # Preparo: dados básicos
    df_clientes['data nascimento'] = pd.to_datetime(df_clientes['data nascimento'], errors='coerce')
    df_clientes['data inicio curso'] = pd.to_datetime(df_clientes['data inicio curso'], errors='coerce')
    df_clientes['ultima data pagamento'] = pd.to_datetime(df_clientes['ultima data pagamento'], errors='coerce')

    # Criar variável "churn": último pagamento há mais de 30 dias e plano ainda vigente
    hoje = pd.Timestamp.now()
    df_clientes['dias_desde_ultimo_pgto'] = (hoje - df_clientes['ultima data pagamento']).dt.days
    df_clientes['duracao_meses'] = df_clientes['plano duracao meses']
    df_clientes['meses_ativos'] = ((hoje - df_clientes['data inicio curso']) / np.timedelta64(1, 'M')).astype(int)
    df_clientes['plano_ativo'] = df_clientes['meses_ativos'] < df_clientes['duracao_meses']
    df_clientes['churn'] = ((df_clientes['dias_desde_ultimo_pgto'] > 30) & df_clientes['plano_ativo']).astype(int)

    # Feature engineering
    df_model = df_clientes.copy()
    df_model['idade'] = hoje.year - df_model['data nascimento'].dt.year
    df_model = df_model.dropna(subset=['idade', 'sexo', 'estado', 'canal aquisicao'])

    # Codificação simples
    df_model = pd.get_dummies(df_model, columns=['sexo', 'estado', 'canal aquisicao'], drop_first=True)

    # Seleção de colunas
    feature_cols = ['idade', 'duracao_meses', 'meses_ativos', 'dias_desde_ultimo_pgto'] + \
                   [col for col in df_model.columns if col.startswith(('sexo_', 'estado_', 'canal aquisicao_'))]

    X = df_model[feature_cols]
    y = df_model['churn']

    # Amostragem estratificada
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    # Modelo simples (mock)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    scores = model.predict_proba(X)[:, 1]

    # Normalizar para 0-1
    scaler = MinMaxScaler()
    churn_scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()

    # Resultado final
    resultado = df_model[['user id']].copy()
    resultado['churn_score'] = churn_scores

    st.subheader("📋 Top 10 alunos com maior risco de churn")
    st.dataframe(resultado.sort_values(by='churn_score', ascending=False).head(10))

    # Exportação CSV
    st.download_button(
        label="📥 Exportar CSV com Scores",
        data=resultado.to_csv(index=False).encode('utf-8'),
        file_name='churn_scores.csv',
        mime='text/csv'
    )
