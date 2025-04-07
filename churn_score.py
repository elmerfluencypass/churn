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
    df_clientes['data_nascimento'] = pd.to_datetime(df_clientes['data_nascimento'], errors='coerce')
    df_clientes['data_inicio_curso'] = pd.to_datetime(df_clientes['data_inicio_curso'], errors='coerce')
    df_clientes['ultima_data_pagamento'] = pd.to_datetime(df_clientes['ultima_data_pagamento'], errors='coerce')

    hoje = pd.Timestamp.now()
    df_clientes['dias_desde_ultimo_pgto'] = (hoje - df_clientes['ultima_data_pagamento']).dt.days
    df_clientes['duracao_meses'] = df_clientes['plano_duracao_meses']
    df_clientes['meses_ativos'] = ((hoje - df_clientes['data_inicio_curso']) / np.timedelta64(1, 'M')).astype(int)
    df_clientes['plano_ativo'] = df_clientes['meses_ativos'] < df_clientes['duracao_meses']
    df_clientes['churn'] = ((df_clientes['dias_desde_ultimo_pgto'] > 30) & df_clientes['plano_ativo']).astype(int)

    df_model = df_clientes.copy()
    df_model['idade'] = hoje.year - df_model['data_nascimento'].dt.year
    df_model = df_model.dropna(subset=['idade', 'sexo', 'estado', 'canal_aquisicao'])

    df_model = pd.get_dummies(df_model, columns=['sexo', 'estado', 'canal_aquisicao'], drop_first=True)

    feature_cols = ['idade', 'duracao_meses', 'meses_ativos', 'dias_desde_ultimo_pgto'] +                    [col for col in df_model.columns if col.startswith(('sexo_', 'estado_', 'canal_aquisicao_'))]

    X = df_model[feature_cols]
    y = df_model['churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    scores = model.predict_proba(X)[:, 1]

    scaler = MinMaxScaler()
    churn_scores = scaler.fit_transform(scores.reshape(-1, 1)).flatten()

    resultado = df_model[['user_id']].copy()
    resultado['churn_score'] = churn_scores

    st.subheader("📋 Top 10 alunos com maior risco de churn")
    st.dataframe(resultado.sort_values(by='churn_score', ascending=False).head(10))

    st.download_button(
        label="📥 Exportar CSV com Scores",
        data=resultado.to_csv(index=False).encode('utf-8'),
        file_name='churn_scores.csv',
        mime='text/csv'
    )
