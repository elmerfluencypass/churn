import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def identificar_padroes(df: pd.DataFrame):
    df_model = df.copy()
    df_model['desistencia'] = df_model['status'] == 'desistente'
    X = df_model.drop(columns=['status', 'data_evento', 'id_aluno', 'desistencia'])
    y = df_model['desistencia']
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    top_features = importances.sort_values(ascending=False).head(10)
    st.write("### Principais Variáveis antes da Desistência")
    st.bar_chart(top_features)
    preds = clf.predict(X_test)
    st.text("Relatório de Classificação:")
    st.text(classification_report(y_test, preds))
