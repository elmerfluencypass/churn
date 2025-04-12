import streamlit as st
import pandas as pd
import gdown
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px

st.title("📈 Score Preditivo de Churn")

@st.cache_data
def baixar_csv(nome, file_id):
    url = f"https://drive.google.com/uc?id={file_id}"
    arq = f"{nome}.csv"
    gdown.download(url, arq, quiet=True)
    return pd.read_csv(arq)

df = baixar_csv("life_cycle", "1kk5PZpfJPuvPFEYx9jO32xHcMdMLpDc8")

st.markdown("Amostra de treino usando 5000 registros...")

# Engenharia básica
df = df.dropna(subset=['user_id', 'user_status'])
df = df.sample(n=5000, random_state=42) if len(df) > 5000 else df
X = df.select_dtypes(include='number').drop(columns=['user_id'], errors='ignore')
y = df['user_status'].astype(str).str.contains("churn|desist", case=False).astype(int)

X = X.fillna(0)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

df['score_churn'] = model.predict_proba(X_scaled)[:, 1]

st.write(df[['user_id', 'score_churn']].head())

fig = px.histogram(df, x='score_churn', nbins=50, title="Distribuição dos Scores de Churn")
st.plotly_chart(fig, use_container_width=True)
