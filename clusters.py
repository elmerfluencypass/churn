import streamlit as st
import pandas as pd
import gdown
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px

def run():
    st.header("🧠 Clusterização de Perfis de Alunos")

    url = "https://drive.google.com/uc?id=1kk5PZpfJPuvPFEYx9jO32xHcMdMLpDc8"
    gdown.download(url, "life_cycle.csv", quiet=True)
    df = pd.read_csv("life_cycle.csv")
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    df = df.select_dtypes(include='number').fillna(0)
    X = df.drop(columns=['user_id'], errors='ignore')
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=df['cluster'].astype(str),
                     labels={'x': 'PCA 1', 'y': 'PCA 2', 'color': 'Cluster'},
                     title="Visualização dos Clusters de Alunos")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📄 Dados com Cluster")
    st.dataframe(df.head())
