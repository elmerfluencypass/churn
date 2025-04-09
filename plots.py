import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from data_processing import (
    identificar_desistentes,
    enriquecer_com_idade,
    preparar_dados_modelagem,
    calcular_metricas_desistentes
)

def show_churn_dashboard(data):
    """Exibe o dashboard completo de análise de churn."""
    try:
        clientes = data["customer_profile_table"]
        pagamentos = data["historico_pagamentos"]
        insider = data["tbl_insider"]
        
        # Enriquecer e filtrar desistentes
        clientes = enriquecer_com_idade(clientes)
        desistentes, _ = identificar_desistentes(clientes, pagamentos)
        
        # Merge com insider
        merged = desistentes.merge(insider, how="left", left_on="user_id", right_on="student_id")
        
        # Mostrar métricas resumidas
        st.subheader("📊 Métricas Gerais de Desistência")
        metricas = calcular_metricas_desistentes(desistentes)
        
        if not metricas.empty:
            st.dataframe(metricas.style.format({
                'idade_media': '{:.1f}',
                'mes_curso_medio': '{:.1f}',
                'taxa_engajamento': '{:.2f}',
                'taxa_atraso': '{:.2%}'
            }), use_container_width=True)
                # Gráficos existentes
        st.subheader("📉 Churn por Mês")
        plot_churn_mensal(desistentes)
        
        st.subheader("👥 Distribuição Etária (Desistentes)")
        plot_distribuicao_etaria(desistentes)
        
        st.subheader("📍 Distribuição por UF")
        plot_uf_pizza(merged)
        
        st.subheader("🎓 Nível do Curso")
        plot_last_level_pizza(merged)
        
        st.subheader("💬 Distribuição por Plano (Bolhas)")
        plot_plan_name_bubbles(merged)
        
        st.subheader("🎯 Objetivos Declarados")
        plot_last_objective_bar(merged)
        
        st.subheader("📊 Matriz: Mês x Período do Curso")
        plot_matriz_periodo_mes(pagamentos, clientes)
        
    except Exception as e:
        st.error(f"Erro ao exibir dashboard: {str(e)}")

def plot_churn_mensal(df):
    """Gráfico de barras de desistências por mês."""
    try:
        df["mes_nome"] = pd.to_datetime(df["ultima_data_pagamento"], errors="coerce").dt.strftime("%b")
        contagem = df["mes_nome"].value_counts().sort_index().reset_index()
        contagem.columns = ["mes_nome", "desistencias"]
        fig = px.bar(contagem, x="mes_nome", y="desistencias", 
                     title="Desistências por Mês",
                     labels={"mes_nome": "Mês", "desistencias": "Número de Desistências"})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao plotar gráfico mensal: {str(e)}")

def plot_distribuicao_etaria(df):
    """Histograma de distribuição de idade."""
    try:
        df = df[df["idade"].notnull() & (df["idade"] > 0)]
        fig = px.histogram(df, x="idade", nbins=20, 
                           title="Distribuição de Idade dos Desistentes",
                           labels={"idade": "Idade", "count": "Número de Alunos"})
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao plotar distribuição etária: {str(e)}")
def show_churn_profile(data):
    """Exibe a análise de perfil dos desistentes."""
    try:
        st.title("🔍 Perfil de Churn")
        
        df = data["df_modelagem"]
        if df.empty:
            st.warning("Nenhum dado disponível para análise.")
            return
        
        df = df[df["churn"] == 1].copy()
        
        if len(df) < 10:
            st.warning("Dados insuficientes para análise. Mínimo de 10 registros necessários.")
            return
        
        features = ["idade", "engagement_score", "pct_atraso_total", "valor_restante_contrato"]
        features = [f for f in features if f in df.columns]
        
        if len(features) < 2:
            st.warning("Features insuficientes para análise.")
            return
        
        # Processamento e clusterização
        df = df.dropna(subset=features)
        for col in features:
            if pd.api.types.is_numeric_dtype(df[col]):
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                df = df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)]
        
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        kmeans = KMeans(n_clusters=min(3, len(X_scaled)-1), random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        df["PCA1"] = X_pca[:, 0]
        df["PCA2"] = X_pca[:, 1]
        df["Cluster"] = clusters.astype(str)
        
        fig = px.scatter(
            df, x="PCA1", y="PCA2", color="Cluster",
            hover_data=["user_id"] + features,
            title="Clusterização de Alunos Desistentes",
            labels={"PCA1": "Componente Principal 1", "PCA2": "Componente Principal 2"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erro ao exibir perfil de churn: {str(e)}")

def calcular_score_de_churn(df_modelagem):
    """Calcula e exibe o score de propensão ao churn."""
    try:
        st.title("🧠 Score de Propensão à Desistência")
        
        df = df_modelagem.copy()
        df = df.dropna(subset=["user_id", "mes_curso"])
        df["mes_curso"] = df["mes_curso"].astype(int)
        
        desistentes = df[df["churn"] == 1].copy()
        ativos = df[df["churn"] == 0].copy()
        
        if len(desistentes) < 10 or len(ativos) < 10:
            st.warning("Dados insuficientes para cálculo de score.")
            return
        
        # Cálculo do score e exibição dos resultados
        prob_mes = desistentes.groupby("mes_curso").size() / df.groupby("mes_curso").size()
        prob_mes = prob_mes.fillna(0).to_dict()
        
        features = {}
        if "engagement_score" in df.columns:
            limiar_engajamento = desistentes["engagement_score"].quantile(0.75)
            features["engajamento_baixo"] = lambda x: 1 if x < limiar_engajamento else 0
        
        if "pct_atraso_total" in df.columns:
            media_atraso = desistentes["pct_atraso_total"].mean()
            features["risco_atraso"] = lambda x: min(x / media_atraso, 2) if x > 0 else 0
        
        def calcular_score(row):
            score = 0.6 * prob_mes.get(row["mes_curso"], 0)
            for feature, func in features.items():
                try:
                    score += 0.4/len(features) * func(row[feature])
                except:
                    pass
            return min(max(score, 0), 1)
        
        ativos["score_churn"] = ativos.apply(calcular_score, axis=1)
        
        cols = ["user_id", "score_churn", "mes_curso"] + list(features.keys())
        st.dataframe(
            ativos[cols].sort_values("score_churn", ascending=False)
            .style.background_gradient(subset=["score_churn"], cmap="Reds")
            .format({"score_churn": "{:.2%}"}),
            use_container_width=True
        )
        
        fig = px.histogram(ativos, x="score_churn", nbins=20,
                           labels={"score_churn": "Score de Churn", "count": "Número de Alunos"})
        st.plotly_chart(fig, use_container_width=True)
        
        csv = ativos[cols].to_csv(index=False)
        st.download_button(
            "📥 Baixar Scores",
            data=csv,
            file_name="scores_churn.csv",
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Erro ao calcular score de churn: {str(e)}")
        
