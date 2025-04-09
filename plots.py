import streamlit as st
import pandas as pd
import plotly.express as px
from data_processing import identificar_desistentes, enriquecer_com_idade
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np

def show_churn_dashboard(data):
    clientes = data["customer_profile_table"]
    pagamentos = data["historico_pagamentos"]
    insider = data["tbl_insider"]
    
    # Enriquecer e filtrar desistentes
    clientes = enriquecer_com_idade(clientes)
    desistentes, _ = identificar_desistentes(clientes, pagamentos)
    
    # Merge com insider
    merged = desistentes.merge(insider, how="left", left_on="user_id", right_on="student_id")
    
    # Gráficos existentes
    st.subheader("📉 Churn por Mês")
    plot_churn_mensal(desistentes)
    
    st.subheader("👥 Distribuição Etária (Desistentes)")
    plot_distribuicao_etaria(desistentes)
    
    st.subheader("📍 Distribuição de Alunos Desistentes por UF")
    plot_uf_pizza(merged)
    
    st.subheader("🎓 Nível do Curso dos Alunos Desistentes")
    plot_last_level_pizza(merged)
    
    st.subheader("💬 Distribuição por Nome do Plano (Bolhas)")
    plot_plan_name_bubbles(merged)
    
    st.subheader("🎯 Objetivos Declarados dos Alunos Desistentes")
    plot_last_objective_bar(merged)
    
    # Nova matriz adicionada
    st.subheader("📊 Matriz: Desistência por Mês x Período do Curso")
    plot_matriz_periodo_mes(pagamentos, clientes)

# 1. Gráfico churn por mês
def plot_churn_mensal(df):
    df["mes_nome"] = pd.to_datetime(df["ultima_data_pagamento"], errors="coerce").dt.strftime("%b")
    contagem = df["mes_nome"].value_counts().sort_index().reset_index()
    contagem.columns = ["mes_nome", "desistencias"]
    fig = px.bar(contagem, x="mes_nome", y="desistencias", 
                 title="Desistências por Mês",
                 labels={"mes_nome": "Mês", "desistencias": "Número de Desistências"})
    st.plotly_chart(fig, use_container_width=True)

# 2. Gráfico distribuição etária
def plot_distribuicao_etaria(df):
    df = df[df["idade"].notnull() & (df["idade"] > 0)]
    fig = px.histogram(df, x="idade", nbins=20, 
                       title="Distribuição de Idade dos Desistentes",
                       labels={"idade": "Idade", "count": "Número de Alunos"})
    st.plotly_chart(fig, use_container_width=True)

# 3. Pizza por UF
def plot_uf_pizza(df):
    uf_counts = df["student_uf"].value_counts().reset_index()
    uf_counts.columns = ["UF", "Quantidade"]
    fig = px.pie(uf_counts, values="Quantidade", names="UF", 
                 title="Distribuição por UF",
                 hole=0.3)
    st.plotly_chart(fig, use_container_width=True)

# 4. Pizza por nível do curso
def plot_last_level_pizza(df):
    levels = df["last_level"].dropna().value_counts().reset_index()
    levels.columns = ["Nível", "Quantidade"]
    fig = px.pie(levels, values="Quantidade", names="Nível", 
                 title="Distribuição por Nível do Curso",
                 hole=0.3)
    st.plotly_chart(fig, use_container_width=True)

# 5. Bolhas por plano
def plot_plan_name_bubbles(df):
    plans = df["plan_name"].dropna().value_counts().reset_index()
    plans.columns = ["Plano", "Quantidade"]
    fig = px.scatter(plans, x="Plano", y="Quantidade", size="Quantidade", 
                     hover_name="Plano", title="Distribuição por Plano Contratado",
                     labels={"Plano": "Nome do Plano", "Quantidade": "Número de Alunos"})
    st.plotly_chart(fig, use_container_width=True)

# 6. Objetivos finais
def plot_last_objective_bar(df):
    objetivos = df["last_objective"].dropna().value_counts().reset_index()
    objetivos.columns = ["Objetivo", "Quantidade"]
    fig = px.bar(objetivos, x="Objetivo", y="Quantidade", 
                 title="Objetivos Declarados por Alunos Desistentes",
                 labels={"Objetivo": "Objetivo do Aluno", "Quantidade": "Número de Alunos"})
    st.plotly_chart(fig, use_container_width=True)

# 7. Matriz mês do ano x período do curso
def plot_matriz_periodo_mes(pagamentos_df, cadastro_df):
    # Preparar dados
    pagamentos_df["user_id"] = pagamentos_df["user_id"].astype(str)
    cadastro_df["user_id"] = cadastro_df["user_id"].astype(str)
    
    # Identificar desistentes
    hoje = pd.to_datetime("today").normalize()
    cadastro_df["ultima_data_pagamento"] = pd.to_datetime(cadastro_df["ultima_data_pagamento"], errors="coerce")
    cadastro_df["dias_sem_pagar"] = (hoje - cadastro_df["ultima_data_pagamento"]).dt.days
    cadastro_df["status_atual"] = cadastro_df["status_atual"].str.lower()
    
    desistentes = cadastro_df[
        (cadastro_df["status_atual"] != "concluído") &
        (cadastro_df["dias_sem_pagar"] > 30)
    ].copy()
    
    # Filtrar pagamentos dos desistentes
    pagamentos_filtrados = pagamentos_df[pagamentos_df["user_id"].isin(desistentes["user_id"])].copy()
    pagamentos_filtrados["data_prevista_pagamento"] = pd.to_datetime(pagamentos_filtrados["data_prevista_pagamento"], errors="coerce")
    pagamentos_filtrados["mes_calendario"] = pagamentos_filtrados["data_prevista_pagamento"].dt.strftime("%B")
    pagamentos_filtrados["mes_curso"] = pagamentos_filtrados["mes_curso"].astype(str)
    
    # Criar matriz
    matriz = pd.pivot_table(
        pagamentos_filtrados,
        values="user_id",
        index="mes_calendario",
        columns="mes_curso",
        aggfunc="count",
        fill_value=0
    )
    
    # Ordenar meses do ano
    ordem_meses = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    matriz = matriz.reindex(ordem_meses)
    
    # Ordenar meses do curso numericamente
    matriz = matriz[sorted(matriz.columns, key=lambda x: int(x) if x.isdigit() else 0)]
    
    # Plotar
    fig = px.imshow(
        matriz,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Matriz de Desistência: Mês do Ano vs Período do Curso",
        aspect="auto",
        labels=dict(x="Período do Curso (meses)", y="Mês do Ano", color="Desistências")
    )
    st.plotly_chart(fig, use_container_width=True)

def show_churn_profile(data):
    st.title("🔍 Perfil de Churn")
    
    df = data["df_modelagem"]
    df = df[df["churn"] == 1].copy()
    
    # Verificar dados mínimos
    if len(df) < 10:
        st.warning("Dados insuficientes para análise de perfil. Necessário pelo menos 10 registros.")
        return
    
    # Selecionar features numéricas relevantes
    features = ["idade", "engagement_score", "pct_atraso_total", "valor_restante_contrato"]
    features = [f for f in features if f in df.columns]
    
    if len(features) < 2:
        st.warning("Features insuficientes para análise. Necessário pelo menos 2 variáveis numéricas.")
        return
    
    # Remover outliers e nulos
    df = df.dropna(subset=features)
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df = df[(df[col] >= q1 - 1.5*iqr) & (df[col] <= q3 + 1.5*iqr)]
    
    # Clusterização
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
    
    # Plot
    fig = px.scatter(
        df, x="PCA1", y="PCA2", color="Cluster",
        hover_data=["user_id"] + features,
        title="Clusterização de Alunos Desistentes",
        labels={"PCA1": "Componente Principal 1", "PCA2": "Componente Principal 2"}
    )
    st.plotly_chart(fig, use_container_width=True)

def calcular_score_de_churn(df_modelagem):
    st.title("🧠 Score de Propensão à Desistência")
    
    df = df_modelagem.copy()
    df = df.dropna(subset=["user_id", "mes_curso"])
    df["mes_curso"] = df["mes_curso"].astype(int)
    
    # Separar desistentes e ativos
    desistentes = df[df["churn"] == 1].copy()
    ativos = df[df["churn"] == 0].copy()
    
    if len(desistentes) < 10 or len(ativos) < 10:
        st.warning("Dados insuficientes para cálculo de score. Necessário pelo menos 10 registros em cada grupo.")
        return
    
    # Calcular probabilidades base
    prob_mes = desistentes.groupby("mes_curso").size() / df.groupby("mes_curso").size()
    prob_mes = prob_mes.fillna(0).to_dict()
    
    # Features adicionais
    features = {}
    if "engagement_score" in df.columns:
        limiar_engajamento = desistentes["engagement_score"].quantile(0.75)
        features["engajamento_baixo"] = lambda x: 1 if x < limiar_engajamento else 0
    
    if "pct_atraso_total" in df.columns:
        media_atraso = desistentes["pct_atraso_total"].mean()
        features["risco_atraso"] = lambda x: min(x / media_atraso, 2) if x > 0 else 0
    
    # Calcular score
    def calcular_score(row):
        score = 0.6 * prob_mes.get(row["mes_curso"], 0)
        
        for feature, func in features.items():
            try:
                score += 0.4/len(features) * func(row[feature])
            except:
                pass
        
        return min(max(score, 0), 1)
    
    ativos["score_churn"] = ativos.apply(calcular_score, axis=1)
    
    # Exibir resultados
    st.subheader("📋 Scores Calculados")
    cols = ["user_id", "score_churn", "mes_curso"] + list(features.keys())
    st.dataframe(
        ativos[cols].sort_values("score_churn", ascending=False)
        .style.background_gradient(subset=["score_churn"], cmap="Reds")
        .format({"score_churn": "{:.2%}"}),
        use_container_width=True
    )
    
    # Gráfico de distribuição
    st.subheader("📊 Distribuição dos Scores")
    fig = px.histogram(ativos, x="score_churn", nbins=20,
                       labels={"score_churn": "Score de Churn", "count": "Número de Alunos"})
    st.plotly_chart(fig, use_container_width=True)
    
    # Download
    csv = ativos[cols].to_csv(index=False)
    st.download_button(
        "📥 Baixar Scores",
        data=csv,
        file_name="scores_churn.csv",
        mime="text/csv"
    )

def calcular_variaveis_score(df_modelagem):
    st.title("📊 Matriz de Variáveis para Churn Score")

    df = df_modelagem.copy()
    df = df[df["churn"] == 1].copy()

    if df.empty:
        st.warning("Não há registros de alunos desistentes suficientes para gerar variáveis.")
        return

    # Criar coluna de tempo até churn com base no número de parcelas pagas
    df["tempo_ate_churn"] = df.groupby("user_id")["mes_curso"].transform("max").astype(float)

    # Agregações temporais
    variaveis = df.groupby("user_id").agg({
        "idade": "first",
        "estado": "first" if "estado" in df.columns else "count",
        "plano_duracao_meses": "first",
        "engagement_score": ["mean", "std", "last"],
        "pct_atraso_total": "mean",
        "qtd_meses_em_atraso": "mean",
        "valor_restante_contrato": "last",
        "tempo_ate_churn": "first"
    })

    # Ajustar colunas multiindex
    variaveis.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in variaveis.columns]
    variaveis.reset_index(inplace=True)

    # Derivar nova variável: tendência de queda no engajamento
    variaveis["variacao_engajamento"] = variaveis["engagement_score_last"] - variaveis["engagement_score_mean"]

    # Renomear para clareza
    variaveis.rename(columns={
        "idade_first": "idade",
        "estado_first": "estado",
        "plano_duracao_meses_first": "plano",
        "engagement_score_mean": "media_engajamento",
        "engagement_score_std": "desvio_engajamento",
        "pct_atraso_total_mean": "media_pct_atraso",
        "qtd_meses_em_atraso_mean": "media_meses_atraso",
        "valor_restante_contrato_last": "valor_restante",
        "tempo_ate_churn_first": "tempo_ate_churn"
    }, inplace=True)

    st.dataframe(variaveis.head(50), use_container_width=True)
    st.success("✅ Matriz de variáveis criada com sucesso.")
