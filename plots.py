import streamlit as st
import pandas as pd
import plotly.express as px
from data_processing import identificar_desistentes, enriquecer_com_idade
from sklearn.preprocessing import StandardScaler


def show_churn_dashboard(data):
    clientes = data["customer_profile_table"]
    pagamentos = data["historico_pagamentos"]
    insider = data["tbl_insider"]
    churn = data["churn_detectado"]

    # Enriquecer e filtrar desistentes
    clientes = enriquecer_com_idade(clientes)
    desistentes, _ = identificar_desistentes(clientes, pagamentos)

    # Merge com insider
    merged = desistentes.merge(insider, how="left", left_on="user_id", right_on="student_id")

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

    st.subheader("📊 Matriz: Desistência por Mês x Período do Curso")
    plot_matriz_periodo_mes(data["historico_pagamentos"], data["customer_profile_table"])

# 1. Gráfico churn por mês
def plot_churn_mensal(df):
    df["mes_nome"] = pd.to_datetime(df["ultima_data_pagamento"], errors="coerce").dt.strftime("%b")
    contagem = df["mes_nome"].value_counts().sort_index().reset_index()
    contagem.columns = ["mes_nome", "desistencias"]
    fig = px.bar(contagem, x="mes_nome", y="desistencias", title="Desistências por Mês")
    st.plotly_chart(fig, use_container_width=True)

# 2. Gráfico distribuição etária
def plot_distribuicao_etaria(df):
    df = df[df["idade"].notnull() & (df["idade"] > 0)]
    fig = px.histogram(df, x="idade", nbins=20, title="Distribuição de Idade dos Desistentes")
    st.plotly_chart(fig, use_container_width=True)

# 3. Pizza por UF
def plot_uf_pizza(df):
    uf_counts = df["student_uf"].value_counts().reset_index()
    uf_counts.columns = ["UF", "Quantidade"]
    fig = px.pie(uf_counts, values="Quantidade", names="UF", title="Distribuição por UF")
    st.plotly_chart(fig, use_container_width=True)

# 4. Pizza por nível do curso
def plot_last_level_pizza(df):
    levels = df["last_level"].dropna().value_counts().reset_index()
    levels.columns = ["Nível", "Quantidade"]
    fig = px.pie(levels, values="Quantidade", names="Nível", title="Distribuição por Nível do Curso")
    st.plotly_chart(fig, use_container_width=True)

# 5. Bolhas por plano
def plot_plan_name_bubbles(df):
    plans = df["plan_name"].dropna().value_counts().reset_index()
    plans.columns = ["Plano", "Quantidade"]
    fig = px.scatter(plans, x="Plano", y="Quantidade", size="Quantidade", hover_name="Plano",
                     title="Distribuição por Plano Contratado")
    st.plotly_chart(fig, use_container_width=True)

# 6. Objetivos finais
def plot_last_objective_bar(df):
    objetivos = df["last_objective"].dropna().value_counts().reset_index()
    objetivos.columns = ["Objetivo", "Quantidade"]
    fig = px.bar(objetivos, x="Objetivo", y="Quantidade", title="Objetivos Declarados por Alunos Desistentes")
    st.plotly_chart(fig, use_container_width=True)

# 7. Matriz mês do ano x período do curso (nova lógica)
def plot_matriz_periodo_mes(pagamentos_df, cadastro_df):
    pagamentos_df["user_id"] = pagamentos_df["user_id"].astype(str)
    cadastro_df["user_id"] = cadastro_df["user_id"].astype(str)

    hoje = pd.to_datetime("today").normalize()
    cadastro_df["ultima_data_pagamento"] = pd.to_datetime(cadastro_df["ultima_data_pagamento"], errors="coerce")
    cadastro_df["dias_sem_pagar"] = (hoje - cadastro_df["ultima_data_pagamento"]).dt.days
    cadastro_df["status_atual"] = cadastro_df["status_atual"].str.lower()

    desistentes = cadastro_df[
        (cadastro_df["status_atual"] != "concluído") &
        (cadastro_df["dias_sem_pagar"] > 30)
    ].copy()

    pagamentos_filtrados = pagamentos_df[pagamentos_df["user_id"].isin(desistentes["user_id"])].copy()
    pagamentos_filtrados["data_prevista_pagamento"] = pd.to_datetime(pagamentos_filtrados["data_prevista_pagamento"], errors="coerce")
    pagamentos_filtrados["mes_calendario"] = pagamentos_filtrados["data_prevista_pagamento"].dt.strftime("%B")
    pagamentos_filtrados["mes"] = pagamentos_filtrados["mes"].astype(str)

    matriz = pd.pivot_table(
        pagamentos_filtrados,
        values="user_id",
        index="mes_calendario",
        columns="mes",
        aggfunc="count",
        fill_value=0
    )

    ordem_meses = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    matriz = matriz.reindex(ordem_meses)

    fig = px.imshow(
        matriz,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Matriz de Desistência por Mês do Ano vs Período do Curso",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)
    from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px

def show_churn_profile(data):
    st.title("🔍 Perfil de Churn")

    df = data["df_modelagem"]
    df = df[df["churn"] == 1].copy()

    # Selecionar features numéricas relevantes
    features = ["idade", "engagement_score", "pct_atraso_total", "valor_restante_contrato"]
    df = df.dropna(subset=features)

    # Amostragem estratificada por estado, se disponível
    if "estado" in df.columns:
        df = df.groupby("estado", group_keys=False).apply(lambda x: x.sample(min(len(x), 50), random_state=42))
    else:
        df = df.sample(n=min(len(df), 200), random_state=42)

    # Escalonar e reduzir dimensionalidade
    X = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # Clusterização
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    df["PCA1"] = X_pca[:, 0]
    df["PCA2"] = X_pca[:, 1]
    df["Cluster"] = clusters.astype(str)

    # Gráfico interativo
    fig = px.scatter(
        df, x="PCA1", y="PCA2", color="Cluster",
        hover_data=["user_id", "idade", "estado"] if "estado" in df.columns else ["user_id", "idade"],
        title="Clusterização de Alunos Desistentes"
    )
    st.plotly_chart(fig, use_container_width=True)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px

def show_churn_profile(data):
    st.title("🔍 Perfil de Churn")

    df = data["df_modelagem"]
    df = df[df["churn"] == 1].copy()

    # Candidatas à clusterização
    candidatas = ["idade", "engagement_score", "pct_atraso_total", "valor_restante_contrato"]

    # Seleciona variáveis com pelo menos algum valor numérico válido
    colunas_validas = []
    for col in candidatas:
        if col in df.columns and df[col].dropna().apply(lambda x: isinstance(x, (int, float))).sum() > 0:
            colunas_validas.append(col)

    st.write(f"✅ Variáveis disponíveis para clusterização: {colunas_validas}")

    if len(colunas_validas) < 2:
        st.warning("⚠️ Não há colunas com dados suficientes para realizar a clusterização.")
        return

    # Manter apenas registros com dados nessas colunas
    df = df.dropna(subset=colunas_validas)

    # Amostragem (pode ser total se pouco dado)
    amostra = df.copy()
    if "estado" in df.columns and df["estado"].nunique() > 1:
        amostra = df.groupby("estado", group_keys=False).apply(lambda x: x.sample(min(len(x), 50), random_state=42))
    elif len(df) > 200:
        amostra = df.sample(n=200, random_state=42)

    # Pré-processamento
    X = amostra[colunas_validas]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA + KMeans
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    amostra["PCA1"] = X_pca[:, 0]
    amostra["PCA2"] = X_pca[:, 1]
    amostra["Cluster"] = clusters.astype(str)

    # Gráfico
    fig = px.scatter(
        amostra,
        x="PCA1",
        y="PCA2",
        color="Cluster",
        hover_data=["user_id"] + [c for c in ["idade", "estado"] if c in amostra.columns],
        title="Clusterização de Alunos Desistentes"
    )
    st.plotly_chart(fig, use_container_width=True)



