import streamlit as st
import pandas as pd
import gdown
import plotly.express as px

def run():
    st.header("💸 Receita Perdida por Aluno Desistente")

    FILE_IDS = {
        "iugu_invoices": "1M16LUHoXHUf_o4JJcE4EuZvkkzwawol2",
        "iugu_subscription": "1kMq0ydf7TL92wz_60wpfTf-obu0IJXtK",
        "cadastro_clientes": "1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT"
    }

    @st.cache_data
    def download_csv(name, file_id):
        url = f"https://drive.google.com/uc?id={file_id}"
        output = f"{name}.csv"
        gdown.download(url, output, quiet=True)
        df = pd.read_csv(output, low_memory=False)
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        return df

    invoices = download_csv("iugu_invoices", FILE_IDS["iugu_invoices"])
    subs = download_csv("iugu_subscription", FILE_IDS["iugu_subscription"])
    cadastro = download_csv("cadastro_clientes", FILE_IDS["cadastro_clientes"])

    if 'due_date' not in invoices.columns:
        st.error("Coluna 'due_date' não encontrada na iugu_invoices.csv.")
        return
    invoices['due_date'] = pd.to_datetime(invoices['due_date'], errors='coerce')
    invoices['customer_name_lower'] = invoices['customer_name'].str.strip().str.lower()

    subs['customer_name_lower'] = subs['customer_name'].str.strip().str.lower()
    subs['price_total'] = subs['price_cents'].fillna(0) / 100
    subs['max_cycles'] = subs['max_cycles'].fillna(12)
    subs['preco_mensal_estimado'] = subs['price_total'] / subs['max_cycles']

    cadastro['nome_lower'] = cadastro['nome'].str.strip().str.lower()
    cadastro['data_inicio_curso'] = pd.to_datetime(cadastro.get('data_inicio_curso'), errors='coerce')

    hoje = pd.to_datetime("today")
    pagas = invoices[invoices['status'] == 'paid']
    pag_por_aluno = pagas.groupby('customer_name_lower').agg(
        qtd_pagamentos=('paid_value', 'count'),
        ultima_data_pagamento=('due_date', 'max')
    ).reset_index()

    df = cadastro.merge(pag_por_aluno, left_on='nome_lower', right_on='customer_name_lower', how='left')
    df = df.merge(subs[['customer_name_lower', 'preco_mensal_estimado', 'max_cycles', 'price_total']], on='customer_name_lower', how='left')

    df['qtd_pagamentos'] = df['qtd_pagamentos'].fillna(0)
    df['preco_mensal_estimado'] = df['preco_mensal_estimado'].fillna(0)
    df['max_cycles'] = df['max_cycles'].fillna(12)
    df['meses_restantes'] = (df['max_cycles'] - df['qtd_pagamentos']).clip(lower=0)
    df['receita_perdida'] = df['meses_restantes'] * df['preco_mensal_estimado']
    df['mes_desistencia'] = df['ultima_data_pagamento'].dt.month_name()
    df['periodo_curso'] = ((df['ultima_data_pagamento'] - df['data_inicio_curso']) / pd.Timedelta(days=30)).fillna(0).astype(int)

    df_desistentes = df[
        (df['ultima_data_pagamento'] < hoje - pd.Timedelta(days=30)) &
        (df.get('status_atual', '').astype(str).str.lower() != "concluído")
    ].copy()

    st.metric("🧑‍🎓 Alunos Desistentes", df_desistentes.get('user_id', pd.Series()).nunique())
    st.metric("💰 Receita Perdida Total (R$)", f"{df_desistentes['receita_perdida'].sum():,.2f}")

    alunos_unicos = sorted(df_desistentes['nome'].dropna().unique()) if 'nome' in df_desistentes else []
    selecionados = st.multiselect("🎯 Filtrar por aluno(s)", alunos_unicos, default=alunos_unicos)
    df_filtrado = df_desistentes[df_desistentes['nome'].isin(selecionados)] if 'nome' in df_desistentes else df_desistentes

    matriz = df_filtrado.pivot_table(
        index='mes_desistencia',
        columns='periodo_curso',
        values='receita_perdida',
        aggfunc='sum',
        fill_value=0
    )

    st.subheader("📊 Matriz de Receita Perdida por Mês e Período")
    st.dataframe(matriz.round(2))

    fig = px.imshow(matriz, text_auto=True, color_continuous_scale='Reds',
                    labels=dict(x="Período (meses)", y="Mês Desistência", color="R$ Perdido"))
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("📥 Baixar CSV", matriz.to_csv().encode('utf-8'), "matriz_receita.csv", "text/csv")
