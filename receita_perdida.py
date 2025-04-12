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

    invoices['paid_at'] = pd.to_datetime(invoices['paid_at'], errors='coerce')
    invoices['customer_name_lower'] = invoices['customer_name'].str.strip().str.lower()
    subs['customer_name_lower'] = subs['customer_name'].str.strip().str.lower()
    subs['preco_mensal'] = subs['price_cents'] / 100
    subs['max_cycles'] = subs['max_cycles'].fillna(12)
    cadastro['nome_lower'] = cadastro['nome'].str.strip().str.lower()

    cadastro['data_inicio_curso'] = pd.to_datetime(cadastro.get('data_inicio_curso'), errors='coerce')
    cadastro['ultima_data_pagamento'] = pd.to_datetime(cadastro.get('ultima_data_pagamento'), errors='coerce')

    hoje = pd.to_datetime("today")
    desistentes = cadastro[
        (cadastro.get('status_atual', '').astype(str).str.lower() != "concluído") &
        (cadastro['ultima_data_pagamento'] < hoje - pd.Timedelta(days=30))
    ].copy()

    pagas = invoices[invoices['status'] == 'paid']
    pag_por_aluno = pagas.groupby('customer_name_lower').agg(
        qtd_pagamentos=('paid_value', 'count'),
        ultima_data_pagamento=('paid_at', 'max')
    ).reset_index()

    df = desistentes.merge(pag_por_aluno, left_on='nome_lower', right_on='customer_name_lower', how='left')
    df = df.merge(subs[['customer_name_lower', 'preco_mensal', 'max_cycles']], on='customer_name_lower', how='left')

    df['qtd_pagamentos'] = df['qtd_pagamentos'].fillna(0)
    df['preco_mensal'] = df['preco_mensal'].fillna(0)
    df['max_cycles'] = df['max_cycles'].fillna(12)
    df['meses_restantes'] = (df['max_cycles'] - df['qtd_pagamentos']).clip(lower=0)
    df['receita_perdida'] = df['meses_restantes'] * df['preco_mensal']
    df['mes_desistencia'] = df['ultima_data_pagamento'].dt.month_name()
    df['periodo_curso'] = ((df['ultima_data_pagamento'] - df['data_inicio_curso']) / pd.Timedelta(days=30)).fillna(0).astype(int)

    st.metric("🧑‍🎓 Alunos Desistentes", df.get('user_id', pd.Series()).nunique())
    st.metric("💰 Receita Perdida Total (R$)", f"{df['receita_perdida'].sum():,.2f}")

    alunos_unicos = sorted(df['nome'].dropna().unique()) if 'nome' in df else []
    selecionados = st.multiselect("🎯 Filtrar por aluno(s)", alunos_unicos, default=alunos_unicos)
    df_filtrado = df[df['nome'].isin(selecionados)] if 'nome' in df else df

    matriz = df_filtrado.pivot_table(
        index='mes_desistencia',
        columns='periodo_curso',
        values='receita_perdida',
        aggfunc='sum',
        fill_value=0
    )
    st.subheader("📊 Matriz de Receita Perdida")
    st.dataframe(matriz.round(2))
    fig = px.imshow(matriz, text_auto=True, color_continuous_scale='Reds',
                    labels=dict(x="Período (meses)", y="Mês Desistência", color="R$ Perdido"))
    st.plotly_chart(fig, use_container_width=True)
    st.download_button("📥 Baixar CSV", matriz.to_csv().encode('utf-8'), "matriz_receita.csv", "text/csv")
