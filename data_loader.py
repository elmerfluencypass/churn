import pandas as pd

def load_all_data():
    # Google Drive URLs (deve estar sempre atualizado com seus links)
    urls = {
        "customer_profile_table": "https://drive.google.com/uc?id=1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT",
        "churn_detectado": "https://drive.google.com/uc?id=1kkdfrCjTjyzjqYfX7C9vDSdfod5HBYQS",
        "historico_pagamentos": "https://drive.google.com/uc?id=1j2RW-ryt4H3iX8nkaw371yBI0_EGA8MY",
        "tbl_insider": "https://drive.google.com/uc?id=1tPqnQWmowQKNAx2M_4rLW5axH9DZ5TyW",
        "iugu_invoices": "https://drive.google.com/uc?id=1eNcobHn8QJKduVRcs79LsbzT_2L7hK88",
    }

    data = {
        name: pd.read_csv(url) for name, url in urls.items()
    }

    # Padronizar colunas para snake_case
    for df in data.values():
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Construção do DataFrame unificado para modelagem
    df_cadastro = data["customer_profile_table"].copy()
    df_pagamentos = data["historico_pagamentos"].copy()

    # Preparar datas
    df_cadastro["data_nascimento"] = pd.to_datetime(df_cadastro["data_nascimento"], errors="coerce")
    df_cadastro["ultima_data_pagamento"] = pd.to_datetime(df_cadastro["ultima_data_pagamento"], errors="coerce")

    # Calcular idade
    hoje = pd.to_datetime("today")
    df_cadastro["idade"] = hoje.year - df_cadastro["data_nascimento"].dt.year

    # Calcular churn (aluno ainda não concluiu e está com +30 dias sem pagar)
    df_cadastro["status_atual"] = df_cadastro["status_atual"].str.lower()
    df_cadastro["dias_sem_pagar"] = (hoje - df_cadastro["ultima_data_pagamento"]).dt.days
    df_cadastro["churn"] = ((df_cadastro["status_atual"] != "concluído") & (df_cadastro["dias_sem_pagar"] > 30)).astype(int)

    # Unir cadastros e pagamentos
    df_unificado = df_pagamentos.merge(
        df_cadastro,
        on="user_id",
        how="left"
    )

    # Renomear campos para o padrão do PDF
    df_unificado.rename(columns={
        "mes": "mes_curso",
        "sexo": "sexo",
        "cidade": "cidade",
        "estado": "estado",
        "plano_duracao_meses": "plano_duracao_meses",
        "engagement_score": "engagement_score",
        "pct_atraso_total": "pct_atraso_total",
        "qtd_meses_em_atraso": "qtd_meses_em_atraso",
        "valor_restante_contrato": "valor_restante_contrato",
        "canal_aquisicao": "canal_aquisicao"
    }, inplace=True)

    # Selecionar colunas desejadas
    colunas_finais = [
        "user_id", "mes_curso", "churn", "idade", "sexo", "cidade", "estado",
        "plano_duracao_meses", "engagement_score", "pct_atraso_total",
        "qtd_meses_em_atraso", "valor_restante_contrato", "canal_aquisicao"
    ]

    df_modelagem = df_unificado[colunas_finais].copy()

    # Retornar os dados brutos + o dataframe final
    data["df_modelagem"] = df_modelagem

    return data
