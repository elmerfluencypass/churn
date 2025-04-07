import pandas as pd

def load_all_data():
    # Google Drive CSV links
    urls = {
        "customer_profile_table": "https://drive.google.com/uc?id=1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT",
        "churn_detectado": "https://drive.google.com/uc?id=1kkdfrCjTjyzjqYfX7C9vDSdfod5HBYQS",
        "historico_pagamentos": "https://drive.google.com/uc?id=1j2RW-ryt4H3iX8nkaw371yBI0_EGA8MY",
        "tbl_insider": "https://drive.google.com/uc?id=1tPqnQWmowQKNAx2M_4rLW5axH9DZ5TyW",
        "iugu_invoices": "https://drive.google.com/uc?id=1eNcobHn8QJKduVRcs79LsbzT_2L7hK88",
    }

    # Leitura dos dados
    data = {name: pd.read_csv(url) for name, url in urls.items()}

    # Padronização de colunas
    for df in data.values():
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Dados base
    df_cadastro = data["customer_profile_table"].copy()
    df_pagamentos = data["historico_pagamentos"].copy()

    # Conversão de datas e correção do erro .dt com timezone
    df_cadastro["data_nascimento"] = pd.to_datetime(df_cadastro["data_nascimento"], errors="coerce").dt.tz_localize(None)
    df_cadastro["ultima_data_pagamento"] = pd.to_datetime(df_cadastro["ultima_data_pagamento"], errors="coerce")

    # Cálculo da idade
    hoje = pd.to_datetime("today").normalize()
    df_cadastro["idade"] = hoje.year - df_cadastro["data_nascimento"].dt.year

    # Cálculo de churn
    df_cadastro["status_atual"] = df_cadastro["status_atual"].str.lower()
    df_cadastro["dias_sem_pagar"] = (hoje - df_cadastro["ultima_data_pagamento"]).dt.days
    df_cadastro["churn"] = ((df_cadastro["status_atual"] != "concluído") & (df_cadastro["dias_sem_pagar"] > 30)).astype(int)

    # União dos dados
    df_unificado = df_pagamentos.merge(df_cadastro, on="user_id", how="left")

    # Renomear conforme modelo
    df_unificado.rename(columns={
        "mes": "mes_curso"
    }, inplace=True)

    # Seleção final de colunas conforme modelo PDF
    colunas_finais = [
        "user_id", "mes_curso", "churn", "idade", "sexo", "cidade", "estado",
        "plano_duracao_meses", "engagement_score", "pct_atraso_total",
        "qtd_meses_em_atraso", "valor_restante_contrato", "canal_aquisicao"
    ]

    # Garante que apenas colunas existentes sejam usadas
    colunas_existentes = [col for col in colunas_finais if col in df_unificado.columns]
    df_modelagem = df_unificado[colunas_existentes].copy()

    # Adiciona ao dicionário
    data["df_modelagem"] = df_modelagem

    return data
