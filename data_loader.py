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

    # Padronizar nomes de colunas
    for df in data.values():
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]

    # Bases principais
    df_cadastro = data["customer_profile_table"].copy()
    df_pagamentos = data["historico_pagamentos"].copy()

    # --- Conversão robusta da data_nascimento (sem timezone) ---
    data_nascimento = pd.to_datetime(df_cadastro["data_nascimento"], errors="coerce")
    data_nascimento = data_nascimento.apply(
        lambda x: x.tz_localize(None) if pd.notnull(x) and hasattr(x, 'tz_localize') and x.tzinfo else x
    )
    df_cadastro["data_nascimento"] = pd.to_datetime(data_nascimento)

    # Conversão de outras datas
    df_cadastro["ultima_data_pagamento"] = pd.to_datetime(df_cadastro["ultima_data_pagamento"], errors="coerce")

    # Cálculo da idade
    hoje = pd.to_datetime("today").normalize()
    df_cadastro["idade"] = hoje.year - df_cadastro["data_nascimento"].dt.year

    # Cálculo de churn
    df_cadastro["status_atual"] = df_cadastro["status_atual"].str.lower()
    df_cadastro["dias_sem_pagar"] = (hoje - df_cadastro["ultima_data_pagamento"]).dt.days
    df_cadastro["churn"] = ((df_cadastro["status_atual"] != "concluído") & (df_cadastro["dias_sem_pagar"] > 30)).astype(int)

    # União entre pagamentos e cadastro
    df_unificado = df_pagamentos.merge(df_cadastro, on="user_id", how="left")

    # Renomear mes → mes_curso
    df_unificado.rename(columns={"mes": "mes_curso"}, inplace=True)

    # Colunas esperadas no modelo
    colunas_finais = [
        "user_id", "mes_curso", "churn", "idade", "sexo", "cidade", "estado",
        "plano_duracao_meses", "engagement_score", "pct_atraso_total",
        "qtd_meses_em_atraso", "valor_restante_contrato", "canal_aquisicao"
    ]

    # Selecionar apenas colunas existentes
    colunas_existentes = [col for col in colunas_finais if col in df_unificado.columns]
    df_modelagem = df_unificado[colunas_existentes].copy()

    # Adicionar ao dicionário
    data["df_modelagem"] = df_modelagem

    return data
