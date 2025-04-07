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

    # Carregar todos os arquivos e padronizar colunas
    data = {}
    for name, url in urls.items():
        df = pd.read_csv(url)
        df.columns = [col.strip().lower().replace(" ", "_") for col in df.columns]
        data[name] = df

    # Base de cadastro
    df_cadastro = data["customer_profile_table"].copy()
    df_pagamentos = data["historico_pagamentos"].copy()

    # Garantir que 'user_id' seja string para evitar erro no merge
    df_cadastro["user_id"] = df_cadastro["user_id"].astype(str)
    df_pagamentos["user_id"] = df_pagamentos["user_id"].astype(str)

    # Limpar valores inválidos antes da conversão
    df_cadastro["data_nascimento"] = df_cadastro["data_nascimento"].replace("-infinity", pd.NaT)
    df_cadastro["data_nascimento"] = pd.to_datetime(df_cadastro["data_nascimento"], errors="coerce")
    df_cadastro["ultima_data_pagamento"] = pd.to_datetime(df_cadastro["ultima_data_pagamento"], errors="coerce")

    # Calcular idade apenas se possível
    hoje = pd.to_datetime("today").normalize()
    df_cadastro["idade"] = df_cadastro["data_nascimento"].apply(
        lambda d: hoje.year - d.year if pd.notnull(d) else None
    )

    # Calcular churn
    df_cadastro["status_atual"] = df_cadastro["status_atual"].str.lower()
    df_cadastro["dias_sem_pagar"] = (hoje - df_cadastro["ultima_data_pagamento"]).dt.days
    df_cadastro["churn"] = ((df_cadastro["status_atual"] != "concluído") & (df_cadastro["dias_sem_pagar"] > 30)).astype(int)

    # Garantir consistência no campo 'mes' do histórico
    if "mes" in df_pagamentos.columns:
        df_pagamentos.rename(columns={"mes": "mes_curso"}, inplace=True)

    # Merge robusto e seguro
    df_unificado = pd.merge(df_pagamentos, df_cadastro, on="user_id", how="left")

    # Campos definidos no PDF
    colunas_modelagem = [
        "user_id", "mes_curso", "churn", "idade", "sexo", "cidade", "estado",
        "plano_duracao_meses", "engagement_score", "pct_atraso_total",
        "qtd_meses_em_atraso", "valor_restante_contrato", "canal_aquisicao"
    ]

    # Selecionar apenas as colunas existentes
    colunas_existentes = [col for col in colunas_modelagem if col in df_unificado.columns]
    df_modelagem = df_unificado[colunas_existentes].copy()

    # Adicionar ao dicionário
    data["df_modelagem"] = df_modelagem

    return data
