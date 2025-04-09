# Atualizando o arquivo data_processing.py
import pandas as pd
from datetime import datetime

def identificar_desistentes(clientes_df, pagamentos_df):
    hoje = pd.to_datetime("today").normalize()

    # Conversão segura de datas
    clientes_df["data_inicio_curso"] = pd.to_datetime(clientes_df["data_inicio_curso"], errors="coerce")
    clientes_df["ultima_data_pagamento"] = pd.to_datetime(clientes_df["ultima_data_pagamento"], errors="coerce")
    
    # Normalização de strings
    clientes_df["status_atual"] = clientes_df["status_atual"].str.lower().str.strip()
    
    # Cálculo de dias sem pagar
    clientes_df["dias_sem_pagar"] = (hoje - clientes_df["ultima_data_pagamento"]).dt.days.fillna(999)
    
    # Identificação de desistentes
    desistentes = clientes_df[
        (clientes_df["status_atual"] != "concluído") &
        (clientes_df["dias_sem_pagar"] > 30) &
        (clientes_df["plano_duracao_meses"] > 0)
    ].copy()
    
    # Identificação de ativos
    ativos = clientes_df[
        (clientes_df["status_atual"] == "ativo") &
        (clientes_df["dias_sem_pagar"] <= 30)
    ].copy()
    
    return desistentes, ativos

def enriquecer_com_idade(df):
    df["data_nascimento"] = pd.to_datetime(df["data_nascimento"], errors="coerce")
    hoje = pd.to_datetime("today").normalize()
    df["idade"] = ((hoje - df["data_nascimento"]).dt.days / 365.25).round().astype('Int64')
    df["idade"] = df["idade"].apply(lambda x: x if 10 < x < 120 else None)
    return df

def preparar_dados_modelagem(clientes_df, pagamentos_df):
    # Garantir tipos corretos
    pagamentos_df["user_id"] = pagamentos_df["user_id"].astype(str)
    clientes_df["user_id"] = clientes_df["user_id"].astype(str)
    
    # Merge das tabelas
    df = pagamentos_df.merge(clientes_df, on="user_id", how="left")
    
    # Engenharia de features
    df["mes_curso"] = df["mes_curso"].astype(int)
    df["churn"] = ((df["status_atual"] != "concluído") & 
                   ((pd.to_datetime("today") - df["ultima_data_pagamento"]).dt.days > 30).astype(int)
    
    # Selecionar colunas relevantes
    colunas_modelagem = [
        "user_id", "mes_curso", "churn", "idade", "sexo", "cidade", "estado",
        "plano_duracao_meses", "engagement_score", "pct_atraso_total",
        "qtd_meses_em_atraso", "valor_restante_contrato", "canal_aquisicao"
    ]
    
    # Manter apenas colunas disponíveis
    colunas_existentes = [col for col in colunas_modelagem if col in df.columns]
    return df[colunas_existentes].copy()
