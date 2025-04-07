import pandas as pd
from datetime import datetime, timedelta

def identificar_desistentes(clientes_df, pagamentos_df):
    hoje = pd.to_datetime("today").normalize()
    
    # Convertendo datas
    clientes_df["data inicio curso"] = pd.to_datetime(clientes_df["data inicio curso"])
    clientes_df["ultima data pagamento"] = pd.to_datetime(clientes_df["ultima data pagamento"])
    
    # Aluno desistente: não finalizou curso e último pagamento > 30 dias
    clientes_df["status atual"] = clientes_df["status atual"].str.lower()
    clientes_df["dias_sem_pagar"] = (hoje - clientes_df["ultima data pagamento"]).dt.days

    desistentes = clientes_df[
        (clientes_df["status atual"] != "concluído") &
        (clientes_df["dias_sem_pagar"] > 30)
    ].copy()
    
    ativos = clientes_df[
        (clientes_df["status atual"] == "ativo") &
        (clientes_df["dias_sem_pagar"] <= 30)
    ].copy()

    return desistentes, ativos

def calcular_idade(data_nasc):
    hoje = pd.to_datetime("today").normalize()
    return hoje.year - data_nasc.year - ((hoje.month, hoje.day) < (data_nasc.month, data_nasc.day))

def enriquecer_com_idade(df):
    df["data nascimento"] = pd.to_datetime(df["data nascimento"], errors="coerce")
    df["idade"] = df["data nascimento"].apply(lambda x: calcular_idade(x) if pd.notnull(x) else None)
    return df

def churn_por_mes(df):
    df = df.copy()
    df["mes_desistencia"] = df["ultima data pagamento"].dt.month
    return df.groupby("mes_desistencia").size()
