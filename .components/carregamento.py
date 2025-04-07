import pandas as pd
import gdown
import os

CSV_URLS = {
    "cadastro_clientes": "1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT",
    "churn_detectado": "1kkdfrCjTjyzjqYfX7C9vDSdfod5HBYQS",
    "historico_pagamentos": "1j2RW-ryt4H3iX8nkaw371yBI0_EGA8MY",
    "tbl_insider": "1tPqnQWmowQKNAx2M_4rLW5axH9DZ5TyW",
    "temp_life_cycle": "1eNcobHn8QJKduVRcs79LsbzT_2L7hK88"
}

def baixar_dados():
    os.makedirs("data", exist_ok=True)
    arquivos = []
    for nome, file_id in CSV_URLS.items():
        path = f"data/{nome}.csv"
        url = f"https://drive.google.com/uc?id={file_id}"
        if not os.path.exists(path):
            gdown.download(url, path, quiet=False)
        arquivos.append(path)
    return arquivos

def carregar_dados():
    arquivos = baixar_dados()
    dfs = [pd.read_csv(f, parse_dates=True, dayfirst=True) for f in arquivos]
    df_geral = pd.concat(dfs, ignore_index=True)
    return df_geral
