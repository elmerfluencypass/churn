import pandas as pd
import gdown
import os

# Lista de arquivos e seus IDs do Google Drive
CSV_URLS = {
    "alunos_1": "1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT",
    "alunos_2": "1kkdfrCjTjyzjqYfX7C9vDSdfod5HBYQS",
    "alunos_3": "1j2RW-ryt4H3iX8nkaw371yBI0_EGA8MY",
    "alunos_4": "1tPqnQWmowQKNAx2M_4rLW5axH9DZ5TyW",
    "alunos_5": "1eNcobHn8QJKduVRcs79LsbzT_2L7hK88"
}

def baixar_dados():
    os.makedirs("data", exist_ok=True)
    arquivos = []
    
    for nome, file_id in CSV_URLS.items():
        output_path = f"data/{nome}.csv"
        url = f"https://drive.google.com/uc?id={file_id}"
        if not os.path.exists(output_path):
            gdown.download(url, output_path, quiet=False)
        arquivos.append(output_path)
    
    return arquivos

def carregar_dados():
    arquivos = baixar_dados()
    dfs = [pd.read_csv(arq, parse_dates=["data_evento"]) for arq in arquivos]
    df_geral = pd.concat(dfs, ignore_index=True)
    return df_geral
