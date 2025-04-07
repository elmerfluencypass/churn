import pandas as pd
import streamlit as st

CSV_LINKS = {
    "customer_profile_table": "https://drive.google.com/uc?id=1MLYWW5Axp_gGFXGF_mPZsK4vqTTBKDlT",
    "churn_detectado": "https://drive.google.com/uc?id=1kkdfrCjTjyzjqYfX7C9vDSdfod5HBYQS",
    "historico_pagamentos": "https://drive.google.com/uc?id=1j2RW-ryt4H3iX8nkaw371yBI0_EGA8MY",
    "tbl_insider": "https://drive.google.com/uc?id=1tPqnQWmowQKNAx2M_4rLW5axH9DZ5TyW",
    "iugu_invoices": "https://drive.google.com/uc?id=1eNcobHn8QJKduVRcs79LsbzT_2L7hK88"
}

@st.cache_data
def load_csv(name):
    url = CSV_LINKS[name]
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df
