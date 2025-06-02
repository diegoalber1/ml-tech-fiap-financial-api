import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import os

def download_data(symbol, start_date, end_date, data_path=None):
    try:
        df = yf.download(symbol, start=start_date, end=end_date)
        if data_path and not (df.empty):
            # Garante que o diretório existe
            df = df[['Close']]
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            df.to_csv(data_path)
        else:
            if data_path and os.path.exists(data_path):
                print("Carregando dados salvos em disco...")
                return pd.read_csv(data_path, index_col=0)
            else:
                print("Arquivo de dados não encontrado em disco.")
                return None    
        return df
    except Exception as e:
        print(f"Erro ao baixar dados: {e}")
        if data_path and os.path.exists(data_path):
            print("Carregando dados salvos em disco...")
            return pd.read_csv(data_path, index_col=0)
        else:
            print("Arquivo de dados não encontrado em disco.")
            return None

def preprocess_data(df, seq_length=60, scaler_path=None):
    scaler = MinMaxScaler()
    df['Close'] = scaler.fit_transform(df[['Close']])
    if scaler_path:
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
    data = df['Close'].values
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y), scaler