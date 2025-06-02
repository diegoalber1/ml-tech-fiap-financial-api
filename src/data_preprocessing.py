import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
import os

def download_data(symbol, start_date, end_date, save_path=None):
    df = yf.download(symbol, start=start_date, end=end_date)
    df = df[['Close']]
    if save_path:
        df.to_csv(save_path)
    return df

def preprocess_data(df, seq_length=60, scaler_path=None):
    scaler = MinMaxScaler()
    df['Close'] = scaler.fit_transform(df[['Close']])
    if scaler_path:
        joblib.dump(scaler, scaler_path)
    data = df['Close'].values
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y), scaler