import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.data_preprocessing import download_data, preprocess_data
from src.model import build_lstm
import tensorflow as tf
import os
import joblib
import mlflow
import mlflow.tensorflow

symbol = 'DIS'
start_date = '2025-01-01'
end_date = '2025-06-01'
seq_length = 60
data_path = f"../data/{symbol}_{start_date}_{end_date}.csv"

df = download_data(symbol, start_date, end_date, data_path)
print(df.head())

X, y, scaler = preprocess_data(df, seq_length, scaler_path='../saved_models/scaler.save')

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

mlflow.set_experiment("lstm-stock-predictor")
with mlflow.start_run():
    # Log de parâmetros
    mlflow.log_param("symbol", symbol)
    mlflow.log_param("seq_length", seq_length)
    mlflow.log_param("epochs", 20)
    mlflow.log_param("batch_size", 32)

    # Ativa autolog para Keras/TensorFlow
    mlflow.tensorflow.autolog()

    model = build_lstm(seq_length)
    history = model.fit(
        X_train.reshape(-1, seq_length, 1), y_train,
        epochs=20, batch_size=32,
        validation_data=(X_test.reshape(-1, seq_length, 1), y_test)
    )

    # Salva modelo e scaler
    model_path = '../saved_models/lstm_stock_model.h5'
    scaler_path = '../saved_models/scaler.save'
    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    mlflow.log_artifact(model_path)
    mlflow.log_artifact(scaler_path)

    # Avaliação
    y_pred = model.predict(X_test.reshape(-1, seq_length, 1))
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Log de métricas
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)