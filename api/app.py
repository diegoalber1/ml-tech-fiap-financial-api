from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import mlflow
import flask_monitoringdashboard as dashboard
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

# Carrega modelo e scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = tf.keras.models.load_model(os.path.join(BASE_DIR, "../saved_models/lstm_stock_model.keras"))
scaler = joblib.load(os.path.join(BASE_DIR, "../saved_models/scaler.save"))


@app.route('/predict', methods=['POST'])
def predict():

    print("Recebido request:", request.json)
    data = request.json['data']  # Espera lista de preços
    look_back = model.input_shape[1]
    print("look_back:", look_back)
    print("len(data):", len(data))
    if len(data) < look_back:
        return jsonify({'error': f'Forneça pelo menos {look_back} valores históricos.'}), 400

    # Prepara dados
    last_data = np.array(data[-look_back:]).reshape(-1, 1)
    last_data_scaled = scaler.transform(last_data)
    X_input = np.reshape(last_data_scaled, (1, look_back, 1))

    # Previsão
    pred_scaled = model.predict(X_input)
    pred = scaler.inverse_transform(pred_scaled)[0, 0]

    #Log MLflow
    with mlflow.start_run(run_name="inference", nested=True):
        mlflow.log_param("input_length", len(data))
        mlflow.log_metric("prediction", pred)

    return jsonify({'prediction': float(pred)})

dashboard.bind(app)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)


