from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import joblib
import mlflow
import flask_monitoringdashboard as dashboard

app = Flask(__name__)

# Carrega modelo e scaler
model = tf.keras.models.load_model("lstm_model.h5")
scaler = joblib.load("scaler.save")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']  # Espera lista de preços
    look_back = model.input_shape[1]
    if len(data) < look_back:
        return jsonify({'error': f'Forneça pelo menos {look_back} valores históricos.'}), 400

    # Prepara dados
    last_data = np.array(data[-look_back:]).reshape(-1, 1)
    last_data_scaled = scaler.transform(last_data)
    X_input = np.reshape(last_data_scaled, (1, look_back, 1))

    # Previsão
    pred_scaled = model.predict(X_input)
    pred = scaler.inverse_transform(pred_scaled)[0, 0]

    # Log MLflow (opcional)
    with mlflow.start_run(run_name="inference", nested=True):
        mlflow.log_param("input_length", len(data))
        mlflow.log_metric("prediction", pred)

    return jsonify({'prediction': float(pred)})

if __name__ == '__main__':
    app.run(debug=True)


dashboard.bind(app)