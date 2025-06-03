#!/bin/bash
# Inicia o Flask em background
python api/app.py &

# Inicia o MLflow em background
mlflow ui --host 0.0.0.0 --port 5001 --backend-store-uri "$(pwd)/notebooks/mlruns" &

# Espera qualquer processo terminar (mantém o container vivo)
wait -n