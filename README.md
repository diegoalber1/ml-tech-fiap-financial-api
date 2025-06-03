# 📈 LSTM Stock Price Predictor

Este projeto implementa um modelo preditivo baseado em redes neurais LSTM para prever o valor de fechamento de ações. O pipeline cobre desde a coleta de dados históricos até o deploy do modelo em uma API RESTful com monitoramento e rastreamento de inferências.

---

## 📌 Funcionalidades

- Coleta automática de dados históricos via [Yahoo Finance](https://finance.yahoo.com/)
- Pré-processamento com `MinMaxScaler`
- Modelo de rede neural LSTM treinado com `TensorFlow`
- Monitoramento com `Flask Monitoring Dashboard`
- Rastreamento de experimentos com `MLflow`
- API RESTful em Flask para inferência
- Pronto para deploy com Docker

---

## 🧠 Tecnologias Usadas

- Python 3.10+
- TensorFlow / Keras
- Scikit-learn
- yfinance
- Flask
- MLflow
- Flask Monitoring Dashboard
- Notebook (jupyter)
- Docker (para deploy)

---

## 📁 Estrutura do Projeto

```plaintext
├── saved_models/ # Modelos e scaler salvos
├── api/
├──── app.py # API Flask para predição
├── notebooks/
├──── data/ # Dados baixados de ações
├──── stocks-model-training.ipynb # API Notebook para treino e deploy do modelo
├── requirements.txt
├── Dockerfile
├── start.sh
└── README.md
```

---

## 🚀 Como Rodar Localmente

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/lstm-stock-predictor.git
cd lstm-stock-predictor
```

### 2. Crie e ative o ambiente virtual (opcional)

```bash
python -m venv venv
source venv/bin/activate 
# Windows: venv\Scripts\activate
```
### 3. Instale as dependências

```bash
pip install -r requirements.txt
```
## 📊 Treinar o Modelo

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/stocks-model-training.ipynb
```
## 🔁 Fazer Predições com a API

### 1. Rodar a API localmente

```bash
python api/app.py

mlflow ui --host 0.0.0.0 --port 5001
```
Acesse a API em http://localhost:5000

Dashboard de monitoramento:

http://localhost:5000/dashboard

Mlops UI:

http://localhost:5001

### 2. Exemplo de requisição com curl:

```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"data": [100.0, 101.2, ..., 110.4]}'
```
## 🐳 Deploy com Docker

### 1. Build da imagem

```bash
docker build -t lstm-stock-api .
```
### 2. Run da API via Docker

```bash
docker run -it --rm -p 5000:5000 -p 5001:5001 lstm-stock-api
```

## 📈 Monitoramento e Rastreamento

A API expõe um painel interativo em /dashboard com estatísticas de tempo de resposta, chamadas e erros.

O MLflow é usado para rastrear todas as execuções de treinamento e inferência.