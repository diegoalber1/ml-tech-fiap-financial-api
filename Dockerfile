FROM python:3.10

WORKDIR /app

# Instala as dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da API e modelos salvos
COPY api/ ./api/
COPY notebooks/ ./notebooks/
COPY start.sh .

# Dá permissão de execução ao script de inicialização
RUN chmod +x start.sh

# Expõe as portas do Flask e do MLflow
EXPOSE 5000 5001

# Comando de inicialização
CMD ["./start.sh"]