# Dockerfile para FastAPI Serving - LSTM Autoencoder RS
FROM python:3.10-slim

# Evitar gravação de arquivos pyc no container
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependências básicas do sistema operacional
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements.txt e instalar dependências
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar código-fonte da aplicação
COPY src/ /app/src/
COPY config/ /app/config/
COPY models/ /app/models/
COPY data/ /app/data/

# Porta de serviço exposta
EXPOSE 8000

# Executar FastAPI em modo de produção
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]
