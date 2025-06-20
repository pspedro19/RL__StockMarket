# Dockerfile para Sistema de Trading RL
FROM python:3.11-slim

# Establecer directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements y instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código fuente
COPY src/ ./src/
COPY configs/ ./configs/
COPY main.py .

# Crear directorios necesarios
RUN mkdir -p data/models data/results data/raw data/processed logs

# Configurar variables de entorno
ENV PYTHONPATH=/app
ENV TRADING_ENV=production

# Exponer puerto para monitoreo
EXPOSE 8080

# Comando por defecto
CMD ["python", "main.py"]

# Etiquetas
LABEL maintainer="Trading System"
LABEL version="2.0.0"
LABEL description="Sistema de Trading con Reinforcement Learning" 