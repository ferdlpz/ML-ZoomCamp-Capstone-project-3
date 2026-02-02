# Usamos una imagen base ligera de Python
FROM python:3.11-slim

# Instalamos uv directamente desde el binario oficial
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Establecemos el directorio de trabajo
WORKDIR /app

# Copiamos los archivos de dependencias primero (para aprovechar el cache de capas)
COPY pyproject.toml uv.lock ./

# Instalamos las dependencias en el python del sistema (sin venv para el contenedor)
RUN uv sync --frozen --no-cache

# Copiamos el código de la aplicación y el modelo
COPY predict.py .
# Asegúrate de que el nombre del archivo coincida exactamente con tu .bin
COPY model_n_estimators=500_max_depth=6_learning_rate=0.01.bin .

# Exponemos el puerto que usa tu Flask
EXPOSE 9696

# Ejecutamos con Gunicorn para producción en lugar del servidor de Flask
# 4 workers es un estándar inicial saludable
CMD ["uv", "run", "gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]