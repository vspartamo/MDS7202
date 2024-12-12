# Dockerfile para Backend
FROM python:3.10-slim

WORKDIR /app

# Copiar archivos de requisitos
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código del backend
COPY src/ .
COPY models/ ./models/

# Exponer puerto
EXPOSE 7888

# Comando para ejecutar
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7888"]