# Dockerfile para Frontend Gradio
FROM python:3.10-slim

WORKDIR /app

# Copiar archivos de requisitos
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código del frontend
COPY src/ .

# Exponer puerto
EXPOSE 7860

# Comando para ejecutar
CMD ["python", "app.py"]