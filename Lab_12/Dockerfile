# Imagen de python
FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# Copiar el archivo requirements.txt al contenedor
COPY requirements.txt .

# Instalar las dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el contenido de la carpeta al contenedor
COPY . .

# Exponer el puerto 7888
EXPOSE 7888

# Comando para ejecutar la aplicación
CMD ["python", "main.py"]