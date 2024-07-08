# Usa una imagen base de Python
FROM python:3.11

# Establece el directorio de trabajo en /app
WORKDIR /app

# Copia los archivos del proyecto al contenedor
COPY . .

# Instala las dependencias desde requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expone el puerto 8000 en el contenedor
EXPOSE 8000

# Comando para ejecutar la aplicación FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
