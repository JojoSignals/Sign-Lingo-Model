FROM python:3.10-slim

# Establecer directorio de trabajo
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Exponer el puerto por defecto de FastAPI
EXPOSE 8000

# Comando para iniciar la API
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
