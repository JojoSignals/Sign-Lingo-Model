FROM python:3.10-slim

# Instalar dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*


# Establecer directorio de trabajo
WORKDIR /app

# Copiar dependencias
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copiar todo el proyecto
COPY . .

# Exponer el puerto por defecto de FastAPI
EXPOSE 8000

# Comando para iniciar la API
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]
