FROM python:3.11-slim

WORKDIR /app

# Copiar requirements.txt e instalar dependencias
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el backend al docker
COPY . .

# puerto 
EXPOSE 8000

# Comando para correr FastAPI dentro del contenedor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
