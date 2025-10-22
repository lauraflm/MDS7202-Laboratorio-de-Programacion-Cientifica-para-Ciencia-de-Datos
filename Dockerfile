# Imagen base de Python (ligera)
FROM python:3.11-slim AS runtime

#Variables para evitar archivos temporales y logs bufferizados
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

#Directorio de trabajo dentro del contenedor
WORKDIR /app

#Instalar paquetes básicos del sistema (y limpiar)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

#Instalar dependencias desde requirements.txt
COPY requirements.txt /app/
RUN pip install -r requirements.txt

#Copiar tu aplicación
COPY main.py /app/

#(Opcional) copiar modelos si los tienes
COPY models/ /app/models/

#Crear usuario no-root
RUN useradd -m appuser
USER appuser

#Exponer puerto (para acceder sin entrar al contenedor)
EXPOSE 8000

#Configurar volúmenes (persistencia de datos y modelo)
VOLUME ["/app/data", "/app/models"]

#Healthcheck para verificar que la API está activa
HEALTHCHECK --interval=30s --timeout=3s CMD wget -qO- http://127.0.0.1:8000/ || exit 1

#Comando por defecto: iniciar servidor Uvicorn
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
