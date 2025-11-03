FROM python:3.10-slim

ENV AIRFLOW_HOME=/opt/airflow
WORKDIR /opt/airflow

# Requisito del enunciado: curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Dependencias (Airflow + ML + Gradio)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Inicializar Airflow + usuario admin
RUN airflow db init && \
    airflow users create \
      --username admin \
      --firstname Admin \
      --lastname User \
      --role Admin \
      --email admin@example.com \
      --password admin

# Copiar DAGs solicitados
COPY dags/ /opt/airflow/dags/

# Puertos: 8080 Airflow, 7860 Gradio
EXPOSE 8080 7860

# Levantar Webserver + Scheduler
CMD ["bash","-lc","airflow webserver -p 8080 & airflow scheduler & tail -f /dev/null"]
