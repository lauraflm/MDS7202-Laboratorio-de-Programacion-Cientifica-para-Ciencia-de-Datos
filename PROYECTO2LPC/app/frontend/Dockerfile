FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# puerto (distinto al backend)
EXPOSE 7860

CMD ["python", "app.py"]
