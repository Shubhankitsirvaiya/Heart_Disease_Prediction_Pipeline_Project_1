FROM python:3.9

WORKDIR /app
COPY model_metrics_exporter.py .

RUN pip install prometheus_client mlflow pandas

CMD ["python", "model_metrics_exporter.py"]
