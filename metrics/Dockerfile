FROM python:3.9
WORKDIR /app
COPY metrics_exporter.py .
RUN pip install prometheus_client boto3
CMD ["python", "metrics_exporter.py"]
