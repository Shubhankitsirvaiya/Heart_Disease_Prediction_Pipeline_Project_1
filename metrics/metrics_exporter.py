from prometheus_client import start_http_server, Gauge
import boto3
import time

accuracy_gauge = Gauge('model_accuracy', 'Latest trained model accuracy')

def get_accuracy_from_minio():
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url="http://minio.kubeflow:9000",
            aws_access_key_id="minio",
            aws_secret_access_key="minio123"
        )
        s3.download_file("mlpipeline", "metrics/accuracy.txt", "accuracy.txt")
        with open("accuracy.txt", "r") as f:
            acc = float(f.read().strip())
            print("📈 Accuracy =", acc)
            return acc
    except Exception as e:
        print("⚠️ Failed to fetch accuracy:", e)
        return 0.0

if __name__ == "__main__":
    start_http_server(8000)
    print("🚀 Exporter started on port 8000")
    while True:
        accuracy = get_accuracy_from_minio()
        accuracy_gauge.set(accuracy)
        time.sleep(30)
