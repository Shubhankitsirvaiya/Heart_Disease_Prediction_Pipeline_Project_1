from prometheus_client import start_http_server, Gauge
import mlflow
import time

# Define a Prometheus metric
accuracy_metric = Gauge('model_accuracy', 'Accuracy of the latest trained model')

def fetch_latest_accuracy_from_mlflow():
    try:
        # Connect to MLflow tracking server (change if remote)
        mlflow.set_tracking_uri("http://mlflow.kubeflow.svc.cluster.local:5000")

        # Use your experiment name
        experiment = mlflow.get_experiment_by_name("Default")
        experiment_id = experiment.experiment_id

        # Fetch latest run
        runs = mlflow.search_runs(
            experiment_ids=[experiment_id],
            order_by=["attributes.start_time desc"],
            max_results=1
        )

        if not runs.empty and "metrics.accuracy" in runs.columns:
            acc = float(runs.iloc[0]["metrics.accuracy"])
            print(f"✅ Latest Accuracy: {acc}")
            return acc
        else:
            print("⚠️ No runs or accuracy metric found.")
            return 0.0

    except Exception as e:
        print(f"❌ Error fetching accuracy from MLflow: {e}")
        return 0.0

if __name__ == "__main__":
    print("🚀 Starting metrics exporter on port 8000")
    start_http_server(8000)

    while True:
        acc = fetch_latest_accuracy_from_mlflow()
        accuracy_metric.set(acc)
        time.sleep(30)
