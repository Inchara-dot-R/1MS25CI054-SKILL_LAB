import os
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "hf-sentiment-champion-challenger"
REGISTERED_NAME = "sentiment-model"

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    client = MlflowClient()
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        raise RuntimeError("Experiment not found. Run python evaluate.py first.")

    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time DESC"])
    model_runs = {}
    summary = None
    for run in runs:
        name = run.data.tags.get("mlflow.runName", "")
        if name.startswith("A_sentiment_") and "A" not in model_runs:
            model_runs["A"] = run
        elif name.startswith("B_sentiment_") and "B" not in model_runs:
            model_runs["B"] = run
        elif name == "compare_sentiment" and summary is None:
            summary = run

    if "A" not in model_runs or "B" not in model_runs or summary is None:
        raise RuntimeError("Could not find both model runs and comparison run. Run evaluate.py.")

    winner = summary.data.tags.get("winner", "A")
    for role in ("A", "B"):
        run = model_runs[role]
        model_uri = f"runs:/{run.info.run_id}/model"
        registered = mlflow.register_model(model_uri, REGISTERED_NAME)
        client.set_registered_model_alias(
            REGISTERED_NAME, "champion" if role == winner else "challenger",
            registered.version
        )
        print(f"Registered Model {role} as version {registered.version}")

    print(f"\nChampion alias points to Model {winner}.")
    print(f"Registry model name: {REGISTERED_NAME}")
    print("Open MLflow UI -> Models -> sentiment-model to inspect aliases.")

if __name__ == "__main__":
    main()
