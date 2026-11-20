import os
from pathlib import Path
import pandas as pd
import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parent
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = "sentiment-model"
ALIAS = "champion"

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    client = MlflowClient()
    alias_version = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    model_uri = f"models:/{MODEL_NAME}@{ALIAS}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Read model identity from registry version metadata/tags.
    run = client.get_run(alias_version.run_id)
    model_id = run.data.params.get("hf_model_id", "not recorded")
    print(f"Loaded registered model: {MODEL_NAME}")
    print(f"Alias: {ALIAS}")
    print(f"Registry version: {alias_version.version}")
    print(f"Hugging Face model ID: {model_id}")

    new_data = pd.read_csv(ROOT / "data" / "new_data.csv")
    predictions = model.predict(new_data[["text"]])
    result = pd.concat([new_data.reset_index(drop=True), predictions[["prediction", "score"]]], axis=1)
    out = ROOT / "outputs" / "new_data_predictions.csv"
    out.parent.mkdir(exist_ok=True)
    result.to_csv(out, index=False)
    print("\nPredictions on NEW examples:")
    print(result.to_string(index=False))
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
