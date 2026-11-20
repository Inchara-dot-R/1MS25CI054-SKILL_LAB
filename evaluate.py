import os
import json
import time
import platform
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from transformers import pipeline

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
DATA.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT = "hf-sentiment-champion-challenger"
MODEL_A = "distilbert-base-uncased-finetuned-sst-2-english"
MODEL_B = "siebert/sentiment-roberta-large-english"
TASK = "sentiment"
EVAL_SAMPLES = 200
SEED = 42
DEVICE = 0 if torch.cuda.is_available() else -1
DEVICE_NAME = "cuda" if DEVICE == 0 else "cpu"
STUDENT_USN = os.getenv("STUDENT_USN", "REPLACE_WITH_YOUR_USN")

# Champion rule is fixed BEFORE seeing the experiment results:
# Challenger must improve macro-F1 by at least 0.01 (one percentage point).
# If it does not, Model A stays Champion. If scores are within the margin,
# prefer the faster model. This rule is not changed after results are seen.
PRIMARY_METRIC = "macro_f1"
MARGIN = 0.01

class HFTextClassifier(mlflow.pyfunc.PythonModel):
    """MLflow wrapper. Hugging Face is loaded inside this model wrapper."""
    def load_context(self, context):
        with open(context.artifacts["config"], "r", encoding="utf-8") as f:
            cfg = json.load(f)
        self.model_id = cfg["model_id"]
        self.classifier = pipeline(
            "sentiment-analysis",
            model=self.model_id,
            device=DEVICE
        )

    def predict(self, context, model_input, params=None):
        texts = model_input["text"].astype(str).tolist()
        raw = self.classifier(texts, truncation=True)
        rows = []
        for text, result in zip(texts, raw):
            label = str(result["label"]).upper()
            if label in ("LABEL_0", "NEGATIVE"):
                label = "NEGATIVE"
            elif label in ("LABEL_1", "POSITIVE"):
                label = "POSITIVE"
            rows.append({"text": text, "prediction": label, "score": float(result["score"])})
        return pd.DataFrame(rows)

def normalize_label(label):
    return "POSITIVE" if int(label) == 1 else "NEGATIVE"

def run_one(model_id, role, model_name, eval_df, y_true):
    print(f"\nLoading {role}: {model_id}")
    clf = pipeline("sentiment-analysis", model=model_id, device=DEVICE)
    texts = eval_df["text"].tolist()
    start = time.perf_counter()
    raw = clf(texts, batch_size=8, truncation=True)
    total_s = time.perf_counter() - start
    preds = [str(x["label"]).upper() for x in raw]
    preds = ["POSITIVE" if p in ("POSITIVE", "LABEL_1") else "NEGATIVE" for p in preds]
    y_pred = [1 if p == "POSITIVE" else 0 for p in preds]
    avg_ms = total_s * 1000 / len(texts)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "avg_latency_ms": float(avg_ms),
        "total_inference_s": float(total_s)
    }
    pred_df = eval_df.copy()
    pred_df["reference"] = [normalize_label(x) for x in y_true]
    pred_df["prediction"] = preds
    pred_df["correct"] = pred_df["reference"] == pred_df["prediction"]
    pred_path = OUT / f"predictions_{model_name}.csv"
    pred_df.to_csv(pred_path, index=False)

    config_path = OUT / f"{model_name}_config.json"
    config_path.write_text(json.dumps({"model_id": model_id}, indent=2), encoding="utf-8")

    with mlflow.start_run(run_name=f"{role}_{TASK}_{model_name}") as run:
        mlflow.log_params({
            "hf_model_id": model_id, "task": TASK,
            "role": "champion candidate" if role == "A" else "challenger candidate",
            "max_length": 512, "batch_size": 8, "eval_samples": len(eval_df),
            "seed": SEED, "device": DEVICE_NAME
        })
        mlflow.set_tags({
            "student_usn": STUDENT_USN,
            "library_versions": f"transformers={__import__('transformers').__version__};torch={torch.__version__};mlflow={mlflow.__version__}",
            "dataset_name": "GLUE SST-2 validation subset"
        })
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        mlflow.log_artifact(str(pred_path))
        mlflow.log_artifact(str(DATA / "eval_inputs.csv"))
        metrics_path = OUT / f"metrics_{model_name}.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(metrics_path))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["NEGATIVE", "POSITIVE"])
        disp.plot()
        plt.title(f"Confusion Matrix - {model_name}")
        cm_path = OUT / f"confusion_matrix_{model_name}.png"
        plt.savefig(cm_path, bbox_inches="tight")
        plt.close()
        mlflow.log_artifact(str(cm_path))

        # The wrapper reloads the same Hugging Face model when served via MLflow.
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=HFTextClassifier(),
            artifacts={"config": str(config_path)},
            pip_requirements=[
                f"mlflow=={mlflow.__version__}",
                "transformers==4.51.3",
                "torch",
                "pandas"
            ]
        )
        return {"run_id": run.info.run_id, "metrics": metrics, "predictions": pred_df}

def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    np.random.seed(SEED)

    print("Downloading/loading GLUE SST-2 validation data...")
    ds = load_dataset("glue", "sst2", split="validation")
    # Deterministic, identical 200-row evaluation set for both models.
    ds = ds.shuffle(seed=SEED).select(range(EVAL_SAMPLES))
    eval_df = pd.DataFrame({"text": ds["sentence"], "label": ds["label"]})
    eval_df.to_csv(DATA / "eval_inputs.csv", index=False)
    y_true = eval_df["label"].astype(int).tolist()

    # Rule is also saved in README before this script is run.
    result_a = run_one(MODEL_A, "A", "distilbert_sst2", eval_df, y_true)
    result_b = run_one(MODEL_B, "B", "roberta_large", eval_df, y_true)

    delta = result_b["metrics"]["macro_f1"] - result_a["metrics"]["macro_f1"]
    if delta >= MARGIN:
        winner = "B"
    else:
        winner = "A"

    # Comparison/parent summary run.
    with mlflow.start_run(run_name=f"compare_{TASK}") as summary:
        mlflow.log_params({"primary_metric": PRIMARY_METRIC, "margin": MARGIN,
                           "model_a_run_id": result_a["run_id"], "model_b_run_id": result_b["run_id"]})
        mlflow.log_metric("delta_primary_metric", delta)
        mlflow.set_tag("winner", winner)
        mlflow.set_tag("decision", "Challenger replaces Champion" if winner == "B" else "Champion remains")
        summary_df = pd.DataFrame([
            {"model": "A - DistilBERT", **result_a["metrics"]},
            {"model": "B - RoBERTa", **result_b["metrics"]}
        ])
        summary_csv = OUT / "comparison_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        mlflow.log_artifact(str(summary_csv))
        # Keep project documentation and dependency list with the comparison run.
        mlflow.log_artifact(str(ROOT / "README.md"))
        mlflow.log_artifact(str(ROOT / "requirements.txt"))

        plt.figure(figsize=(8, 5))
        summary_df.set_index("model")[["accuracy", "macro_f1"]].plot(kind="bar")
        plt.ylabel("Score")
        plt.title("Champion vs Challenger metrics")
        plt.tight_layout()
        chart_path = OUT / "model_comparison.png"
        plt.savefig(chart_path)
        plt.close()
        mlflow.log_artifact(str(chart_path))

        disagreements = result_a["predictions"][["text", "reference", "prediction"]].rename(
            columns={"prediction": "model_a_prediction"})
        disagreements["model_b_prediction"] = result_b["predictions"]["prediction"].values
        disagreements = disagreements[disagreements["model_a_prediction"] != disagreements["model_b_prediction"]].head(5)
        disagreements["comment"] = "Models disagree; compare both predictions with the reference label."
        disagree_path = OUT / "five_disagreements.csv"
        disagreements.to_csv(disagree_path, index=False)
        mlflow.log_artifact(str(disagree_path))

    print("\n===== COMPARISON RESULTS =====")
    print(summary_df.to_string(index=False))
    print(f"\nMacro-F1 difference (B - A): {delta:.4f}")
    print(f"Winner under the pre-written rule: Model {winner}")
    print("\nNext run: python register_champion.py")

if __name__ == "__main__":
    main()
