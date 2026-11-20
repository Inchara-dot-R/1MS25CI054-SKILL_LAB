# Champion–Challenger Model Comparison with Hugging Face and MLflow

## 1. Task and models
Task: Binary sentiment classification (positive/negative). Dataset: GLUE SST-2 validation split, deterministic 200-example subset (seed 42).
- Model A (initial Champion): `distilbert-base-uncased-finetuned-sst-2-english`
- Model B (Challenger): `siebert/sentiment-roberta-large-english`

## 2. How to run
Run these commands from this project folder. Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000
```
Keep the MLflow server terminal open. Open a second terminal in this folder and run:
```bash
python evaluate.py
python register_champion.py
python predict_from_mlflow.py
```
Open http://127.0.0.1:5000 to view MLflow.

## 3. Champion rule (written before final run)
Primary metric: macro-F1 (higher is better). The Challenger must exceed the Champion by at least 0.01 (one percentage point) to replace it. Otherwise Model A remains Champion. If scores are within the margin, prefer the faster model. The margin avoids promoting a model for a tiny difference that may be noise in a limited evaluation sample. Accuracy and macro-F1 are higher-is-better; latency and inference time are lower-is-better.

## 4. Results
Run `python evaluate.py`. The actual scores and latency are generated from the run and saved in `outputs/comparison_summary.csv`. Do not invent or manually change scores.

## 5. Decision
The script applies the pre-declared rule and records the winner in the MLflow comparison run. After running, use the actual values in `outputs/comparison_summary.csv` to write a 150–250 word discussion: report the macro-F1 difference, whether it exceeds 0.01, whether 200 examples are enough to support a strong conclusion, and the speed/model-size trade-off. A Challenger that loses is a valid result.

## 6. Limitations and declaration
The experiment uses a 200-example validation subset, so results may vary on other data. The larger RoBERTa model may take longer and require more memory. First-time execution downloads pretrained models and dataset files. AI assistance was used to help prepare and explain the project code; the student must understand and verify every part before submission.

## Repository contents
- `evaluate.py`: evaluates both models and logs MLflow runs, metrics and artifacts.
- `register_champion.py`: registers both runs and assigns champion/challenger aliases.
- `predict_from_mlflow.py`: loads the champion by MLflow alias and predicts on unseen examples.
- `data/`: evaluation and new examples (evaluation CSV is created by evaluate.py).
- `outputs/`: generated predictions, metrics and plots.
- `screenshots/`: add the three required screenshots.
