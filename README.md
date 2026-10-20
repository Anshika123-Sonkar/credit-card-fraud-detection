# Credit Card Fraud Detection — Research-Oriented ML Project

An end-to-end machine-learning project that compares models for highly imbalanced credit-card transaction data and exposes predictions through a Streamlit app.

> **Educational/research prototype only — not a bank decision system.** A prediction is not proof of fraud and must not be used for real financial decisions.

## Features

- Dataset schema and input checks
- Stratified train / validation / test split
- Logistic Regression with class weights
- Random Forest with class weights
- Logistic Regression with SMOTE in an imbalanced-learn pipeline
- Model selection by validation average precision (PR-AUC)
- Decision threshold selected on validation F1
- Final hold-out test metrics: precision, recall, F1, PR-AUC, ROC-AUC, and accuracy
- Confusion matrix, precision–recall curve, and feature-importance/coefficient plot
- Streamlit interface for single-transaction and batch CSV scoring
- Downloadable CSV output for batch predictions
- Unit test(s) for evaluation helpers

## Tech stack

Python · pandas · NumPy · scikit-learn · imbalanced-learn · Matplotlib · Seaborn · joblib · Streamlit · pytest

## Dataset

Download the public Kaggle **Credit Card Fraud Detection** dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Extract it and place the CSV here:

```text
data/creditcard.csv
```

The CSV should include the target column `Class` (1 = fraud, 0 = legitimate), along with the transaction features. **The dataset is not included in this repository.** Follow the dataset provider's terms and never commit the CSV or real customer data.

## Setup on Windows (PowerShell)

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this in the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Train models

```powershell
python train.py --data data/creditcard.csv
```

The script creates a trained model under `models/` and evaluation outputs under `reports/`. These files are generated locally and excluded from GitHub by `.gitignore`. Run training before starting the app.

For a development-only sample:

```powershell
python train.py --data data/creditcard.csv --max-rows 50000
```

Do not use sample-based metrics as final research claims.

## Run the web app

```powershell
streamlit run app.py
```

Open `http://localhost:8501`.

## Run tests

```powershell
python -m pytest -q
```

## Evaluation methodology

- Stratified split: 60% train, 20% validation, 20% final test.
- Imputation and scaling are fit inside pipelines.
- SMOTE is applied only within the training pipeline, not to validation/test data.
- Model selection uses validation average precision (PR-AUC); the threshold is selected on validation F1.
- Final reported metrics come from the held-out test set.
- Accuracy alone can be misleading when fraud is rare; inspect precision, recall, F1, PR-AUC, confusion matrix, threshold, and class prevalence.

## Example result from one local run

A run shown during development reported:
- Precision: **0.948**
- Recall: **0.768**
- F1-score: **0.849**
- PR-AUC: **0.857**
- Test rows: **56,746**
- Fraud prevalence: **0.167%**

Your results may differ with a different dataset file, dependency versions, or code changes. Re-run training and use the values generated in your own `reports/metrics.json`; do not treat these example values as a guarantee.

## Project structure

```text
Credit_Card_Fraud_Detection_Research_Project/
├── app.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/        # put creditcard.csv here locally; CSV is ignored by Git
├── models/      # trained model generated locally; ignored by Git
├── reports/     # metrics and plots generated locally; ignored by Git
├── src/
│   ├── data.py
│   └── evaluate.py
└── tests/
    └── test_evaluate.py
```

## Limitations and future work

- Dataset is anonymized and old; real-world distribution shift is not addressed.
- No real-time payment integration, cost-sensitive business policy, drift monitoring, or human review workflow is implemented.
- The single-transaction form uses manually entered feature values; meaningful testing should use representative rows with known labels in a controlled evaluation workflow.
- Do not deploy this as a real banking/fraud decision service without security, privacy, monitoring, validation, and domain review.

## Suggested demo screenshots

Add screenshots of the Streamlit interface, evaluation metrics, confusion matrix, and precision–recall curve to a `screenshots/` folder before publishing.

## License

No license is specified yet. Add a license file if you want to explicitly allow others to reuse the code.
