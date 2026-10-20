# Telco Customer Churn Prediction

A machine-learning project for exploring the Telco Customer Churn dataset, training a churn classifier, and serving individual predictions with Streamlit. The app reports an estimated probability; it is a decision-support prototype, not a guarantee or a substitute for human review.

## Project Contents

- `app.py`: Streamlit interface for entering customer details and predicting churn.
- `artifacts/churn_model1.pkl`: fitted preprocessing and prediction pipeline loaded by the app.
- `EDA.ipynb`: exploratory analysis, model training, evaluation, and artifact creation.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: dataset used in the notebook.
- `requirements.txt`: Python dependencies for the app and serialized model.

## Run Locally

Python 3.14.5 was used to create the saved model. Use that version with the pinned dependencies to reduce serialization compatibility issues.

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

The app expects the model at `artifacts/churn_model1.pkl`. If you retrain the model, save the fitted pipeline to that path. Do not load model files from untrusted sources: Python pickle-based formats can execute code when loaded.

## Reproduce the Analysis

Open `EDA.ipynb` from the project root and run its cells in order. The notebook reads the CSV from the project directory and saves the fitted pipeline under `artifacts/`. Some notebook cells perform model tuning and MLflow tracking, which can take longer than running the prediction app.

## Dataset

The included CSV is the IBM Telco Customer Churn sample dataset. It contains customer account, service, and billing attributes along with the historical churn label. The label is used during training and is not an input to the app.

## GitHub Upload Notes

Commit `app.py`, `README.md`, `requirements.txt`, `EDA.ipynb`, the CSV, and `artifacts/churn_model1.pkl`. The `.gitignore` excludes local environments and generated MLflow tracking files. The model and dataset are required for the current app to run as checked out; the model is below GitHub's 100 MB per-file limit.

Check the dataset's source terms and choose a repository license before publishing if you plan to redistribute or reuse the data or code.