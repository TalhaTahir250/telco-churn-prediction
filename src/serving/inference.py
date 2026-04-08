# src/serving/inference.py
import joblib
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.data.preprocess import preprocess_data


def predict(customer_data: dict, model_path: str = "model.pkl") -> dict:
    """
    Takes a single customer as a dictionary and returns churn prediction.
    """
    # Step 1 — Load artifact
    artifact  = joblib.load(model_path)
    model     = artifact["model"]
    threshold = artifact["threshold"]
    feature_cols = artifact["feature_cols"]

    # Step 2 — Convert to DataFrame
    df = pd.DataFrame([customer_data])

    # Step 3 — Preprocess (strip, drop ID, fix types)
    df = preprocess_data(df, target_col="Churn")

    # Step 4 — Apply one-hot encoding
    # Use pd.get_dummies then reindex to match training columns exactly
    df = pd.get_dummies(df)

    # Step 5 — Reindex to match exact training feature columns
    # Missing columns become 0, extra columns are dropped
    df = df.reindex(columns=feature_cols, fill_value=0)

    # Step 6 — Remove target if it survived
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    # Step 7 — Predict
    probability = model.predict_proba(df)[0][1]
    prediction  = int(probability >= threshold)

    return {
        "churn":             bool(prediction),
        "churn_probability": round(float(probability), 4),
        "threshold_used":    float(threshold),
    }