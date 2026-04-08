# src/models/train.py
import mlflow
import joblib
import pandas as pd
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score


def train_model(
    df:         pd.DataFrame,
    target_col: str,
    model_path: str  = "model.pkl",
    params:     dict = None
):
    # ── STEP 1: Split features and target ─────────────────────────────────────
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ── STEP 2: Extract threshold from params ──────────────────────────────────
    # Threshold is not an XGBoost parameter — pull it out before passing to model
    threshold = 0.5
    if params and "threshold" in params:
        threshold = params.pop("threshold")

    # ── STEP 3: Build model parameters ────────────────────────────────────────
    # scale_pos_weight handles class imbalance — ratio of negatives to positives
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    if params:
        model_params = params.copy()
        model_params.setdefault("scale_pos_weight", scale_pos_weight)
        model_params.setdefault("random_state", 42)
        model_params.setdefault("n_jobs", -1)
        model_params.setdefault("eval_metric", "logloss")
    else:
        model_params = {
            "n_estimators":     300,
            "learning_rate":    0.1,
            "max_depth":        6,
            "scale_pos_weight": scale_pos_weight,
            "random_state":     42,
            "n_jobs":           -1,
            "eval_metric":      "logloss",
        }

    # ── STEP 4: Train with MLflow tracking ────────────────────────────────────
    with mlflow.start_run():
        model = XGBClassifier(**model_params)
        model.fit(X_train, y_train)

        # Use custom threshold for predictions — not default 0.5
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= threshold).astype(int)

        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds)

        # Log everything to MLflow
        mlflow.log_params(model_params)
        mlflow.log_param("threshold", threshold)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.xgboost.log_model(model, "model")

        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}")

    # ── STEP 5: Save artifact ──────────────────────────────────────────────────
    # Save model + threshold + feature columns together
    # feature_cols is critical for inference — guarantees same column order
    artifact = {
        "model":        model,
        "threshold":    threshold,
        "feature_cols": list(X_train.columns),
    }

    joblib.dump(artifact, model_path)
    print(f"Model and threshold saved to {model_path}")
    print(f"Feature columns saved: {len(X_train.columns)} features")

    return model, X_test, y_test, threshold