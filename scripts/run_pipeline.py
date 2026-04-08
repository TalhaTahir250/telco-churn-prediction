import argparse
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.models.train import train_model
from src.models.tune import tune_model
from src.models.evaluate import evaluate_model


def run_pipeline(file_path: str, target_col: str = "Churn", tune: bool = False, model_path: str = "model.pkl"):
    # Step 1 — Load
    print("\n--- Step 1: Loading Data ---")
    df = load_data(file_path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Step 2 — Preprocess
    print("\n--- Step 2: Preprocessing ---")
    df = preprocess_data(df, target_col)

    # Step 3 — Feature Engineering
    print("\n--- Step 3: Building Features ---")
    df = build_features(df, target_col)

    # Step 4 — Optional Tuning
    if tune:
        print("\n--- Step 4: Tuning Hyperparameters ---")
        X = df.drop(columns=[target_col])
        y = df[target_col]
        best_params = tune_model(X, y)
        print(f"Best params found: {best_params}")

    # Step 5 — Train
    print("\n--- Step 5: Training ---")
    best_params = best_params if tune else None
    model, X_test, y_test, threshold = train_model(df, target_col, model_path=model_path, params=best_params)

    # Step 6 — Evaluate
    print("\n--- Step 6: Evaluating ---")
    metrics = evaluate_model(model, X_test, y_test, threshold=threshold)
    print(f"\nFinal Recall: {metrics['recall']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to raw CSV")
    parser.add_argument("--target", default="Churn")
    parser.add_argument("--tune", action="store_true", help="Run Optuna tuning")
    parser.add_argument("--model-path", default="model.pkl")
    args = parser.parse_args()

    run_pipeline(
        file_path=args.file,
        target_col=args.target,
        tune=args.tune,
        model_path=args.model_path
    )