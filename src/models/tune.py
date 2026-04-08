import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score


def tune_model(X, y):
    """
    Tunes an XGBoost model using Optuna.
    Optimizes for recall on churners with class imbalance handling.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 300, 800),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0, 5),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0, 5),
            "scale_pos_weight":  scale_pos_weight,
            "random_state":      42,
            "n_jobs":            -1,
            "eval_metric":       "logloss"
        }

        # Find best threshold per trial
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        best_recall = 0
        best_threshold = 0.5
        for threshold in [i / 100 for i in range(20, 60)]:
            y_pred = (proba >= threshold).astype(int)
            recall = recall_score(y_test, y_pred, pos_label=1)
            if recall > best_recall:
                best_recall = recall
                best_threshold = threshold

        trial.set_user_attr("threshold", best_threshold)
        return best_recall

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    best_params = study.best_params
    best_params["threshold"] = study.best_trial.user_attrs["threshold"]
    best_params["scale_pos_weight"] = scale_pos_weight

    print(f"Best Recall: {study.best_value:.4f}")
    print(f"Best Threshold: {best_params['threshold']}")
    print(f"Best Params: {best_params}")

    return best_params