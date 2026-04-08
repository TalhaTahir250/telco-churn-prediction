from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

def evaluate_model(model, X_test, y_test, threshold: float = 0.5):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    metrics = {
        "accuracy":  accuracy_score(y_test, preds),
        "recall":    recall_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "f1":        f1_score(y_test, preds),
    }

    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print(f"Recall: {metrics['recall']:.4f}  |  Precision: {metrics['precision']:.4f}  |  F1: {metrics['f1']:.4f}")

    return metrics