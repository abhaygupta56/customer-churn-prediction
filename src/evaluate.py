import numpy as np

from sklearn.metrics import (
accuracy_score,
precision_score,
recall_score,
f1_score,
roc_auc_score,
classification_report,
confusion_matrix
)

def evaluate_model(model, X_test, y_test, threshold=0.5):


    yes_index = list(model.classes_).index("Yes")
    y_prob = model.predict_proba(X_test)[:, yes_index]

    y_pred = np.where(y_prob >= threshold, "Yes", "No")

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label="Yes"),
        "Recall": recall_score(y_test, y_pred, pos_label="Yes"),
        "F1 Score": f1_score(y_test, y_pred, pos_label="Yes"),
        "ROC AUC": roc_auc_score(
            (y_test == "Yes").astype(int),
            y_prob
        )
    }

    print(f"Evaluation Threshold: {threshold}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nMetrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    return metrics

