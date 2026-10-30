from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    yes_index = list(model.classes_).index("Yes")
    y_prob = model.predict_proba(X_test)[:, yes_index]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label="Yes"),
        "Recall": recall_score(y_test, y_pred, pos_label="Yes"),
        "F1 Score": f1_score(y_test, y_pred, pos_label="Yes"),
        "ROC AUC": roc_auc_score(
            (y_test == "Yes").astype(int), y_prob
        )
    }

    print("Classification Report:\n")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    return metrics