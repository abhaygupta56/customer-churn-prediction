from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib

def build_model(preprocessor):
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000))
        ]
    )

    return model

def train_model(model,X_train,y_train):
    model.fit(X_train,y_train)
    return model


def save_model(model,file_path="models/churn_model.pkl"):
    joblib.dump(model,file_path)
    print(f"Model saved successfully at {file_path}")


def load_model(file_path="models/churn_model.pkl"):
    model = joblib.load(file_path)
    return model