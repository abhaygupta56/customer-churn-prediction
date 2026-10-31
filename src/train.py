import os
import joblib

def train_model(model,X_train,y_train):
    model.fit(X_train,y_train)
    return model

def save_model(model,file_path="models/churn_model.pkl"):
    os.makedirs(os.path.dirname(file_path),exist_ok=True)
    joblib.dump(model,file_path)
    print(f"Model saved successfully at {file_path}")


def load_model(file_path="models/churn_model.pkl"):
    model = joblib.load(file_path)
    return model