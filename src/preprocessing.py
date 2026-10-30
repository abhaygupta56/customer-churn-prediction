import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def split_features_target(df):
    X = df.drop(columns=["Churn","customerID"])
    y = df["Churn"]
    
    return X,y

def clean_data(df):
    df = df.copy()
    
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],errors = "coerce")
    
    df = df.dropna(subset=["TotalCharges"])
    
    return df

def get_feature_types(X):
    numerical_features = X.select_dtypes(include=["int64","float64"]).columns.tolist()
    
    categorical_features = X.select_dtypes(include = ["object","category","bool"]).columns.tolist()
    
    return numerical_features,categorical_features

def build_preprocessor(numerical_features, categorical_features):

    numerical_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(
        handle_unknown="ignore"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor