from src.preprocessing import (load_data,clean_data,split_features_target,get_feature_types,build_preprocessor)

from src.split_data import split_data
from src.train import build_model,train_model,save_model
from src.evaluate import evaluate_model

def main():
    df = load_data("data/Telco-Customer-Churn.csv")
    df = clean_data(df)
    
    X,y = split_features_target(df)
    
    numerical_features, categorical_features = get_feature_types(X)
    preprocessor = build_preprocessor(numerical_features,categorical_features)
    
    X_train,X_test,y_train,y_test = split_data(X,y)
    
    model = build_model(preprocessor)
    model = train_model(model,X_train,y_train)
    
    save_model(model)
    
    results = evaluate_model(model,X_test,y_test)
    
    print("Model performance : ")
    for metric , score in results.items():
        print(f"{metric}: {score:.4f}")
        
        
if __name__ == "__main__":
    main()