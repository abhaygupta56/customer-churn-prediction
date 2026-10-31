from src.train import load_model
from src.predict import predict_churn

def main():
    model = load_model("models/churn_model.pkl")
    
    print("Churn prediction model loaded sucessfully")
    
    customer_data = { "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 75.5, "TotalCharges": 906.0 }
    
    
    result = predict_churn(model, customer_data, threshold=0.5)
    
    print(f"Prediction: {result['Prediction']}")
    print(f"Churn Probability: {result['Churn Probability']}%")
    print(f"Threshold: {result['Threshold']}")

if __name__ == "__main__":
    main()