import pandas as pd

    
def predict_churn(model,customer_data,threshold=0.5):
    customer_df = pd.DataFrame([customer_data])
    probability = model.predict_proba(customer_df)[0]
    
    yes_index = list(model.classes_).index("Yes")
    churn_probability = probability[yes_index]
    
    prediction = "Yes" if churn_probability >= threshold else "No"
    
    return {
        "Prediction": prediction,
        "Churn Probability": round(float(churn_probability) * 100, 2),
        "Threshold": threshold
    }