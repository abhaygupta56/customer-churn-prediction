import pandas as pd

def predict_churn(model,customer_data):
    customer_df = pd.DataFrame([customer_data])
    prediction = model.predict(customer_df)[0]
    probability = model.predict_proba(customer_df)[0]
    
    yes_index = list(model.classes_).index("Yes")
    churn_probability = probability[yes_index]
    
    return {
        "Prediction": prediction,
        "Churn Probability": round(churn_probability*100 , 2)
    }