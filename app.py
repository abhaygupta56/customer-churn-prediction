import streamlit as st

from src.predict import predict_churn
from src.train import load_model

MODEL_PATH = "models/churn_model.pkl"
THRESHOLD = 0.5  # Replace with your notebook's best_threshold

YES_NO = ["Yes", "No"]
YES_NO_NOINTERNET = ["Yes", "No", "No internet service"]

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🔷",
    layout="centered",
)

st.title("Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to leave.")

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

try:
    model = get_model()
except FileNotFoundError:
    st.error(f"Model file not found at `{MODEL_PATH}`. Train the model first.")
    st.stop()

st.subheader("Customer Info")

with st.form("customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])

        senior = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
        )

        partner = st.selectbox("Partner", YES_NO)
        dependents = st.selectbox("Dependents", YES_NO)

        tenure = st.number_input(
            "Tenure (months)", 0, 72, 12
        )

        phone = st.selectbox("Phone Service", YES_NO)

        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["Yes", "No", "No phone service"],
        )

        internet = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"],
        )

        online_security = st.selectbox(
            "Online Security", YES_NO_NOINTERNET
        )

        online_backup = st.selectbox(
            "Online Backup", YES_NO_NOINTERNET
        )

    with col2:
        device_protection = st.selectbox(
            "Device Protection", YES_NO_NOINTERNET
        )

        tech_support = st.selectbox(
            "Tech Support", YES_NO_NOINTERNET
        )

        streaming_tv = st.selectbox(
            "Streaming TV", YES_NO_NOINTERNET
        )

        streaming_movies = st.selectbox(
            "Streaming Movies", YES_NO_NOINTERNET
        )

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"],
        )

        paperless = st.selectbox(
            "Paperless Billing", YES_NO
        )

        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=75.5,
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=906.0,
        )

    submitted = st.form_submit_button("Predict Churn")

if submitted:
    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    try:
        with st.spinner("Predicting..."):
            result = predict_churn(
                model,
                customer_data,
                threshold=THRESHOLD,
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.subheader("Prediction Result")

    if result["Prediction"] == "Yes":
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")

    probability = result["Churn Probability"]

    st.metric(
        "Churn Probability",
        f"{probability}%",
    )

    st.progress(
        min(probability / 100, 1.0)
    )
