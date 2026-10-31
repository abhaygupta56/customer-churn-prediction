# Customer Churn Prediction

An end-to-end Machine Learning project that predicts whether a telecom customer is likely to leave the company (churn) using customer demographics, account information, and subscribed services.

## Project Overview

Customer churn is a major challenge for telecom companies. Predicting customers who are likely to leave helps businesses take preventive actions and improve customer retention.

This project uses machine learning classification algorithms to predict customer churn.

## Dataset

* **Dataset:** IBM Telco Customer Churn
* **Source:** IBM Sample Data
* **Target Variable:** `Churn`
* **Classes:** Yes / No

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook
* Joblib
* Git & GitHub

## Project Workflow

1. Data Loading
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering and Preprocessing
5. Train, Validation and Test Split
6. Model Training
7. Hyperparameter Tuning
8. Model Comparison
9. Threshold Optimization
10. Final Model Evaluation
11. Model Saving
12. Customer Churn Prediction

## Machine Learning Algorithms

The following classification algorithms were implemented:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier

Hyperparameter tuning was performed using `GridSearchCV` with 5-fold cross-validation and F1-score optimization.

## Evaluation Metrics

Models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

The final model was selected using validation performance and evaluated on the separate test dataset.

## 📊 Model Performance

The final selected model is **Tuned Logistic Regression**.

Evaluated on the held-out test dataset using the optimized classification threshold of **0.55**.

### Evaluation Metrics

| Metric    |  Score |
| --------- | -----: |
| Accuracy  | 74.41% |
| Precision | 51.27% |
| Recall    | 75.67% |
| F1-Score  | 61.12% |
| ROC-AUC   | 83.44% |

### Confusion Matrix

| Actual / Predicted | No Churn | Churn |
| ------------------ | -------: | ----: |
| No Churn           |      764 |   269 |
| Churn              |       91 |   283 |

### Key Observations

* The model correctly identifies 283 customers who churned.
* It detects approximately 75.67% of actual churn cases.
* ROC-AUC of 83.44% indicates good ability to distinguish between churn and non-churn customers.
* The model prioritizes identifying customers who may leave, which is important for customer retention.

## Project Structure

```text
churn-prediction-model/
│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── models/
│   └── churn_model.pkl
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── split_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

Navigate to the project directory:

```bash
cd churn-prediction-model
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Project

Execute:

```bash
python main.py
```

The application loads the saved machine learning model and predicts churn for the sample customer.

## 🚀 Live Demo

Try the deployed Customer Churn Prediction application:

[🔗 Launch Customer Churn Prediction App](https://customer-churn-prediction-gy2aujks7kbpzgpuwjvjgo.streamlit.app/)

### Features
- Predict customer churn using machine learning.
- Display churn probability.
- Interactive Streamlit interface.
- Optimized prediction threshold: 0.55.
## Sample Prediction

```text
Churn Prediction Model Loaded Successfully

Prediction: Yes
Churn Probability: 88.42%
Threshold: 0.5
```

*Sample output from a local model run.*

## Future Improvements

* Develop an interactive Streamlit dashboard.
* Add customer-level churn explanations.
* Improve model performance through additional feature engineering.
* Deploy the model as a web application.

## Author

**Abhay Gupta**

Machine Learning | Artificial Intelligence | Python
