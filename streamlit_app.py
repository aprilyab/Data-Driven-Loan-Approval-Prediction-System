import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ------------------------------
# Load the necessary files
# ------------------------------
@st.cache_resource
def load_files(model_path, data_path, scaler_path):
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return df, model, scaler

df, model, scaler = load_files(
    "outputs/models/Decision_Tree_Loan_Approval_Predictor_model.pkl", 
    "data/processed/processed_loan_approval_dataset.csv", 
    "outputs/Scaler"
)

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("Loan Approval Prediction App")
st.image("data/raw/loan approval prediction system.jpeg", use_container_width=True)
st.write("""
This app predicts whether a **loan application will be approved** based on financial and personal features.
""")

# Sidebar Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio(
    "Go To", ["Overview", "Data Exploration", "Make Prediction"]
)

# ------------------------------
# Overview Section
# ------------------------------
if options == "Overview":
    st.header("About the Project")
    st.write("""
    - Dataset: Loan Approval Dataset (Kaggle)  
    - Goal: Predict loan approval (Approved / Rejected)  
    - Features: Income, Loan Amount, Number of Dependents, Employment Status, Assets, etc.
    """)

    # Loan Status Reference Table
    loan_status_ref = pd.DataFrame({
        "Class (Loan_Status)": [0, 1],
        "Loan Status Name": ["Rejected", "Approved"],
        "Description": [
            "The loan application was not approved by the bank or financial institution.",
            "The loan application was approved and the applicant will receive the funds."
        ]
    })

    # Main Factors Affecting Approval
    loan_factors = pd.DataFrame({
        "Feature": [
            "income_annum",
            "loan_amount",
            "no_of_dependents",
            "self_employed",
            "education",
            "cibil_score",
            "residential_assets_value",
            "commercial_assets_value",
            "luxury_assets_value",
            "bank_asset_value",
            "loan_term"
        ],
        "Impact on Approval": [
            "Higher income increases likelihood of approval.",
            "Very high loan amount may reduce approval chances.",
            "More dependents can reduce approval probability.",
            "Self-employed applicants may be scrutinized more closely.",
            "Graduates tend to have higher approval chances.",
            "Higher CIBIL score strongly increases approval probability.",
            "Higher residential assets add to applicant credibility.",
            "Commercial assets improve applicant financial stability.",
            "Luxury assets may indicate financial capability.",
            "Higher bank asset value improves approval chances.",
            "Longer loan term may be favorable or unfavorable depending on risk assessment."
        ]
    })

    st.title("Loan Approval Reference")
    st.subheader("Loan Status Reference")
    st.dataframe(loan_status_ref)

    st.subheader("Main Factors Influencing Loan Approval")
    st.dataframe(loan_factors)

# ------------------------------
# Data Exploration
# ------------------------------
elif options == "Data Exploration":
    st.subheader("Sample Data")
    st.dataframe(df.head())
    st.write(f"Dataset shape: {df.shape}")
    st.write(df.columns)

    st.subheader("Feature Distribution")
    feature = st.selectbox("Select feature to visualize", df.columns)
    st.bar_chart(df[feature].value_counts())

    st.subheader("Correlation HeatMap")
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), ax=ax, cmap="coolwarm")
    st.pyplot(fig)

# ------------------------------
# Make Prediction
# ------------------------------
elif options == "Make Prediction":
    st.subheader("Predict Loan Approval")
    st.write("Enter the applicant’s financial and personal details below:")

    # Numeric Inputs
    no_of_dependents = st.number_input("Number of Dependents (0–5)", 0, 5, 3)
    income_annum = st.number_input("Annual Income (₹)", 200000, 9900000, 500000)
    loan_amount = st.number_input("Loan Amount (₹)", 300000, 39500000, 1000000)
    loan_term = st.number_input("Loan Term (Years)", 2, 20, 10)
    cibil_score = st.number_input("CIBIL Score", 300, 900, 700)
    residential_assets_value = st.number_input("Residential Assets Value (₹)", -100000, 29100000, 5000000)
    commercial_assets_value = st.number_input("Commercial Assets Value (₹)", 0, 19400000, 2000000)
    luxury_assets_value = st.number_input("Luxury Assets Value (₹)", 300000, 39200000, 1000000)
    bank_asset_value = st.number_input("Bank Asset Value (₹)", 0, 14700000, 500000)

    # Categorical Inputs
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed?", ["No", "Yes"])

    user_input = {
        " no_of_dependents": no_of_dependents,
        " income_annum": income_annum,
        " loan_amount": loan_amount,
        " loan_term": loan_term,
        " cibil_score": cibil_score,
        " residential_assets_value": residential_assets_value,
        " commercial_assets_value": commercial_assets_value,
        " luxury_assets_value": luxury_assets_value,
        " bank_asset_value": bank_asset_value,
        " education_ Not Graduate": 1 if education == "Not Graduate" else 0,
        " self_employed_ Yes": 1 if self_employed == "Yes" else 0
    }


    # Show Input Summary
    st.subheader("Applicant Details")
    st.write(pd.DataFrame(user_input, index=[0]))

    # Create DataFrame for Model
    feature_names = df.drop(columns=[" loan_status"]).columns
    input_features = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
    for feature, value in user_input.items():
        if feature in input_features.columns:
            input_features[feature] = value

    # Scale numeric features only
    numeric_cols = [" no_of_dependents"," income_annum"," loan_amount"," loan_term",
                    " cibil_score"," residential_assets_value"," commercial_assets_value",
                    " luxury_assets_value"," bank_asset_value"]
    input_features[numeric_cols] = scaler.transform(input_features[numeric_cols])

    # Prediction
    if st.button("Predict Loan Approval"):
        result = model.predict(input_features)[0]
        status = "Approved ✅" if result == 1 else "Rejected ❌"
        st.success(f"Loan Status Prediction: {status}")

        if result == 1:
            st.image("data/raw/loan approved.png")
        else:
            st.image("data/raw/loan rejected.jpeg")
