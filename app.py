import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load(r"C:\APP CREATE\survival_prediction_model.pkl")

# Extract the best model if it's a GridSearchCV object
if hasattr(model, "best_estimator_"):
    best_model = model.best_estimator_
else:
    best_model = model

# Ensure the model has feature names (LightGBM models use booster_)
if hasattr(best_model, "feature_name_"):
    feature_names = best_model.feature_name_
elif hasattr(best_model, "get_booster"):
    feature_names = best_model.get_booster().feature_name()
else:
    feature_names = None  # Handle other cases

# Title of the app
st.title("HCT Survival Prediction App")
st.write("This app predicts survival outcomes for patients who have undergone Hematopoietic Cell Transplantation (HCT).")

# Input fields for user data
st.header("Enter Patient Details")
age_at_hct = st.number_input("Age at HCT", min_value=0, max_value=100, value=50)
donor_age = st.number_input("Donor Age", min_value=0, max_value=100, value=40)
karnofsky_score = st.number_input("Karnofsky Score", min_value=0, max_value=100, value=90)
comorbidity_score = st.number_input("Comorbidity Score", min_value=0, max_value=10, value=2)
dri_score = st.number_input("DRI Score", min_value=0.0, max_value=10.0, value=2.5)
cyto_score = st.number_input("Cytogenetic Risk Score", min_value=0.0, max_value=10.0, value=1.5)
tbi_status = st.selectbox("TBI Status", ["Yes", "No"])

# Create a dictionary to map user inputs to feature names
input_data = {
    'age_at_hct': age_at_hct,
    'donor_age': donor_age,
    'karnofsky_score': karnofsky_score,
    'comorbidity_score': comorbidity_score,
    'dri_score': dri_score,
    'cyto_score': cyto_score,
    'tbi_status': 1 if tbi_status == "Yes" else 0,
}

# Convert input data to a DataFrame
input_df = pd.DataFrame([input_data])

# Ensure all columns used during training are present
if feature_names:
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match the training data
    input_df = input_df[feature_names]

# Predict button
if st.button("Predict Survival"):
    # Make prediction
    prediction = best_model.predict_proba(input_df)[:, 1]
    survival_probability = prediction[0]

    # Display result
    st.subheader("Prediction Result")
    st.write(f"The predicted probability of survival is: **{survival_probability:.2f}**")
    if survival_probability >= 0.5:
        st.success("The patient is likely to survive.")
    else:
        st.error("The patient is unlikely to survive.")
