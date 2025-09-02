import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import warnings

# Suppress the UserWarning from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning)

# --- Load the Models and Scaler ---
# Make sure these files are in the same directory as this script.
try:
    risk_classifier = joblib.load('risk_strat_classifier.joblib')
    los_regressor = joblib.load('los_regressor.joblib')
    kmeans_model = joblib.load('kmeans_model.joblib')
    cluster_scaler = joblib.load('cluster_scaler.joblib')
    # Load the new association rule models
    rules = joblib.load('association_rules.joblib')
except FileNotFoundError:
    st.error("Model files not found! Please ensure all .joblib files are in the same directory and pushed to your GitHub repository.")
    st.stop()

# --- App Title and Description ---
st.title("👨‍⚕️ HealthAI Suite: Patient Analytics")
st.markdown("### Intelligent Analytics for Patient Care")
st.markdown("""
This application predicts a patient's **disease risk**, **length of stay**, and potential **future disease associations** based on their health metrics.
""")

# --- Feature Definitions for UI ---
GENDER_OPTIONS = ['Male', 'Female', 'Other']
DIAGNOSIS_OPTIONS = ['Hypertension', 'Diabetes', 'Asthma', 'Arthritis', 'Migraine', 'Cold', 'Pneumonia', 'None']
MEDICATION_OPTIONS = ['Statins', 'Insulin', 'Antibiotics', 'Beta-blockers', 'Painkillers', 'Antihistamines', 'None']
PROCEDURE_OPTIONS = ['ECG', 'Blood Test', 'X-Ray', 'Physical Exam', 'Biopsy', 'MRI Scan', 'None']
BMI_CATEGORY_OPTIONS = ['Underweight', 'Normal', 'Overweight', 'Obese']

# --- User Input Fields ---
st.header("Patient Information")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 20, 90, 45)
    gender = st.selectbox("Gender", GENDER_OPTIONS)
    systolic_bp = st.slider("Systolic BP", 80, 200, 120)
    diastolic_bp = st.slider("Diastolic BP", 50, 120, 80)
    heart_rate = st.slider("Heart Rate", 50, 120, 75)
    
with col2:
    cholesterol = st.slider("Cholesterol", 100, 300, 200)
    blood_sugar = st.slider("Blood Sugar", 50, 200, 100)
    bmi = st.slider("BMI", 15.0, 40.0, 25.0)
    diagnosis = st.selectbox("Diagnosis", DIAGNOSIS_OPTIONS)
    medication = st.selectbox("Medication", MEDICATION_OPTIONS)
    procedure = st.selectbox("Procedure", PROCEDURE_OPTIONS)

# --- Prediction Button ---
if st.button("Predict"):
    # --- 1. Preprocess the Input Data ---
    
    input_df = pd.DataFrame([{
        'age': age,
        'gender': gender,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'heart_rate': heart_rate,
        'cholesterol': cholesterol,
        'blood_sugar': blood_sugar,
        'bmi': bmi,
        'diagnosis': diagnosis,
        'medication': medication,
        'procedure': procedure
    }])
    
    # Feature Engineering
    input_df['hypertension'] = ((input_df['systolic_bp'] > 130) | (input_df['diastolic_bp'] > 80)).astype(int)
    def bmi_category(bmi):
        if bmi < 18.5: return 'Underweight'
        if bmi < 24.9: return 'Normal'
        if bmi < 29.9: return 'Overweight'
        return 'Obese'
    input_df['bmi_category'] = input_df['bmi'].apply(bmi_category)
    
    categorical_features = ['gender', 'diagnosis', 'medication', 'procedure', 'bmi_category']
    all_categories = {
        'gender': ['Female', 'Male', 'Other'],
        'diagnosis': ['Arthritis', 'Asthma', 'Cold', 'Diabetes', 'Hypertension', 'Migraine', 'None', 'Pneumonia'],
        'medication': ['Antihistamines', 'Antibiotics', 'Beta-blockers', 'Insulin', 'None', 'Painkillers', 'Statins'],
        'procedure': ['Biopsy', 'Blood Test', 'ECG', 'MRI Scan', 'None', 'Physical Exam', 'X-Ray'],
        'bmi_category': ['Normal', 'Obese', 'Overweight', 'Underweight']
    }
    
    for col, categories in all_categories.items():
        for cat in categories:
            col_name = f'{col}_{cat}'
            input_df[col_name] = (input_df[col] == cat).astype(int)

    input_df = input_df.drop(columns=categorical_features, errors='ignore')
    
    training_cols = risk_classifier.feature_names_in_
    input_df = input_df.reindex(columns=training_cols, fill_value=0)

    # --- 2. Make Predictions ---
    
    # Prediction 1: Disease Risk
    risk_prediction = risk_classifier.predict(input_df)[0]
    risk_map = {0: 'Low', 1: 'Moderate', 2: 'High'}
    predicted_risk = risk_map.get(risk_prediction, "Unknown")

    # Prediction 2: Length of Stay
    los_prediction = los_regressor.predict(input_df)[0]
    
    # Prediction 3: Patient Cluster
    X_cluster_input = input_df[['age', 'systolic_bp', 'diastolic_bp', 'heart_rate', 'cholesterol', 'blood_sugar', 'bmi']]
    X_cluster_scaled = cluster_scaler.transform(X_cluster_input)
    cluster_prediction = kmeans_model.predict(X_cluster_scaled)[0]

    # Prediction 4: Future Disease Association
    user_antecedents = set()
    if diagnosis != 'None': user_antecedents.add(diagnosis)
    if medication != 'None': user_antecedents.add(medication)
    if procedure != 'None': user_antecedents.add(procedure)

    relevant_rules = rules[rules['antecedents'].apply(lambda x: user_antecedents.issuperset(set(x)))]
    
    relevant_rules = relevant_rules.sort_values(by='confidence', ascending=False)
    
    top_associations = relevant_rules.head(3)

    # --- 3. Display Results ---
    st.header("Prediction Results")
    st.markdown("---")

    col_res1, col_res2, col_res3 = st.columns(3)
    
    with col_res1:
        st.subheader("Disease Risk")
        st.metric(label="Predicted Risk", value=f"**{predicted_risk}**")
        st.caption("_(Based on classification model)_")

    with col_res2:
        st.subheader("Length of Stay")
        st.metric(label="Predicted Days", value=f"**{los_prediction:.1f}**")
        st.caption("_(Based on regression model)_")

    with col_res3:
        st.subheader("Patient Cluster")
        st.metric(label="Predicted Cluster", value=f"**{cluster_prediction}**")
        st.caption("_(Based on clustering model)_")

    st.markdown("---")
    st.subheader("Potential Future Associations")
    
    if not top_associations.empty:
        st.markdown("Based on common patterns, patients with a similar profile are also associated with:")
        for idx, row in top_associations.iterrows():
            antecedents = ', '.join(list(row['antecedents']))
            consequents = ', '.join(list(row['consequents']))
            confidence = row['confidence']
            st.info(f"**Associated with:** {consequents}  \n**Reason:** Commonly found with {antecedents} (Confidence: {confidence:.2f})")
    else:
        st.info("No strong associations found based on the provided inputs.")
