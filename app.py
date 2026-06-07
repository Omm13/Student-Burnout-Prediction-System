import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# -----------------------------
# LOAD MODEL & FEATURES
# -----------------------------
model = joblib.load("data/processed/burnout_model.pkl")
feature_names = joblib.load("feature_names.pkl")

st.set_page_config(page_title="Burnout Predictor", layout="centered")

st.title("🎓 Student Burnout Prediction System")
st.write("Enter student details to predict burnout risk")

# -----------------------------
# USER INPUTS
# -----------------------------
study_hours = st.slider("Study Hours per Day", 0, 15, 6)
sleep_hours = st.slider("Sleep Hours per Day", 0, 12, 7)
gpa = st.slider("GPA", 0.0, 10.0, 7.0)

social_hours = st.slider("Social Hours", 0, 10, 2)
physical_activity = st.slider("Physical Activity", 0, 5, 1)

age = st.slider("Age", 16, 35, 21)
gender = st.selectbox("Gender", ["Male", "Female"])

financial_stress = st.slider("Financial Stress (1-5)", 1, 5, 3)
study_satisfaction = st.slider("Study Satisfaction (1-5)", 1, 5, 3)

peer_competition = st.slider("Peer Competition (1-5)", 1, 5, 3)
relationship_stress = st.slider("Relationship Stress (1-5)", 1, 5, 2)

sleep_problems = st.slider("Sleep Problems (1-5)", 1, 5, 2)
irritability = st.slider("Irritability (1-5)", 1, 5, 2)

# Encode gender
gender_val = 1 if gender == "Male" else 0

# -----------------------------
# CREATE INPUT DATAFRAME
# -----------------------------
input_data = pd.DataFrame([{
    "study_hours": study_hours,
    "sleep_hours": sleep_hours,
    "social_hours": social_hours,
    "physical_activity": physical_activity,
    "gpa": gpa,
    "gender": gender_val,
    "age": age,
    "financial_stress": financial_stress,
    "study_satisfaction": study_satisfaction,
    "peer_competition": peer_competition,
    "relationship_stress": relationship_stress,
    "sleep_problems": sleep_problems,
    "irritability": irritability
}])

# Ensure same feature order as training
input_data = input_data[feature_names]

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Burnout"):

    prediction = model.predict(input_data)[0]

    st.subheader(f"Predicted Burnout Level: {prediction}")

    # -----------------------------
    # COLOR OUTPUT
    # -----------------------------
    if prediction == "High":
        st.error("High Burnout Risk ⚠️")
    elif prediction == "Medium":
        st.warning("Medium Burnout Risk ⚡")
    else:
        st.success("Low Burnout Risk ✅")

    # -----------------------------
    # SUGGESTIONS
    # -----------------------------
    st.write("### Suggestions:")

    if study_hours > 8:
        st.write("- Reduce study hours")

    if sleep_hours < 6:
        st.write("- Improve sleep duration")

    if sleep_problems > 3:
        st.write("- Work on sleep quality")

    if financial_stress > 3:
        st.write("- Manage financial stress")

    # -----------------------------
    # SHAP EXPLANATION (FIXED)
    # -----------------------------
    st.write("### Feature Impact (SHAP)")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_data)

    # Get predicted class index
    class_index = model.predict_proba(input_data).argmax()

    fig = plt.figure()
    shap.plots.waterfall(shap_values[0, :, class_index], show=False)
    st.pyplot(fig)