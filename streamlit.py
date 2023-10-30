import streamlit as st
import numpy as np
import pandas as pd
import joblib

def main():
    st.title("Heart Disease Detection App")

    # Load the model
    model = joblib.load('heart_disease_model.pkl')

    # Get user input for each feature
    age = st.slider("Age", 0, 100, 50)
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.slider("Resting Blood Pressure", 90, 200, 120)
    chol = st.slider("Cholesterol", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
    thalach = st.slider("Maximum Heart Rate Achieved", 70, 220, 150)
    exang = st.radio("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.slider("Oldpeak", 0.0, 6.2, 3.1)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment", [0, 1, 2])
    ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 1)
    thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

    # Convert categorical features to numerical
    sex_mapping = {"Male": 1, "Female": 0}
    fbs_mapping = {"<= 120 mg/dl": 1, "> 120 mg/dl": 0}
    exang_mapping = {"No": 0, "Yes": 1}

    sex = sex_mapping[sex]
    fbs = fbs_mapping[fbs]
    exang = exang_mapping[exang]

    # Button to predict heart disease
    if st.button("Predict Heart Disease"):
        # Perform input data preprocessing
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'cp': [cp],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs],
            'restecg': [restecg],
            'thalach': [thalach],
            'exang': [exang],
            'oldpeak': [oldpeak],
            'slope': [slope],
            'ca': [ca],
            'thal': [thal]
        })

        # Make the prediction using the model
        result = model.predict(input_data)

        # Display the result
        st.success(f"The model predicts: {'Heart Disease' if result[0] == 1 else 'No Heart Disease'}")

if __name__ == "__main__":
    main()
