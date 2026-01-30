import streamlit as st
import pandas as pd
import pickle

model = pickle.load(open("disease_model.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

df = pd.read_csv("dataset.csv")
symptoms = df.columns[:-1]  # all symptom names

st.title("Medical Disease Prediction System")

selected_symptoms = st.multiselect(
    "Select symptoms you have:",
    symptoms
)

if st.button("Predict Disease"):
    input_data = [0] * len(symptoms)

    for symptom in selected_symptoms:
        index = list(symptoms).index(symptom)
        input_data[index] = 1

    prediction = model.predict([input_data])
    disease = le.inverse_transform(prediction)

    st.success(f"Predicted Disease: {disease[0]}")
