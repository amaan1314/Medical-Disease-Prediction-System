import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Medical Disease Prediction System")

# Load dataset
df = pd.read_csv("dataset.csv")

X = df.drop("prognosis", axis=1)
y = df["prognosis"]

# Train model if not found
if not os.path.exists("disease_model.pkl") or not os.path.exists("label_encoder.pkl"):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    model = RandomForestClassifier()
    model.fit(X, y_encoded)

    pickle.dump(model, open("disease_model.pkl", "wb"))
    pickle.dump(le, open("label_encoder.pkl", "wb"))
else:
    model = pickle.load(open("disease_model.pkl", "rb"))
    le = pickle.load(open("label_encoder.pkl", "rb"))

# Streamlit UI
st.title("🏥 Medical Disease Prediction System")
st.write("Select the symptoms you are experiencing:")

symptoms = X.columns.tolist()

selected_symptoms = st.multiselect(
    "Symptoms",
    symptoms
)

if st.button("Predict Disease"):
    input_data = [0] * len(symptoms)

    for symptom in selected_symptoms:
        index = symptoms.index(symptom)
        input_data[index] = 1

    prediction = model.predict([input_data])
    disease = le.inverse_transform(prediction)

    st.success(f"🩺 Predicted Disease: {disease[0]}")
