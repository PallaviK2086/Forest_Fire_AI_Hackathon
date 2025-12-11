import streamlit as st
import numpy as np
import joblib

# Load trained model and feature list
model = joblib.load("fire_risk_rf.joblib")
features = joblib.load("features.joblib")  # ["latitude","longitude","brightness","temp","humidity","month"]

st.title("India Fire Risk Predictor")

st.write(
    "This app uses a Random Forest model trained on NASA FIRMS fire detections "
    "and weather data to estimate whether conditions are high or low fire risk."
)

# Inputs
col1, col2 = st.columns(2)
with col1:
    lat = st.number_input("Latitude", value=20.0, format="%.4f")
    brightness = st.number_input("Brightness", value=330.0)
    humidity = st.number_input("Humidity (%)", value=40.0)
with col2:
    lon = st.number_input("Longitude", value=78.0, format="%.4f")
    temp = st.number_input("Temperature (°C)", value=30.0)
    month = st.slider("Month", 1, 12, 5)

if st.button("Predict Fire Risk"):
    x = np.array([[lat, lon, brightness, temp, humidity, month]])
    prob_high = float(model.predict_proba(x)[0][1])
    pred_class = int(model.predict(x)[0])

    label = "High Fire Risk" if pred_class == 1 else "Low Fire Risk"
    st.subheader(label)
    st.write(f"Predicted probability of high fire risk: {prob_high:.2f}")

    st.caption(
        "Model: RandomForestClassifier trained on historical FIRMS fire points "
        "with weather and seasonal features."
)
