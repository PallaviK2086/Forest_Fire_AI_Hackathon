# Forest_Fire_AI_Hackathon
India Fire Risk Predictor
This project is a machine learning–based India Fire Risk Predictor that estimates whether a location is at high or low forest‑fire risk. It uses satellite‑derived fire information and a Random Forest classifier, deployed through a simple Streamlit web app.​

Project Overview
Goal: Help prioritize potentially risky locations by predicting fire risk from satellite‑based brightness and related features.​

Type: End‑to‑end ML pipeline – data preprocessing, model training, evaluation, and deployment in a web interface.​

Features
User‑friendly Streamlit UI for entering input values such as latitude, longitude, brightness, month, etc.​

Backend Random Forest model that outputs a fire‑risk class (High / Low) along with a probability score.​

Modular code structure with separate scripts for data preparation, model training, and the app (app.py).​

Tech Stack
Programming language: Python

Libraries: Streamlit, scikit‑learn, pandas, numpy, joblib (or pickle) for model saving/loading.​

How to Run
Clone the repository:
git clone <repo_link>

Install dependencies:
pip install -r requirements.txt

Run the Streamlit app:
streamlit run app.py

Open the local URL shown in the terminal (usually http://localhost:8501) in your browser.​

Model and Data
Dataset: Processed satellite fire data (e.g., NASA FIRMS subset for India) with features such as latitude, longitude, brightness, and month.​

Model: Random Forest classifier, chosen for good performance on tabular data, robustness to overfitting, and easy interpretability in a student project.​

Future Improvements
Add more environmental features (temperature, humidity, wind, vegetation indices) to improve accuracy.​

Experiment with other models (e.g., Gradient Boosting, XGBoost) and hyperparameter tuning.​

Deploy the app to a cloud platform for public acces
