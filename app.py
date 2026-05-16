
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Define the path to the models
# Using relative paths for better deployment flexibility
MODEL_DIR = os.path.dirname(__file__) # This will be '/content/drive/MyDrive/ecoride/' when run from there
PIPELINE_PATH = os.path.join(MODEL_DIR, 'pipeline_preproc.pkl')
MODEL_PATH = os.path.join(MODEL_DIR, 'modelo_churn.pkl')

# Load the pipeline and model
@st.cache_resource
def load_models():
    try:
        pipeline = joblib.load(PIPELINE_PATH)
        model = joblib.load(MODEL_PATH)
        return pipeline, model
    except FileNotFoundError:
        st.error(f"Error: Model files not found. Please ensure '{PIPELINE_PATH}' and '{MODEL_PATH}' exist.")
        st.stop()

pipeline_preproc, modelo_churn = load_models()

# Streamlit App Title
st.title('Sistema de Alerta Temprana de Churn - Eco-Ride')

st.write("Introduce los datos del cliente para evaluar su riesgo de abandono (churn).")

# Input controls
edad = st.slider('Edad', min_value=18, max_value=100, value=30)
plan = st.selectbox('Plan', ['básico', 'premium', 'elite'])
uso_mensual_km = st.slider('Uso Mensual (Km)', min_value=0, max_value=200, value=100)
soporte_tickets = st.slider('Soporte Tickets', min_value=0, max_value=7, value=1)
gasto_promedio = st.slider('Gasto Promedio', min_value=0.0, max_value=500.0, value=50.0, format='%.2f')
dias_antiguedad = st.slider('Días Antigüedad', min_value=0, max_value=1825, value=365) # Max 5 years
region = st.selectbox('Región', ['Norte', 'Sur', 'Centro'])

# Create a DataFrame for prediction
# IMPORTANT: Pass Plan and Region as strings directly, the pipeline is expected to handle encoding.
input_data = pd.DataFrame({
    'Edad': [edad],
    'Plan': [plan],
    'Uso_Mensual_Km': [uso_mensual_km],
    'Soporte_Tickets': [soporte_tickets],
    'Gasto_Promedio': [gasto_promedio],
    'Dias_Antiguedad': [dias_antiguedad],
    'Region': [region]
})

# Button to analyze risk
if st.button('Analizar Riesgo'):
    # Preprocess the input data
    try:
        processed_data = pipeline_preproc.transform(input_data)
        
        # Make prediction
        churn_probability = modelo_churn.predict_proba(processed_data)[:, 1][0]
        
        # Decision threshold
        CHURN_THRESHOLD = 0.20
        churn_prediction = 1 if churn_probability >= CHURN_THRESHOLD else 0
        
        st.subheader('Resultado del Análisis:')
        if churn_prediction == 1:
            st.markdown(f"<h3 style='color: red;'>¡ALERTA! Alto riesgo de abandono. Probabilidad: {churn_probability:.2f}</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color: green;'>Bajo riesgo de abandono. Probabilidad: {churn_probability:.2f}</h3>", unsafe_allow_html=True)
            
        st.write(f"(Umbral de decisión para CHURN: {CHURN_THRESHOLD})")
            
    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")

