
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Define the path to the models
MODEL_DIR = '/content/drive/MyDrive/ecoride/'
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
uso_mensual_km = st.slider('Uso Mensual (Km)', min_value=0, max_value=1000, value=100)
soporte_tickets = st.slider('Soporte Tickets', min_value=0, max_value=7, value=1)
gasto_promedio = st.slider('Gasto Promedio', min_value=0.0, max_value=500.0, value=50.0, format='%.2f')
dias_antiguedad = st.slider('Días Antigüedad', min_value=0, max_value=365*5, value=365)
region = st.selectbox('Región', ['Norte', 'Sur', 'Centro'])

# Map categorical features for consistent input to the pipeline/model
def map_plan(plan_str):
    return {
        'básico': 0,
        'premium': 1,
        'elite': 2
    }.get(plan_str, -1) # Default to -1 or handle error

def map_region(region_str):
    return {
        'Norte': 0,
        'Sur': 1,
        'Centro': 2
    }.get(region_str, -1) # Default to -1 or handle error

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    'Edad': [edad],
    'Plan': [map_plan(plan)],
    'Uso_Mensual_Km': [uso_mensual_km],
    'Soporte_Tickets': [soporte_tickets],
    'Gasto_Promedio': [gasto_promedio],
    'Dias_Antiguedad': [dias_antiguedad],
    'Region': [map_region(region)]
})

# Button to analyze risk
if st.button('Analizar Riesgo'):
    # Preprocess the input data
    try:
        processed_data = pipeline_preproc.transform(input_data)
        
        # Make prediction
        churn_probability = modelo_churn.predict_proba(processed_data)[:, 1][0]
        churn_prediction = 1 if churn_probability >= 0.20 else 0
        
        st.subheader('Resultado del Análisis:')
        if churn_prediction == 1:
            st.markdown(f"<h3 style='color: red;'>¡ALERTA! Alto riesgo de abandono. Probabilidad: {churn_probability:.2f}</h3>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h3 style='color: green;'>Bajo riesgo de abandono. Probabilidad: {churn_probability:.2f}</h3>", unsafe_allow_html=True)
            
        st.write(f"(Umbral de decisión para CHURN: 0.20)")
            
    except Exception as e:
        st.error(f"Ocurrió un error durante la predicción: {e}")

