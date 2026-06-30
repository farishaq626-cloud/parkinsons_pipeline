import streamlit as st
import requests

# Page Layout
st.set_page_config(page_title="Parkinson's Prediction Dashboard", layout="centered")

st.title("Parkinson's Prediction Dashboard")
st.markdown("Enter patient metrics below to generate a risk stratification.")

# Sidebar for Inputs
st.sidebar.header("Clinical Inputs")
age = st.sidebar.slider("Patient Age at Onset", 40, 90, 60)
updrs = st.sidebar.number_input("UPDRS Motor Score", 0.0, 100.0, 20.0)
cog = st.sidebar.number_input("Cognitive Score", 0.0, 30.0, 25.0)

# The Prediction Button
if st.sidebar.button("Run Prediction"):
    payload = {
        "age": age,
        "updrs": updrs,
        "cognitive_score": cog
    }
    
    try:
        # Send request to the API
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            
            # Display Results
            st.success("Analysis Complete")
            st.metric(label="Risk Assessment", value=result["prediction"])
            st.progress(float(result["probability"]))
            st.caption(f"Progression Probability: {result['probability']*100:.1f}%")
        else:
            st.error(f"Error: API returned {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        st.error("Connection Error: Make sure your 'api.py' is running!")