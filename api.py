from fastapi import FastAPI
from pydantic import BaseModel
import model_engine

# Initialize the API
app = FastAPI(title="Parkinsons Pipeline API")

# Define the data structure for the input
class PatientData(BaseModel):
    patient_id: str
    age: float
    updrs_motor_score: float

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Parkinsons Pipeline API is active."}

# Prediction endpoint
@app.post("/predict")
def predict_risk(data: PatientData):
    # This calls your existing model_engine logic
    # Ensure model_engine has a function like 'get_prediction'
    prediction = model_engine.get_prediction(
        data.patient_id, 
        data.age, 
        data.updrs_motor_score
    )
    return {"patient_id": data.patient_id, "prediction": prediction}