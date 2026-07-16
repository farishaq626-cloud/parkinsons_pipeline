from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Initialize the API
app = FastAPI()

# This defines the data structure the dashboard will send to the API
class PredictionRequest(BaseModel):
    age: float
    updrs: float
    cognitive_score: float

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # --- INSERT YOUR MODEL LOGIC HERE ---
        # Currently, this is a placeholder. When you are ready to use your 
        # actual model, you would import it here and call: 
        # prediction = my_model.predict([[request.age, request.updrs, request.cognitive_score]])
        
        # Mock logic to show it's working
        if request.updrs > 50:
            prediction = "High Risk of Progression"
            probability = 0.85
        else:
            prediction = "Stable"
            probability = 0.15
            
        return {
            "prediction": prediction,
            "probability": probability
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)