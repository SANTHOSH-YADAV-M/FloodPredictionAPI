
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Flood Prediction API")

model = joblib.load("flood_risk_model.joblib")

FEATURES = [
    "rain_adc",
    "rain_intensity",
    "water_level_cm",
    "rise_rate",
    "rain_duration_min",
    "recent_rain_fraction"
]

class SensorData(BaseModel):
    rain_adc: int
    rain_intensity: float
    water_level_cm: float
    rise_rate: float
    rain_duration_min: float
    recent_rain_fraction: float

@app.get("/")
def home():
    return {"message": "Flood Prediction API is running"}

@app.post("/predict")
def predict(data: SensorData):

    values = pd.DataFrame([{
        "rain_adc": data.rain_adc,
        "rain_intensity": data.rain_intensity,
        "water_level_cm": data.water_level_cm,
        "rise_rate": data.rise_rate,
        "rain_duration_min": data.rain_duration_min,
        "recent_rain_fraction": data.recent_rain_fraction
    }], columns=FEATURES)

    prediction = model.predict(values)[0]
    probabilities = model.predict_proba(values)[0]

    confidence = float(max(probabilities))

    return {
        "flood_risk": str(prediction),
        "confidence": round(confidence, 4)
    }
