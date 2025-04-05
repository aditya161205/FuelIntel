from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib

app = FastAPI()

# Load model and scaler
model = tf.keras.models.load_model("model/fuel_model.h5")
scaler = joblib.load("model/scaler.save")

# Input format
class FuelSequence(BaseModel):
    sequence: list[list[float]]

@app.post("/predict")
def predict_fuel(data: FuelSequence):
    input_seq = np.array(data.sequence).reshape(1, 5, -1)
    prediction = model.predict(input_seq)[0][0]

    # Dummy logic for alerts and tips
    alert = "None"
    tip = "Maintain regular idle checks"
    if prediction < 20:
        alert = "Low fuel usage – possible theft"
    elif prediction > 80:
        tip = "Reduce excessive fuel usage"

    return {
        "predicted_fuel_consumption": round(prediction, 2),
        "anomalies": alert,
        "efficiency_tips": tip
    }
