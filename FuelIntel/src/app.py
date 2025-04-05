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
    sequence: list[list[float]]  # 2D list: shape (5 timesteps, n_features)

@app.post("/predict")
def predict_fuel(data: FuelSequence):
    # Convert and scale input
    input_seq = np.array(data.sequence)
    scaled_seq = scaler.transform(input_seq)  # shape: (5, n_features)
    scaled_seq = scaled_seq.reshape(1, 5, -1)

    # Predict
    prediction = model.predict(scaled_seq)[0][0]

    # Extract last known feature values for logic (most recent timestep)
    latest = input_seq[-1]  # shape: (n_features,)
    
    # Feature index reference (customize based on your data column order)
    FUEL_LEVEL_IDX = 4
    IDLE_TIME_IDX = 7
    ENGINE_TEMP_IDX = 15
    FUEL_EFFICIENCY_IDX = 21

    fuel_level = latest[FUEL_LEVEL_IDX]
    idle_time = latest[IDLE_TIME_IDX]
    engine_temp = latest[ENGINE_TEMP_IDX]
    fuel_efficiency = latest[FUEL_EFFICIENCY_IDX]

    # Rules for anomaly detection
    anomalies = []
    if fuel_level < 10:
        anomalies.append("Possible fuel theft")
    if idle_time > 180:
        anomalies.append("High idle time")
    if prediction > 80:
        anomalies.append("Abnormal fuel usage")

    # Rules for insights/tips
    tips = []
    if fuel_efficiency < 4:
        tips.append("Consider optimizing driving style")
    if engine_temp > 95:
        tips.append("Check engine cooling system")
    if fuel_level < 15:
        tips.append("Refuel soon to avoid dry tank")

    return {
        "predicted_fuel_consumption": round(prediction, 2),
        "anomalies": anomalies if anomalies else ["None"],
        "efficiency_tips": tips if tips else ["All good"]
    }
