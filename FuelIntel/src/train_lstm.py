import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import joblib
import os

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
df = pd.read_csv("../data/fuel_data.csv")

# -----------------------------
# Step 2: Select Important Features
# -----------------------------
selected_features = [
    'Fuel Consumed (L/hr)',
    'Fuel Tank Capacity (L)',
    'Idle Time (min)',
    'Average Load Last 5hr (%)',
    'Throttle Position (%)',
    'Battery Voltage (V)',
    'Engine Temperature (°C)',
    'Oil Pressure (psi)',
    'Fuel Efficiency (km/L)'
]

df = df[selected_features].copy()
df.fillna(method='ffill', inplace=True)  # Fill missing values forward

# -----------------------------
# Step 3: Scale the Data
# -----------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# Save the scaler for use during prediction
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.save")  # <-- Matches your FastAPI load

# -----------------------------
# Step 4: Create Sequences for LSTM
# -----------------------------
sequence_length = 5  # Using past 5 records
target_col = selected_features.index("Fuel Consumed (L/hr)")

X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i][target_col])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)  # (samples, sequence_length, features)
print("y shape:", y.shape)

# -----------------------------
# Step 5: Build LSTM Model
# -----------------------------
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(1))  # Predicting one value: fuel consumption

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# -----------------------------
# Step 6: Train the Model
# -----------------------------
model.fit(X, y, epochs=50, batch_size=4, validation_split=0.1)

# -----------------------------
# Step 7: Save the Model
# -----------------------------
model.save("model/fuel_model.h5")
print("✅ Model and scaler saved!")
