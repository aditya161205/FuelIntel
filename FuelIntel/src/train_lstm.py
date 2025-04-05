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
df = pd.read_csv("data/fuel_data.csv")

# -----------------------------
# Step 2: Select Important Features
# -----------------------------
selected_features = [
    "Fuel_Level (L)", "Fuel_Consumption", "Fuel_Tank Capacity", "Fuel_Efficiency",
    "Idle_Time (min)", "Load (%)", "RPM", "Time (min)", "Average_Load",
    "Speed (km/h)", "Throttle_Pos", "Battery_Volt",
    "Engine_Temperature", "Oil_Pressure",
    "Fuel_Price (₹)", "Temperature"
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
joblib.dump(scaler, "model/fuel_scaler.pkl")

# -----------------------------
# Step 4: Create Sequences for LSTM
# -----------------------------
sequence_length = 5  # Suitable for small dataset
target_col = selected_features.index("Fuel_Consumption")

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
