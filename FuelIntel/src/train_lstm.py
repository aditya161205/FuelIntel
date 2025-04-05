import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import os

# Load data
df = pd.read_csv("data/fuel_data.csv")

# Select features
features = ['EquipmentHours', 'IdleTime', 'FuelAdded', 'PrevFuelConsumption']
target = 'FuelConsumption'

# Scale data
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])

# Save the scaler
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.save")

# Create sequences
X, y = [], []
sequence_length = 5
for i in range(len(scaled) - sequence_length):
    X.append(scaled[i:i+sequence_length])
    y.append(df[target].iloc[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Build LSTM model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(sequence_length, len(features))),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=1)

# Save model
model.save("model/fuel_model.h5")
