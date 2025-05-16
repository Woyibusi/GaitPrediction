import numpy as np
from tensorflow.keras.models import load_model
import joblib

# === CONFIG ===
MODEL_PATH = "../models/best_lstm_model.h5"
SCALER_PATH = "../data/scaler.pkl"
INPUT_SEQUENCE = "../data/new_input.npy"  # shape: (48, 4)

# === LOAD MODEL & SCALER ===
print("📂 Loading model and scaler...")
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# === LOAD AND NORMALIZE INPUT ===
X = np.load(INPUT_SEQUENCE)  # shape: (48, 4)
X_scaled = scaler.transform(X)  # shape: (48, 4)
X_scaled = X_scaled.reshape(1, 48, 4)  # add batch dimension

# === PREDICT ===
y_pred = model.predict(X_scaled)
label = int(y_pred[0][0] > 0.5)

print("\n🧠 Prediction:")
print(f"Probability of unsteady gait: {y_pred[0][0]:.4f}")
print(f"Predicted label: {'Unsteady' if label == 1 else 'Steady'}")


import numpy as np
X = np.load("../data/new_input.npy")
print("👀 Feature stats:", np.mean(X, axis=0))
