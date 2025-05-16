import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# === CONFIGURATION ===
INPUT_FILE = "../data/X.npy"   # Input from anatomical normalization
OUTPUT_FILE = "../data/X_scaled.npy"      # Output for LSTM training
SCALER_FILE = "../data/scaler.pkl"        # Saved StandardScaler model

# === LOAD DATA ===
print(f"📂 Loading {INPUT_FILE} ...")
X = np.load(INPUT_FILE)  # shape: (n_samples, 48, 4)

# === RESHAPE FOR SCALING ===
n_samples, seq_len, n_features = X.shape
X_reshaped = X.reshape(-1, n_features)  # shape: (n_samples * 48, 4)

# === FIT AND APPLY SCALER ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_reshaped)

# === RESHAPE BACK ===
X_scaled = X_scaled.reshape(n_samples, seq_len, n_features)

# === SAVE RESULTS ===
np.save(OUTPUT_FILE, X_scaled)
joblib.dump(scaler, SCALER_FILE)

print("\n✅ StandardScaler normalization complete.")
print(f"🔹 Saved normalized sequences to: {OUTPUT_FILE}")
print(f"🔹 Saved scaler to: {SCALER_FILE}")
print(f"🧪 Final shape: {X_scaled.shape}")
