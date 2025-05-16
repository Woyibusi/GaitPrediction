import sys
import subprocess
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# === CONFIG ===
MODEL_PATH = "best_lstm_model.h5"
SCALER_PATH = "../data/scaler.pkl"
VIDEO_DIR = "../data/raw"
SEQUENCE_FILE = "../data/new_input.npy"

# === Get video name from command-line argument ===
if len(sys.argv) < 2:
    print("❌ Usage: python run_prediction.py <video_filename>")
    sys.exit(1)

video_name = sys.argv[1]
video_path = os.path.join(VIDEO_DIR, video_name)
video_path_posix = video_path.replace("\\", "/")
video_path_line = f'VIDEO_PATH = "{video_path_posix}"\n'

# === Step 1: Inject video path into extract_single_sequence.py ===
with open("../source/extract_single_sequence.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

with open("../source/extract_single_sequence.py", "w", encoding="utf-8") as f:
    for line in lines:
        if line.strip().startswith("VIDEO_PATH ="):
            f.write(video_path_line)
        else:
            f.write(line)

# === Step 2: Run extraction script ===
print(f"\n🚶 Extracting features from {video_name}...")
result = subprocess.run(["python", "../source/extract_single_sequence.py"])

if result.returncode != 0:
    print("❌ Extraction failed. Cannot proceed to prediction.")
    sys.exit(1)

# === Step 3: Load model & scaler ===
print("\n📂 Loading model and scaler...")
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# === Step 4: Load and reshape input sequence ===
X = np.load(SEQUENCE_FILE)  # shape: (48, 4)
X = X.reshape(1, 48, 4)     # Add batch dimension

# === Step 5: Predict ===
y_pred = model.predict(X)[0][0]
label = "Unsteady" if y_pred > 0.5 else "Steady"

print(f"\n🧠 Prediction result:")
print(f"Probability of unsteady gait: {y_pred:.4f}")
print(f"Predicted class: {label}")

# === Step 6: Play the video ===
print("\n🎬 Playing the video (press Q to quit)...")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Failed to open video: {video_path}")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("🎥 Input Video", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
