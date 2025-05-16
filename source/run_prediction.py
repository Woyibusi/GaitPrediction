import sys
import subprocess
import os
import cv2

# === Get video name from command-line argument ===
if len(sys.argv) < 2:
    print("❌ Usage: python run_prediction.py <video_filename>")
    sys.exit(1)

video_name = sys.argv[1]
video_path = os.path.join("../data/raw", video_name)
video_path_line = f'VIDEO_PATH = "../data/raw/{video_name}"\n'

# === Update extract_single_sequence.py dynamically ===
with open("extract_single_sequence.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

with open("extract_single_sequence.py", "w", encoding="utf-8") as f:
    for line in lines:
        if line.strip().startswith("VIDEO_PATH ="):
            f.write(video_path_line)
        else:
            f.write(line)

print(f"📂 Target video set to: {video_path}")

# === Step 1: Extract Sequence ===
print("🚶 Extracting sequence...")
result = subprocess.run(["python", "extract_single_sequence.py"])

if result.returncode != 0:
    print("❌ Sequence extraction failed. Aborting prediction.")
    sys.exit(1)

# === Step 2: Predict ===
print("🧠 Running prediction...")
subprocess.run(["python", "predict.py"])

# === Step 3: Show the video ===
print("🎬 Playing the video (press Q to quit)...")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"❌ Failed to open video: {video_path}")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("📹 Input Video", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
