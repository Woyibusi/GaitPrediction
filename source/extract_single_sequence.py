import cv2
import numpy as np
import joblib
import mediapipe as mp
import os
import sys

# === CONFIGURATION ===
VIDEO_PATH = "../data/raw/video222.mp4"
SCALER_PATH = "../data/scaler.pkl"
OUTPUT_PATH = "../data/new_input.npy"
SEQUENCE_LENGTH = 48

# === INIT MEDIAPIPE ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

def sample_frames_evenly(frames, target_len=SEQUENCE_LENGTH):
    total = len(frames)
    indices = np.linspace(0, total - 1, target_len).astype(int)
    return [frames[i] for i in indices]

def pad_sequence_to_length(frames, target_len=SEQUENCE_LENGTH):
    while len(frames) < target_len:
        frames.append(frames[-1])
    return frames

# === READ VIDEO ===
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"❌ Failed to open video: {VIDEO_PATH}")
    sys.exit(1)

frames = []
valid_frames = 0
total_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        try:
            # Required landmarks
            la = lm[27]  # Left ankle
            ra = lm[28]  # Right ankle
            lh = lm[23]  # Left hip
            rh = lm[24]  # Right hip
            ls = lm[11]  # Left shoulder
            rs = lm[12]  # Right shoulder

            # Compute body reference
            hip_x = (lh.x + rh.x) / 2
            hip_y = (lh.y + rh.y) / 2
            shoulder_y = (ls.y + rs.y) / 2
            d = max(abs(hip_y - shoulder_y), 1e-5)

            # Anatomical normalization
            LA_xn = abs((la.x - hip_x) / d) * 100
            LA_yn = abs((la.y - hip_y) / d) * 100
            RA_xn = abs((ra.x - hip_x) / d) * 100
            RA_yn = abs((ra.y - hip_y) / d) * 100

            frame_data = [LA_xn, LA_yn, RA_xn, RA_yn]
            valid_frames += 1

        except Exception as e:
            print(f"⚠️ Landmark extraction failed for a frame: {e}")
            frame_data = [0.0, 0.0, 0.0, 0.0]
    else:
        frame_data = [0.0, 0.0, 0.0, 0.0]

    frames.append(frame_data)

cap.release()

print(f"\n📊 Total frames read: {total_frames}")
print(f"✅ Frames with valid pose: {valid_frames}")

# === HANDLE EMPTY VIDEO ===
if len(frames) == 0:
    print(f"❌ No frames extracted from video: {VIDEO_PATH}")
    sys.exit(1)

# === NORMALIZE TO 48 FRAMES ===
if len(frames) >= SEQUENCE_LENGTH:
    sequence = sample_frames_evenly(frames, SEQUENCE_LENGTH)
else:
    sequence = pad_sequence_to_length(frames, SEQUENCE_LENGTH)

X_seq = np.array(sequence)  # shape (48, 4)

# === APPLY SCALER ===
if not os.path.exists(SCALER_PATH):
    print(f"❌ Scaler file not found: {SCALER_PATH}")
    sys.exit(1)

scaler = joblib.load(SCALER_PATH)
X_scaled = scaler.transform(X_seq)

# === SAVE FOR PREDICTION ===
np.save(OUTPUT_PATH, X_scaled)
print(f"✅ Sequence saved to: {OUTPUT_PATH}")
