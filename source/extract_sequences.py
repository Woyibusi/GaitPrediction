import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

# === CONFIGURATION ===
VIDEO_DIR = "../data/raw"              # Folder with videos
LABELS_FILE = "../data/labels.csv"     # filename,label
SEQUENCE_LENGTH = 48           # Frames per video
OUTPUT_X = "X.npy"
OUTPUT_Y = "y.npy"

# === INITIALIZE MEDIAPIPE ===
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# === LOAD LABELS ===
df_labels = pd.read_csv(LABELS_FILE)

X_data = []
y_data = []

# === FRAME SAMPLING ===
def sample_frames_evenly(frames, target_len=SEQUENCE_LENGTH):
    total = len(frames)
    indices = np.linspace(0, total - 1, target_len).astype(int)
    return [frames[i] for i in indices]

def pad_sequence_to_length(frames, target_len=SEQUENCE_LENGTH):
    while len(frames) < target_len:
        frames.append(frames[-1])
    return frames

# === PROCESS EACH VIDEO ===
for _, row in tqdm(df_labels.iterrows(), total=len(df_labels), desc="Processing videos"):
    filename = row["filename"]
    label = int(row["label"])
    video_path = os.path.join(VIDEO_DIR, filename)

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Ankle landmarks
            la = lm[27]  # Left Ankle
            ra = lm[28]  # Right Ankle

            # Shoulder and hip landmarks
            try:
                ls = lm[11]  # Left Shoulder
                rs = lm[12]  # Right Shoulder
                lh = lm[23]  # Left Hip
                rh = lm[24]  # Right Hip

                # Compute hip and shoulder centers
                hip_x = (lh.x + rh.x) / 2
                hip_y = (lh.y + rh.y) / 2
                shoulder_y = (ls.y + rs.y) / 2

                # Reference distance
                d = abs(hip_y - shoulder_y)
                d = max(d, 1e-5)  # Avoid division by zero

                # Anatomical normalization
                LA_xn = abs((la.x - hip_x) / d) * 100
                LA_yn = abs((la.y - hip_y) / d) * 100
                RA_xn = abs((ra.x - hip_x) / d) * 100
                RA_yn = abs((ra.y - hip_y) / d) * 100

                frame_data = [LA_xn, LA_yn, RA_xn, RA_yn]

            except:
                # Fallback if landmarks are missing
                frame_data = [0.0, 0.0, 0.0, 0.0]

        else:
            frame_data = [0.0, 0.0, 0.0, 0.0]

        frames.append(frame_data)

    cap.release()

    # Normalize frame count
    if len(frames) >= SEQUENCE_LENGTH:
        sequence = sample_frames_evenly(frames, SEQUENCE_LENGTH)
    elif len(frames) > 0:
        sequence = pad_sequence_to_length(frames, SEQUENCE_LENGTH)
    else:
        print(f"⚠️ Skipping empty video: {filename}")
        continue

    X_data.append(sequence)
    y_data.append(label)

# === CONVERT TO NUMPY AND SAVE ===
X = np.array(X_data)  # shape: (num_videos, 48, 4)
y = np.array(y_data)  # shape: (num_videos,)

np.save(OUTPUT_X, X)
np.save(OUTPUT_Y, y)

print("\n✅ Anatomical normalization complete.")
print(f"Saved X to {OUTPUT_X} with shape {X.shape}")
print(f"Saved y to {OUTPUT_Y} with shape {y.shape}")
