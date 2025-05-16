import cv2
import mediapipe as mp

# === CONFIGURATION ===
video_path = "data/raw/video300.mp4"  # Change this to test any video

# === INITIALIZE MEDIAPIPE POSE ===
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    model_complexity=1)

# === OPEN VIDEO ===
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Failed to open video: {video_path}")
    exit()

print("🎥 Starting pose detection...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw pose landmarks
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

    # Show output
    cv2.imshow('Pose Detection (Press Q to quit)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Done.")
