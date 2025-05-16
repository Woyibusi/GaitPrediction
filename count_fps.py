import cv2
import os

video_path = "data/raw/video300.mp4"  # Change this

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / fps if fps else 0

print(f"Filename: {os.path.basename(video_path)}")
print(f"FPS: {fps}")
print(f"Total Frames: {total_frames}")
print(f"Duration (s): {duration_sec:.2f}")

cap.release()


import os
import cv2

# === CONFIG ===
video_dir = "data/raw"  # Replace this path

fps_list = []

for filename in sorted(os.listdir(video_dir)):
    if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue

    path = os.path.join(video_dir, filename)
    cap = cv2.VideoCapture(path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0

    cap.release()

    fps_list.append((filename, fps, total_frames, duration))

# === Analysis ===
fps_values = [f[1] for f in fps_list]
min_fps = min(fps_values)
max_fps = max(fps_values)
avg_fps = sum(fps_values) / len(fps_values)

min_video = next(f for f in fps_list if f[1] == min_fps)
max_video = next(f for f in fps_list if f[1] == max_fps)

print(f"\n🔍 Total Videos Checked: {len(fps_list)}")
print(f"📉 Lowest FPS: {min_fps:.2f}  → {min_video[0]}")
print(f"📈 Highest FPS: {max_fps:.2f} → {max_video[0]}")
print(f"📊 Average FPS: {avg_fps:.2f}")

print("\n🎬 Sample Summary:")
print("Filename\t\tFPS\tFrames\tDuration(s)")
for f in fps_list[:5]:  # show first 5
    print(f"{f[0]}\t{f[1]:.2f}\t{f[2]}\t{f[3]:.2f}")
