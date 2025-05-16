import os

# === CONFIGURATION ===
video_dir = "unsteady2/Sensory Ataxia (Stomping Gait)"  # ← Replace with the actual path
prefix = "video"
start_index = 365  # Start numbering from here

# === FETCH AND SORT FILES ===
video_files = sorted([
    f for f in os.listdir(video_dir)
    if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))  # Add more extensions if needed
])

# === RENAME LOOP ===
for i, filename in enumerate(video_files, start=start_index):
    ext = os.path.splitext(filename)[1]  # keep original extension
    new_name = f"{prefix}{i:03d}{ext}"   # zero-padded (e.g., video001.mp4)
    
    old_path = os.path.join(video_dir, filename)
    new_path = os.path.join(video_dir, new_name)
    
    os.rename(old_path, new_path)
    print(f"Renamed: {filename} → {new_name}")

print("\n✅ All videos renamed successfully.")
