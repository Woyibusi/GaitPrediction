import csv

# === CONFIGURATION ===
output_file = "labels.csv"
total_videos = 384
split_index = 201  # All videos up to this index are steady (0), rest are unsteady (1)

# === GENERATE LABELS ===
with open(output_file, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "label"])  # Header
    
    for i in range(1, total_videos + 1):
        filename = f"video{i:03d}.mp4"
        label = 0 if i <= split_index else 1
        writer.writerow([filename, label])

print("✅ labels.csv generated successfully.")
