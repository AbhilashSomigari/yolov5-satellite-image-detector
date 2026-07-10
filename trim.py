import os

# Path to your folder
folder_path = "data/pred_boxes"

for filename in os.listdir(folder_path):
    if ".gradcam" in filename:
        old_path = os.path.join(folder_path, filename)
        new_filename = filename.replace(".gradcam", "")
        new_path = os.path.join(folder_path, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_filename}")

print("✅ Done! All '.gradcam' removed from filenames.")
