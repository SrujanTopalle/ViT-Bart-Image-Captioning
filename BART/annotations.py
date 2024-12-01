import os
import json

# Paths
image_dir = r"C:\\Users\\Srujan Topalle\\Desktop\\Deep learning project\\coco2017\\train"  # Directory containing image files
annotations_file = "./captions_train2017.json"  # JSON file with annotations

# Step 1: Extract and sort image IDs from filenames
image_filenames = os.listdir(image_dir)
image_ids = sorted([int(filename.split('.')[0]) for filename in image_filenames if filename.endswith(".jpg")])

# Step 2: Select the first 2000 image IDs
selected_image_ids = set(image_ids[:2000])

# Step 3: Load annotations and filter based on selected IDs
with open(annotations_file, "r") as f:
    annotations = json.load(f)

selected_annotations = [
    ann for ann in annotations["annotations"] if ann["image_id"] in selected_image_ids
]

# Step 4: Save filtered annotations for reuse
filtered_annotations_file = "./filtered_annotations.json"
with open(filtered_annotations_file, "w") as f:
    json.dump({"annotations": selected_annotations}, f)

print(f"Filtered annotations saved to {filtered_annotations_file}")
