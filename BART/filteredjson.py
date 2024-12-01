import os
import json

def filter_json_by_image_names(json_file, txt_file, output_json_file):
    # Load image names from the text file
    with open(txt_file, "r") as f:
        valid_image_names = set(line.strip() for line in f.readlines())

    # Convert image names to image IDs
    valid_image_ids = {int(name.split('.')[0]) for name in valid_image_names}

    # Load the original JSON file
    with open(json_file, "r") as f:
        original_data = json.load(f)

    # Filter the JSON data
    filtered_annotations = [
        entry for entry in original_data["annotations"]
        if entry["image_id"] in valid_image_ids
    ]

    # Save the filtered JSON
    filtered_data = {"annotations": filtered_annotations}
    with open(output_json_file, "w") as f:
        json.dump(filtered_data, f, indent=4)

    print(f"Filtered JSON saved to {output_json_file}")

# Example usage
json_file = "./captions_train2017.json"  # Original JSON file
txt_file = "./image_names.txt"          # Text file with image names
output_json_file = "./filtered_annotations.json"  # Filtered output JSON
filter_json_by_image_names(json_file, txt_file, output_json_file)
