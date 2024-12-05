import os
import json

def filter_json_by_image_names(json_file, txt_file, output_json_file):
    with open(txt_file, "r") as f:
        valid_image_names = set(line.strip() for line in f.readlines())

    valid_image_ids = {int(name.split('.')[0]) for name in valid_image_names}

    with open(json_file, "r") as f:
        original_data = json.load(f)

    filtered_annotations = [
        entry for entry in original_data["annotations"]
        if entry["image_id"] in valid_image_ids
    ]

    filtered_data = {"annotations": filtered_annotations}
    with open(output_json_file, "w") as f:
        json.dump(filtered_data, f, indent=4)

    print(f"Filtered JSON saved to {output_json_file}")

json_file = "./captions_train2017.json" 
txt_file = "./image_names.txt"          
output_json_file = "./filtered_annotations.json"  
filter_json_by_image_names(json_file, txt_file, output_json_file)
