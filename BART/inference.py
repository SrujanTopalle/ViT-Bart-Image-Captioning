import os
import json
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoProcessor
import torch
from PIL import Image
from tqdm import tqdm

pretrained_model_dir = "./checkpoints" 
images_folder = r"path/to/img/folder"  
output_json_path = "./generated_captions.json" 

model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_dir)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
processor = AutoProcessor.from_pretrained(model.config.encoder._name_or_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

model.eval()

def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
        outputs = model.generate(pixel_values, max_length=32, num_beams=4)
        caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

results = []
for image_file in tqdm(os.listdir(images_folder), desc="Processing Images"):
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(images_folder, image_file)
        caption = generate_caption(image_path)
        if caption:
            image_id = int(os.path.splitext(image_file)[0])  
            results.append({"image_id": image_id, "caption": caption})

with open(output_json_path, "w") as f:
    json.dump(results, f)

print(f"Captions generated and saved to {output_json_path}")
