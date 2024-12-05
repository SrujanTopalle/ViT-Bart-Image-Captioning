from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoProcessor
import torch
from PIL import Image

pretrained_model_dir = "./checkpoints"
image_path = "path/to/image" 

model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_dir)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
processor = AutoProcessor.from_pretrained(model.config.encoder._name_or_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

image = Image.open(image_path).convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

model.eval()
outputs = model.generate(pixel_values, max_length=32, num_beams=4)

caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Caption:", caption)
