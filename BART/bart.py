from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoProcessor
import torch
from PIL import Image

# Define model and image paths
pretrained_model_dir = "./checkpoints"  # Replace with your model directory or Hugging Face model name
image_path = r"C:\\Users\\Srujan Topalle\\Desktop\\Deep learning project\\coco2017\\test2017\\000000000001.jpg"  # Replace with your image file path

# Load the pre-trained VisionEncoderDecoderModel
model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_dir)

# Load the tokenizer and processor
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir)
processor = AutoProcessor.from_pretrained(model.config.encoder._name_or_path)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Preprocess the input image
image = Image.open(image_path).convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

# Perform inference
model.eval()
outputs = model.generate(pixel_values, max_length=32, num_beams=4)

# Decode the generated caption
caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Caption:", caption)
