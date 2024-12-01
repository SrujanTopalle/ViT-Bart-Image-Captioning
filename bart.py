from transformers import VisionEncoderDecoderModel, AutoProcessor, AutoTokenizer
import torch
from PIL import Image

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Vision Encoder
encoder_model = "google/vit-base-patch16-224-in21k"  # Vision Transformer
decoder_model = "facebook/bart-large"  # Replace with desired decoder (e.g., T5, mBART)

# Load the VisionEncoderDecoderModel with a custom decoder
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model, decoder_model)

# Update the configuration
model.config.decoder_start_token_id = model.config.pad_token_id
model.config.eos_token_id = model.config.pad_token_id
model.config.max_length = 16
model.config.num_beams = 4

# Move the model to the device
model.to(device)

# Load the tokenizer and processor
processor = AutoProcessor.from_pretrained(encoder_model)
tokenizer = AutoTokenizer.from_pretrained(decoder_model)

# Load and preprocess an image
image_path = r"C:\\Users\\Srujan Topalle\\Desktop\\Deep learning project\\test2017\\000000000180.jpg"
image = Image.open(image_path)
if image.mode != "RGB":
    image = image.convert(mode="RGB")

# Preprocess the image
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

# Generate captions
output_ids = model.generate(pixel_values)
caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("Generated Caption:", caption)
