from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

# Load the fine-tuned model and associated processor and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load and preprocess the image
image = Image.open(r"C:\\Users\\username\\Desktop\\Deep learning project\\test2017\\000000000180.jpg")
if image.mode != "RGB":
    image = image.convert(mode="RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

# Generate the caption
output_ids = model.generate(pixel_values, max_length=16, num_beams=4)
caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("Generated Caption:", caption)
