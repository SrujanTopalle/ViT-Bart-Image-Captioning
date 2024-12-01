import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, VisionEncoderDecoderModel
from PIL import Image
import numpy as np

# Define encoder and processor
encoder_model = "google/vit-base-patch16-224-in21k"
processor = AutoProcessor.from_pretrained(encoder_model)
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model, "facebook/bart-large")

# Freeze encoder weights
for param in model.encoder.parameters():
    param.requires_grad = False

model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Define dataset
class ImageDataset(Dataset):
    def __init__(self, image_dir, processor):
        self.image_dir = image_dir
        self.processor = processor
        self.image_files = os.listdir(image_dir)
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(image_path)

        # Convert image to RGB if necessary
        if image.mode != "RGB":
            image = image.convert(mode="RGB")

        # Preprocess image
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        return {"image_name": self.image_files[idx], "pixel_values": pixel_values}

# Paths
image_dir = r"C:\\Users\\Srujan Topalle\\Desktop\\Deep learning project\\coco2017\\train"
output_dir = r"C:\\Users\\Srujan Topalle\\Desktop\\Deep learning project\\deep\\encodings"

# Create dataset and dataloader
dataset = ImageDataset(image_dir, processor)
dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

# Save encodings
os.makedirs(output_dir, exist_ok=True)
with torch.no_grad():
    for batch in dataloader:
        pixel_values = torch.stack([item for item in batch["pixel_values"]]).to("cuda")
        outputs = model.encoder(pixel_values=pixel_values)
        encodings = outputs.last_hidden_state.cpu().numpy()

        # Save each encoding
        for i, encoding in enumerate(encodings):
            image_name = batch["image_name"][i]
            np.save(os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.npy"), encoding)
