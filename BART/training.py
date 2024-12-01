import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, AutoTokenizer
import numpy as np

# Define paths
encodings_dir = os.path.join(".", "encodings")  # Directory with .npy files for encodings
annotations_file = os.path.join(".", "filtered_annotations.json")  # JSON file with annotations

# Define Dataset
class MSCOCOEncodedDataset(Dataset):
    def __init__(self, encodings_dir, annotations_file, tokenizer, max_length=32):
        self.encodings_dir = encodings_dir
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load annotations
        with open(annotations_file, "r") as f:
            annotations = json.load(f)

        # Group captions by image_id
        self.image_id_to_captions = {}
        for ann in annotations["annotations"]:
            image_id = ann["image_id"]
            if image_id not in self.image_id_to_captions:
                self.image_id_to_captions[image_id] = []
            self.image_id_to_captions[image_id].append(ann["caption"])

        # Convert grouped annotations to a list of image_ids
        self.image_ids = list(self.image_id_to_captions.keys())
    
    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # Get the image_id and its captions
        image_id = self.image_ids[idx]
        captions = self.image_id_to_captions[image_id]

        # Select a random caption for this image
        caption = captions[np.random.randint(len(captions))]

        # Convert image_id to filename
        image_filename = f"{image_id:012d}.npy"  # Zero-pad image_id and add .npy extension
        encoding_path = os.path.join(self.encodings_dir, image_filename)

        # Load the precomputed encoding
        encoding = torch.tensor(np.load(encoding_path))

        # Tokenize the caption
        labels = self.tokenizer(caption, padding="max_length", truncation=True, max_length=self.max_length).input_ids

        return {"encoding": encoding, "labels": torch.tensor(labels)}


# Define tokenizer for the decoder
encoder_model_name = "google/vit-base-patch16-224-in21k"
decoder_model_name = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)

# Initialize Dataset and DataLoader
dataset = MSCOCOEncodedDataset(encodings_dir, annotations_file, tokenizer)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)


# Load VisionEncoderDecoderModel
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encoder_model_name, decoder_model_name)

model.config.decoder_start_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id
# Freeze encoder (not used but part of the architecture)
for param in model.encoder.parameters():
    param.requires_grad = False

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer
optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=5e-5)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        encodings = batch["encoding"].to(device)  # Use precomputed encodings
        labels = batch["labels"].to(device)

        # Forward pass (provide encodings as `encoder_hidden_states`)
        outputs = model(labels=labels, encoder_outputs=(encodings,))
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")

# Save the fine-tuned decoder
model.save_pretrained("./checkpoints")
tokenizer.save_pretrained("./checkpoints")

print("Decoder training complete!")
