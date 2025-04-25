from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Calculate class weights
labels = dataset["labels"]
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
print("Class Weights:", class_weights)

# Apply weights during training (e.g., in loss function)
loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))

from flux.tokenizer import FluxTokenizer
from torch.utils.data import DataLoader

# Load and tokenize dataset
tokenizer = FluxTokenizer.from_pretrained("flux1-dev")

def preprocess(text):
    # Tokenize with padding and truncation
    return tokenizer(text, padding='max_length', truncation=True, max_length=512)

# Define a custom dataset
class CustomDataset:
    def __init__(self, data, preprocess_func):
        self.data = data
        self.preprocess = preprocess_func

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        tokens = self.preprocess(text)
        return {"input_ids": tokens["input_ids"], "labels": label}

# Create DataLoader for training
dataset = CustomDataset(data, preprocess)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

from loralib import LoRA
from flux.models import FluxModel

# Step 1: Load the pre-trained model
print("Loading pre-trained Flux.1-dev model...")
model = FluxModel.from_pretrained("flux1-dev")

# Step 2: Add LoRA layers
print("Adding LoRA layers to target modules...")
lora_model = LoRA(
    model, 
    target_modules=["attention", "ffn"],  # Adapt attention and feed-forward layers
    rank=4  # Low-rank dimension
)

# Step 3: Freeze base model parameters
lora_model.freeze_base_model()
print("Base model parameters frozen. Only LoRA layers will be updated during training.")

# Optimizer and loss function
optimizer = torch.optim.AdamW(lora_model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
print("Starting training...")
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch in dataloader:
        # Forward pass
        outputs = lora_model(batch["input_ids"])
        loss = loss_fn(outputs, batch["labels"])
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")