import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # current script directory
DATA_DIR = os.path.join(BASE_DIR, "../data/food-101")  # adjust as needed
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "../models/food101_vit_base_patch16_224.pth")

# === Check directories ===
if not os.path.exists(os.path.join(DATA_DIR, "train")):
    raise FileNotFoundError(f"❌ Training folder not found at: {os.path.join(DATA_DIR, 'train')}\n"
                            "👉 Make sure your dataset is extracted here, e.g.:\n"
                            "Allergen_detector/data/food-101/train/...")

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# === Feature extractor / image processor ===
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# === Datasets & loaders ===
train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "train"), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# === Model ===
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(train_dataset.classes),
    ignore_mismatched_sizes=True  # 👈 This fixes the size mismatch
)

model.to(device)

# === Training setup ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# === Training loop ===
epochs = 2  # Try small first
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"📉 Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")

# === Save model ===
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")
