import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTFeatureExtractor

MODEL_PATH = "../models/food101_vit_base_patch16_224.pth"

# Load feature extractor and model
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224", num_labels=101)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Transform image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(inputs).logits
        predicted_class = logits.argmax(-1).item()
    print(f"Predicted class ID: {predicted_class}")

# Example usage:
# predict("../data/images/pizza.jpg")
