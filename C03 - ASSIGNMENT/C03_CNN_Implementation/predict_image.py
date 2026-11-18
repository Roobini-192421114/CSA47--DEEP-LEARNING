import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from models import build_model

def main():
    parser = argparse.ArgumentParser(description="Predict the defect class for one image.")
    parser.add_argument("--model", choices=["custom", "mobilenetv2"], required=True)
    parser.add_argument("--image", required=True, help="Path to an image file.")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else (
        Path("outputs") / args.model / "best_model.pth"
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint["classes"]

    model = build_model(args.model, len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    confidence, class_id = torch.max(probabilities, dim=0)

    print("Predicted defect:", classes[class_id.item()])
    print("Confidence:", f"{confidence.item() * 100:.2f}%")
    print("\nClass probabilities:")
    for index in torch.argsort(probabilities, descending=True):
        print(f"{classes[index.item()]}: {probabilities[index].item() * 100:.2f}%")

if __name__ == "__main__":
    main()
