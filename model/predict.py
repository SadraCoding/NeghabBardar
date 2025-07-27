import argparse
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import torch
import os

def main():
    parser = argparse.ArgumentParser(description="Predict image class using fine-tuned model")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "./sdxl-detector-finetuned")
    model = AutoModelForImageClassification.from_pretrained(model_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]

    print(f"Predicted class index: {predicted_class_idx}")
    print(f"Predicted label: {predicted_label}")

if __name__ == "__main__":
    main()
