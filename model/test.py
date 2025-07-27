from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from datasets import Dataset, Image
from sklearn.metrics import accuracy_score
import os
import torch

model_dir = "./sdxl-detector-finetuned"
model = AutoModelForImageClassification.from_pretrained(model_dir)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def list_test_images_and_labels(test_root_dir):
    classes = sorted(os.listdir(test_root_dir))
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    images, labels = [], []
    for cls in classes:
        cls_dir = os.path.join(test_root_dir, cls)
        for filename in os.listdir(cls_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                images.append(os.path.join(cls_dir, filename))
                labels.append(class_to_idx[cls])
    return images, labels, classes

test_dir = "./dataset/test"
test_images, test_labels, class_names = list_test_images_and_labels(test_dir)
test_dataset = Dataset.from_dict({"image": test_images, "label": test_labels}).cast_column("image", Image())

def preprocess_function(examples):
    inputs = feature_extractor(examples["image"], return_tensors="pt")
    return inputs

test_dataset = test_dataset.map(preprocess_function, batched=True)
test_dataset.set_format(type="torch", columns=["pixel_values", "label"])

all_preds = []
all_labels = []
for example in test_dataset:
    pixel_values = example['pixel_values'].unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(pixel_values)
    pred = outputs.logits.argmax(-1).cpu().item()
    all_preds.append(pred)
    all_labels.append(example['label'])

acc = accuracy_score(all_labels, all_preds)
print(f"Accuracy on test set: {acc:.4f}")


for i in range(min(5, len(test_images))):
    print(f"File: {test_images[i]}  --> Predicted: {class_names[all_preds[i]]}    Actual: {class_names[all_labels[i]]}")
