from flask import Flask, request, jsonify
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import torch
import os
import tempfile

model_path = "../model/sdxl-detector-finetuned"
model = AutoModelForImageClassification.from_pretrained(model_path)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "فایلی ارسال نشده"}), 400

    file = request.files["file"]

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
            file.save(temp.name)
            image = Image.open(temp.name).convert("RGB")
            inputs = feature_extractor(images=image, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()

            if predicted_class_idx == 0:
                prediction_text = "ساخته شده توسط هوش مصنوعی"
            elif predicted_class_idx == 1:
                prediction_text = "چهره واقعی"
            else:
                prediction_text = f"کلاس ناشناخته: {predicted_class_idx}"

        os.remove(temp.name)
        return jsonify({"prediction": prediction_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8000)
