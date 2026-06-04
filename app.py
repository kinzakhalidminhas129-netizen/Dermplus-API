import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import io
import cv2
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.applications.efficientnet import preprocess_input as effnet_preprocess
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

app = Flask(__name__)

# --- MODEL REGISTRY ---
MODEL_REGISTRY = {
    'fphl': {
        'model_path': 'models/female_pattern_hairloss/fphl model.pkl',
        'type': 'tf'
    },
    'alopecia': {
        'model_path': 'models/alopecia_areata/alopecia model.pkl',
        'type': 'tf'
    },
    'psoriasis': {
        'model_path': 'models/psoriasis/psoriasis model.pkl',
        'type': 'torch_extractor',
        'extractor_type': 'resnet'
    },
    'melanoma': {
        'model_path': 'models/melanoma/melanoma model.pkl',
        'type': 'traditional'
    },
    'basal cell carcinoma': {
        'model_path': 'models/basal_cell_carcinoma/bcc model.pkl',
        'type': 'torch_extractor',
        'extractor_type': 'resnet'
    }
}

loaded_models = {}
extractors = {}

# --- LOAD MODEL ---
def get_model(disease):
    if disease not in loaded_models:
        path = MODEL_REGISTRY[disease]['model_path']
        with open(path, 'rb') as f:
            loaded_models[disease] = pickle.load(f)
    return loaded_models[disease]

# --- RESNET EXTRACTOR (LIGHTWEIGHT SAFE) ---
def get_extractor(name):
    if name not in extractors:
        model = resnet18(weights=None)   # ⚠️ safe for Railway
        model.fc = nn.Identity()
        model.eval()
        extractors[name] = model
    return extractors[name]

# --- ROUTE ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files or 'disease' not in request.form:
            return jsonify({"success": False, "error": "Send image + disease"}), 400

        disease = request.form['disease'].lower()

        if disease not in MODEL_REGISTRY:
            return jsonify({"success": False, "error": "Invalid disease"}), 400

        config = MODEL_REGISTRY[disease]
        model = get_model(disease)

        # IMPORTANT: read ONCE
        file_bytes = request.files['image'].read()

        # =========================
        # TF MODELS (EfficientNet)
        # =========================
        if config['type'] == 'tf':
            img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
            img = img.resize((224, 224))

            img_array = np.array(img)
            img_array = effnet_preprocess(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array)
            raw_id = int(np.argmax(prediction, axis=1)[0])

            labels = {0: "Detected", 1: "Healthy"}

        # =========================
        # TORCH FEATURE EXTRACTOR
        # =========================
        elif config['type'] == 'torch_extractor':
            img_np = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({"success": False, "error": "Invalid image"}), 400

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            tensor_img = transform(img_rgb).unsqueeze(0)

            with torch.no_grad():
                extractor = get_extractor(config['extractor_type'])
                features = extractor(tensor_img).numpy()

            raw_id = int(model.predict(features)[0])

            labels = {
                0: f"{disease.title()} Detected",
                1: "Healthy"
            }

        # =========================
        # TRADITIONAL MODELS
        # =========================
        elif config['type'] == 'traditional':
            img_np = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            if img is None:
                return jsonify({"success": False, "error": "Invalid image"}), 400

            img = cv2.resize(img, (64, 64))
            processed = (img / 255.0).flatten().reshape(1, -1)

            raw_id = int(model.predict(processed)[0])

            labels = {
                0: "Healthy (Benign)",
                1: "Melanoma Detected (Malignant)"
            }

        return jsonify({
            "success": True,
            "prediction": labels.get(raw_id, "Unknown"),
            "raw_id": raw_id
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# =========================
# RAILWAY ENTRY POINT FIX
# =========================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)