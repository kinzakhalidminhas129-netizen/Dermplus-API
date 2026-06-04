import os
import pickle
import torch
import torch.nn as nn
import numpy as np
import io
import cv2
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.applications.efficientnet
import preprocess_input as effnet_preprocess
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights

app = Flask(__name__)

# --- MODEL REGISTRY & CONFIG ---
# Merged registry maintaining Code 1 structure with Code 2's specific logic for target diseases
MODEL_REGISTRY = {
    'fphl': {'model_path': 'models/female_pattern_hairloss/fphl model.pkl', 'type': 'tf'},
    'alopecia': {'model_path': 'models/alopecia_areata/alopecia model.pkl', 'type': 'tf'},
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
extractors = {"resnet": None}

def get_extractor(type):
    if extractors[type] is None:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.fc = nn.Identity()
        model.eval()
        extractors[type] = model
    return extractors[type]

def get_model(disease):
    if disease not in loaded_models:
        with open(MODEL_REGISTRY[disease]['model_path'], 'rb') as f:
            loaded_models[disease] = pickle.load(f)
    return loaded_models[disease]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'disease' not in request.form:
        return jsonify({"success": False, "error": "Please provide both 'image' and 'disease'"}), 400

    disease = request.form['disease'].lower()
    file = request.files['image']

    if disease not in MODEL_REGISTRY:
        return jsonify({"success": False, "error": "Invalid disease type"}), 400

    try:
        config = MODEL_REGISTRY[disease]
        model = get_model(disease)
        
        # Handle different preprocessing flows
        if config['type'] == 'tf':
            img = Image.open(io.BytesIO(file.read())).convert('RGB')
            processed_img = effnet_preprocess(np.array(img.resize((224, 224))))
            processed_img = np.expand_dims(processed_img, axis=0)
            prediction = model.predict(processed_img)
            raw_id = int(np.argmax(prediction, axis=1)[0])
            labels = {0: "Detected", 1: "Healthy"}
            
        elif config['type'] == 'torch_extractor':
            image_data = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            tensor_img = transform(img_rgb).unsqueeze(0)
            with torch.no_grad():
                features = get_extractor(config['extractor_type'])(tensor_img).numpy()
            raw_id = int(model.predict(features)[0])
            labels = {0: f"{disease.replace('_', ' ').title()} Detected", 1: "Healthy"}
            
        elif config['type'] == 'traditional':
            image_data = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
            img_res = cv2.resize(img, (64, 64))
            processed = (img_res / 255.0).flatten().reshape(1, -1)
            raw_id = int(model.predict(processed)[0])
            labels = {0: "Healthy (Benign)", 1: "Melanoma Detected (Malignant)"}

        return jsonify({"success": True, "prediction": labels.get(raw_id, "Unknown"), "raw_id": raw_id})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)