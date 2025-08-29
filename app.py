# app.py
from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import base64
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# main.py se model classes import karein
from main import EnhancedCNN, TransferLearningModel

app = Flask(__name__, template_folder='templates', static_folder='static')

# ================================================================
# SECTION: Configuration and Global Model Loading
# ================================================================
MODEL_PATH = "models/best_model_fold_4.pth"
IMAGE_SIZE = 224
MODEL_TYPE = "EnhancedCNN" 

# Yahan apne model ke sahi parameters dalein
if MODEL_TYPE == "EnhancedCNN":
    MODEL_PARAMS = {
        "num_layers": 3,
        "filter_size": 3,
        "num_filters": 64,
        "dropout_rate": 0.2,
        "num_classes": 2
    }
elif MODEL_TYPE == "TransferLearningModel":
    MODEL_PARAMS = {
        "base_model_name": "resnet18",
        "num_classes": 2,
        "dropout_rate": 0.5
    }
else:
    MODEL_PARAMS = {}

device = torch.device("cpu")
model = None
if os.path.exists(MODEL_PATH):
    try:
        if MODEL_TYPE == "EnhancedCNN":
            model = EnhancedCNN(**MODEL_PARAMS)
        elif MODEL_TYPE == "TransferLearningModel":
            model = TransferLearningModel(**MODEL_PARAMS)
        else:
            raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None
else:
    print(f"Error: Model file not found at {MODEL_PATH}")

# ================================================================
# SECTION: Helper Functions
# ================================================================
def get_test_transforms(image_size):
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def predict_image(model, image_bytes, transforms, device):
    try:
        original_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        print(f"Error: Could not load image from bytes: {e}")
        return None, None, None, None

    image_np = np.array(original_image)
    transformed_image = transforms(image=image_np)['image']
    image_tensor = transformed_image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_class = torch.max(outputs, 1)

    return predicted_class.item(), probabilities.cpu().numpy()[0], image_tensor, original_image

def generate_grad_cam(model, target_layer, input_tensor, original_image, predicted_class):
    try:
        cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
        targets = [ClassifierOutputTarget(predicted_class)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        resized_grayscale_cam = cv2.resize(grayscale_cam, (original_image.width, original_image.height))
        rgb_img = np.float32(original_image) / 255
        visualization = show_cam_on_image(rgb_img, resized_grayscale_cam, use_rgb=True)
        visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
        is_success, buffer = cv2.imencode(".jpg", visualization)
        if is_success:
            return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Error generating Grad-CAM: {e}")
        return None

# ================================================================
# SECTION: Flask Routes
# ================================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about_us')
def about_us():
    team_members = [
        {"name": "Team Member 1", "role": "Project Lead", "image": "team_member_1.jpg", "description": "Ek behtareen lider jo har challenge ko solution mein badal deta hai."},
        {"name": "Team Member 2", "role": "Data Scientist", "image": "team_member_2.jpg", "description": "Data analysis aur model training mein mahir. Inki wajha se accuracy high hai."},
        {"name": "Team Member 3", "role": "Backend Developer", "image": "team_member_3.jpg", "description": "Poore server aur backend logic ko sambhalne wale mastermind."},
        {"name": "Team Member 4", "role": "Frontend Developer", "image": "team_member_4.jpg", "description": "Website ki khoobsurti aur user experience inki zimmedari hai."},
        {"name": "Team Member 5", "role": "Researcher", "image": "team_member_5.jpg", "description": "Naye algorithms aur technology ki khoj mein lagatar kaam karte hain."}
    ]
    return render_template('about_us.html', members=team_members)

@app.route('/plots')
def plots():
    # Yahan hum dummy data use kar rahe hain
    # Aap isko apne actual results se replace kar sakte hain
    accuracy_data = [
        {"Fold": 1, "Accuracy": 92.5, "Sensitivity": 91.0, "Specificity": 94.0},
        {"Fold": 2, "Accuracy": 93.1, "Sensitivity": 92.5, "Specificity": 93.7},
        {"Fold": 3, "Accuracy": 91.8, "Sensitivity": 90.1, "Specificity": 93.5},
        {"Fold": 4, "Accuracy": 94.2, "Sensitivity": 95.0, "Specificity": 93.4},
        {"Fold": 5, "Accuracy": 92.9, "Sensitivity": 91.8, "Specificity": 94.0},
        {"Fold": "Average", "Accuracy": 92.9, "Sensitivity": 92.1, "Specificity": 93.7}
    ]
    
    plots_list = [
        {"title": "ROC Curve", "image_path": "/static/images/roc_curve.png", "description": "Model ki ROC curve, jo uski performance dikhati hai."},
        {"title": "Confusion Matrix", "image_path": "/static/images/confusion_matrix.png", "description": "Model ke predictions aur actual labels ka comparison."}
    ]
    
    return render_template('plots.html', accuracy_data=accuracy_data, plots_list=plots_list)

@app.route('/technology')
def technology():
    return render_template('technology.html')

@app.route('/future')
def future():
    return render_template('future.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model loaded nahi hua. Server logs check karein.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'Koi image file nahi mili'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Koi file select nahi ki'}), 400
    
    if file:
        try:
            image_bytes = file.read()
            transforms = get_test_transforms(IMAGE_SIZE)
            predicted_label, probabilities, image_tensor, original_image = predict_image(model, image_bytes, transforms, device)
            
            if predicted_label is not None:
                class_labels = {0: "Non-Glaucoma", 1: "Glaucoma"}
                predicted_class_name = class_labels.get(predicted_label, "Unknown")
                confidence = probabilities[predicted_label]

                if MODEL_TYPE == "EnhancedCNN":
                    target_layer = model.features[-1]
                elif MODEL_TYPE == "TransferLearningModel":
                    target_layer = model.base_model.layer4[-1]
                else:
                    target_layer = None

                grad_cam_image_base64 = None
                if target_layer:
                    grad_cam_image_base64 = generate_grad_cam(model, target_layer, image_tensor, original_image, predicted_label)

                return jsonify({
                    'prediction': predicted_class_name,
                    'confidence': float(confidence),
                    'grad_cam_image': grad_cam_image_base64
                })
            else:
                return jsonify({'error': 'Image process karne mein nakamyab.'}), 500

        except Exception as e:
            print(f"Prediction ke dauran ek error aaya: {e}")
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
