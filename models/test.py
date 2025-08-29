# predict_unseen.py
import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image 
from main import EnhancedCNN, TransferLearningModel # Import model classes from main.py

def get_test_transforms(image_size):
    """
    Defines the transformations for testing.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def load_model(model_path, model_type, model_params, device):
    """
    Loads a saved model.
    """
    if model_type == "EnhancedCNN":
        model = EnhancedCNN(**model_params)
    elif model_type == "TransferLearningModel":
        model = TransferLearningModel(**model_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_image(model, image_path, transforms, device):
    """
    Makes a prediction on a single image and returns the result along with the preprocessed image.
    """
    try:
        original_image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found: {image_path}")
        return None, None, None
    except Exception as e:
        print(f"Error: Could not load image: {e}")
        return None, None, None

    # Albumentations works with NumPy arrays, so convert the PIL Image
    image_np = np.array(original_image)
    transformed_image = transforms(image=image_np)['image']
    
    # Add a batch dimension for the model
    image_tensor = transformed_image.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_class = torch.max(outputs, 1)

    return predicted_class.item(), probabilities.cpu().numpy()[0], image_tensor, original_image

def generate_grad_cam(model, target_layer, input_tensor, original_image, predicted_class, output_path):
    """
    Generates and saves a Grad-CAM heatmap for the given input.
    """
    # Removed the 'use_cuda' argument as it is no longer supported in recent versions of the library.
    # The device is now inferred from the input tensor.
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    
    # The target is the predicted class
    targets = [ClassifierOutputTarget(predicted_class)]
    
    # Generate the heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    
    # Resize the heatmap to the original image dimensions before visualization
    resized_grayscale_cam = cv2.resize(grayscale_cam, (original_image.width, original_image.height))

    # Convert original image to a NumPy array for visualization
    rgb_img = np.float32(original_image) / 255
    visualization = show_cam_on_image(rgb_img, resized_grayscale_cam, use_rgb=True)
    
    # Convert to BGR for OpenCV and save
    visualization = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, visualization)
    print(f"Grad-CAM image saved to: {output_path}")

if __name__ == "__main__":
    # Configuration
    MODEL_PATH = "models/best_model_fold_4.pth"  # Update with the path to your saved model
    IMAGE_PATH = "BEH-180.png"                      # Update with the path to the image you want to test
    IMAGE_SIZE = 224  # Image size used during your training
    MODEL_TYPE = "EnhancedCNN"  # "EnhancedCNN" or "TransferLearningModel"
    GRAD_CAM_OUTPUT_DIR = "results/grad_cam_unseen"

    # You need to fill in the parameters of your best model here.
    # These parameters would have come from your NSGA-II run.
    # Corrected Example for EnhancedCNN:
    if MODEL_TYPE == "EnhancedCNN":
        MODEL_PARAMS = {
            "num_layers": 3,
            "filter_size": 3,
            "num_filters": 64,  # Corrected value based on the error log
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

    # Device setup
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found: {MODEL_PATH}")
    else:
        print("Loading model...")
        model = load_model(MODEL_PATH, MODEL_TYPE, MODEL_PARAMS, device)
        print("Model loaded successfully.")
        
        print(f"Making prediction on image: {IMAGE_PATH}")
        transforms = get_test_transforms(IMAGE_SIZE)
        predicted_label, probabilities, image_tensor, original_image = predict_image(model, IMAGE_PATH, transforms, device)
        
        if predicted_label is not None:
            class_labels = {0: "Non-Glaucoma", 1: "Glaucoma"}
            predicted_class_name = class_labels.get(predicted_label, "Unknown")
            
            print("="*30)
            print(f"Image: {os.path.basename(IMAGE_PATH)}")
            print(f"Predicted Result: {predicted_class_name}")
            print(f"Probabilities: Non-Glaucoma={probabilities[0]:.4f}, Glaucoma={probabilities[1]:.4f}")
            print("="*30)

            # --- Grad-CAM Generation ---
            if not os.path.exists(GRAD_CAM_OUTPUT_DIR):
                os.makedirs(GRAD_CAM_OUTPUT_DIR)

            # Define the target layer for Grad-CAM. This might need to be adjusted.
            # For a custom CNN, this is usually the last convolutional layer.
            # For a transfer learning model (like ResNet), it's often the last layer of the features block.
            if MODEL_TYPE == "EnhancedCNN":
                # Correcting the target layer based on the AttributeError log.
                # Assuming the main convolutional blocks are stored in a 'features' attribute.
                target_layer = model.features[-1]
            elif MODEL_TYPE == "TransferLearningModel":
                # Assuming ResNet-like model, the last convolutional layer is in layer4
                target_layer = model.base_model.layer4[-1]
            else:
                target_layer = None
                print("Warning: Could not determine target layer for Grad-CAM. Skipping.")

            if target_layer:
                output_filename = f"grad_cam_{os.path.basename(IMAGE_PATH)}"
                output_path = os.path.join(GRAD_CAM_OUTPUT_DIR, output_filename)
                
                print("Generating Grad-CAM heatmap...")
                generate_grad_cam(model, target_layer, image_tensor, original_image, predicted_label, output_path)
