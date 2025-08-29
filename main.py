#!/usr/bin/env python3
# main.py
import os
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
import torchvision.models as models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from deap import base, creator, tools, algorithms
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image 

# --- GPU / Speed Optimization ---
# This line can accelerate training if your input size is fixed.
torch.backends.cudnn.benchmark = True

# ==============================================================================
# SECTION: Configuration and Setup
# ==============================================================================

def create_dirs(config):
    """Creates necessary project directories based on the configuration."""
    os.makedirs(config['data']['dataset_path'], exist_ok=True)
    os.makedirs(os.path.join(config['results']['base_dir'], config['results']['logs_dir']), exist_ok=True)
    os.makedirs(os.path.join(config['results']['base_dir'], config['results']['models_dir']), exist_ok=True)
    os.makedirs(os.path.join(config['results']['base_dir'], config['results']['metrics_dir']), exist_ok=True)
    os.makedirs(os.path.join(config['results']['base_dir'], config['results']['plots_dir']), exist_ok=True)
    os.makedirs(os.path.join(config['results']['base_dir'], config['results']['nsga_runs_dir']), exist_ok=True)
    print("Project directories have been created.")

# Define the configuration as a dictionary
config = {
    "project_name": "glaucoma_detection_nsga",
    "data": {
        "dataset_path": "datasets",
        "acrima_path": "datasets/Acrima",
        "drishti_path": "datasets/Drishti-GS1",
        "hrf_path": "datasets/HRF",
        "origa_path": "datasets/ORIGA-LIGHT",
        "rim_one_path": "datasets/Rim-One",
        "image_size": 224,
        "batch_size": 32,
        # UPDATED: Use the complete dataset for better accuracy
        "data_subset_ratio": 1.0
    },
    "model": {
        "num_classes": 2,
        "base_model": "resnet18" # Not used in EnhancedCNN, but kept for consistency
    },
    "nsga_ii": {
        "population_size": 3,
        "generations": 3,
        "crossover_probability": 0.9,
        "mutation_probability": 0.2,
        "n_hyperparams": 5 # Not directly used but good for documentation
    },
    "training": {
        # UPDATED: Increased epochs to allow more training time
        "epochs": 10,
        "early_stopping_patience": 3,
        # UPDATED: Increased k-folds for more robust evaluation and better generalization
        "k_folds": 5
    },
    "results": {
        "base_dir": "results",
        "logs_dir": "logs",
        "models_dir": "models",
        "metrics_dir": "metrics",
        "plots_dir": "plots",
        "nsga_runs_dir": "nsga_runs"
    }
}

# Save the configuration to a JSON file for reproducibility
try:
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)
    print("Configuration saved to 'config.json'.")
except IOError as e:
    print(f"Error saving config.json: {e}")

create_dirs(config)


# ==============================================================================
# SECTION: Data Handling & Preprocessing
# ==============================================================================

class GlaucomaDataset(Dataset):
    """
    A custom PyTorch dataset that loads and manages multiple datasets.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a dummy black image if loading fails
            image = Image.fromarray(np.zeros((config['data']['image_size'], config['data']['image_size'], 3), dtype=np.uint8))
            
        if self.transform:
            image = self.transform(image=np.array(image))['image']
        return image, torch.tensor(label, dtype=torch.long)

def get_data_paths_and_labels(subset_ratio=1.0):
    """
    Loads image paths and labels from the actual dataset directories.
    Assumes a structure like: datasets/Acrima/Glaucoma/, datasets/Acrima/Normal/ etc.
    Glaucoma: 1, Non-Glaucoma: 0
    Also includes an option to use a subset of the data.
    """
    all_image_paths = []
    all_labels = []

    base_dataset_path = config['data']['dataset_path']
    dataset_keys = ["acrima_path", "drishti_path", "hrf_path", "origa_path", "rim_one_path"]
    
    for dataset_key in dataset_keys:
        current_dataset_full_path = config['data'][dataset_key]

        glaucoma_dir = os.path.join(current_dataset_full_path, 'Glaucoma')
        normal_dir = os.path.join(current_dataset_full_path, 'Normal')

        if os.path.exists(glaucoma_dir):
            for img_name in os.listdir(glaucoma_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    all_image_paths.append(os.path.join(glaucoma_dir, img_name))
                    all_labels.append(1)

        if os.path.exists(normal_dir):
            for img_name in os.listdir(normal_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                    all_image_paths.append(os.path.join(normal_dir, img_name))
                    all_labels.append(0)
    
    if not all_image_paths:
        print("Warning: No actual images found. Using simulated mock data for development/testing.")
        total_images = 500
        mock_paths = []
        # Ensure mock data directory exists
        mock_data_dir = os.path.join(config['data']['dataset_path'], "mock_data")
        os.makedirs(mock_data_dir, exist_ok=True)

        for i in range(total_images):
            path = os.path.join(mock_data_dir, f"mock_image_{i}.jpg")
            mock_paths.append(path)
            if not os.path.exists(path):
                arr = np.random.randint(0, 255, (config['data']['image_size'], config['data']['image_size'], 3), dtype=np.uint8)
                try:
                    Image.fromarray(arr).save(path)
                except Exception as e:
                    print(f"Error saving mock image {path}: {e}")
        mock_labels = np.random.choice([0, 1], size=total_images, p=[0.5, 0.5])
        print(f"{len(mock_paths)} mock image paths generated (fallback).")
        return mock_paths, np.array(mock_labels)

    # --- Added: Use a subset of the data for faster testing ---
    if subset_ratio < 1.0:
        total_size = len(all_image_paths)
        subset_size = int(total_size * subset_ratio)
        indices = list(range(total_size))
        random.seed(42)  # For reproducibility
        random.shuffle(indices)
        subset_indices = indices[:subset_size]
        all_image_paths = [all_image_paths[i] for i in subset_indices]
        all_labels = np.array(all_labels)[subset_indices]
        print(f"Using a subset of {len(all_image_paths)} images ({subset_ratio*100:.0f}%) for faster runs.")

    num_glaucoma = sum(1 for label in all_labels if label == 1)
    num_normal = sum(1 for label in all_labels if label == 0)
    print(f"Loaded {len(all_image_paths)} actual image paths.")
    print(f"Number of Glaucoma cases: {num_glaucoma}")
    print(f"Number of Normal cases: {num_normal}")
    print(f"Dataset is balanced: {num_glaucoma == num_normal}")
    
    return all_image_paths, np.array(all_labels)

# Define preprocessing and augmentation pipelines using Albumentations
def get_train_transforms(image_size):
    """
    Returns a composed set of transformations for training data.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=45, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_val_transforms(image_size):
    """Returns a composed set of transformations for validation data."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

# ==============================================================================
# SECTION: Model Architecture
# ==============================================================================

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class ChannelAttention(nn.Module):
    def __init__(self, channel_in, reduction_ratio):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channel_in, max(channel_in // reduction_ratio, 1), 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(max(channel_in // reduction_ratio, 1), channel_in, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_out = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv(x_out)
        return self.sigmoid(x_out)

class CBAMBlock(nn.Module):
    def __init__(self, channel_in, reduction_ratio=16, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(channel_in, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class EnhancedCNN(nn.Module):
    def __init__(self, num_layers, filter_size, num_filters, dropout_rate, num_classes=2):
        super(EnhancedCNN, self).__init__()
        self.features = nn.Sequential()
        in_channels = 3

        for i in range(num_layers):
            out_channels = int(num_filters * (2 ** i))
            block = nn.Sequential(
                DepthwiseSeparableConv(in_channels, out_channels, kernel_size=filter_size, padding=filter_size//2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                CBAMBlock(out_channels),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.features.add_module(f'conv_block_{i}', block)
            in_channels = out_channels

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )
        # final_features for Grad-CAM: pick last conv block if exists, else features
        if len(self.features) > 0:
            self.final_features = self.features[-1]
        else:
            self.final_features = None

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class TransferLearningModel(nn.Module):
    def __init__(self, base_model_name, num_classes=2, dropout_rate=0.5):
        super(TransferLearningModel, self).__init__()
        # Ensure correct weights are loaded based on the base_model_name
        if base_model_name == "resnet18":
            self.features = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            num_ftrs = self.features.fc.in_features
            self.features.fc = nn.Identity() # Remove the original classification layer
            self.final_features = self.features.layer4[-1] # Target layer for Grad-CAM
        elif base_model_name == "mobilenet_v3_small":
            self.features = models.mobilenet_v3_small(weights=models.MobileNetV3_Small_Weights.IMAGENET1K_V1)
            num_ftrs = self.features.classifier[0].in_features
            self.features.classifier = nn.Identity() # Remove the original classification layer
            self.final_features = self.features.features[-1] # Target layer for Grad-CAM
        else:
            raise ValueError(f"Base model '{base_model_name}' not supported for transfer learning.")
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(num_ftrs, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ==============================================================================
# SECTION: Training and Evaluation
# ==============================================================================

def train_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0
    y_true_train, y_pred_train = [], []

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        with torch.amp.autocast(device.type):
            outputs = model(images)
            loss = criterion(outputs, labels.long())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        y_true_train.extend(labels.cpu().numpy())
        y_pred_train.extend(preds.cpu().numpy())

    return total_loss / len(dataloader.dataset), y_true_train, y_pred_train

def evaluate_model(model, dataloader, criterion, device, threshold=0.5):
    model.eval()
    total_loss = 0
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.long())
            
            total_loss += loss.item() * images.size(0)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]
            preds = (probabilities > threshold).long()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probabilities.cpu().numpy())

    avg_loss = total_loss / len(dataloader.dataset) if len(dataloader.dataset) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
    except ValueError: # Handles cases where only one class is present in y_true
        auc_roc = 0.0
    except Exception as e:
        print(f"Error calculating ROC AUC: {e}")
        auc_roc = 0.0

    metrics = {
        'accuracy': acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'loss': avg_loss
    }
    
    return metrics, y_true, y_pred, y_prob

def plot_metrics(y_true, y_pred, y_prob, epoch, fold, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    # Confusion Matrix
    if len(np.unique(y_true)) > 1: # Only plot if both classes are present
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Glaucoma', 'Glaucoma'], yticklabels=['Non-Glaucoma', 'Glaucoma'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - Epoch {epoch}, Fold {fold}')
        plt.savefig(os.path.join(save_dir, f'cm_epoch_{epoch}_fold_{fold}.png'))
        plt.close()
    else:
        print(f"Skipping Confusion Matrix plot for epoch {epoch}, fold {fold}: Only one class present in true labels.")
    
    # ROC Curve
    try:
        if len(np.unique(y_true)) > 1 and len(np.unique(y_prob)) > 1: # Ensure enough variety for ROC
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC - Epoch {epoch}, Fold {fold}')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(save_dir, f'roc_epoch_{epoch}_fold_{fold}.png'))
            plt.close()
        else:
            print(f"Skipping ROC plot for epoch {epoch}, fold {fold}: Not enough unique labels/probabilities.")
    except Exception as e:
        print(f"Error plotting ROC curve for epoch {epoch}, fold {fold}: {e}")
        plt.close()
        
    # Precision-Recall Curve
    try:
        if len(np.unique(y_true)) > 1 and len(np.unique(y_prob)) > 1: # Ensure enough variety for PR
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            pr_auc = auc(recall, precision)
            plt.figure()
            plt.plot(recall, precision, lw=2, label=f'PR curve (area = {pr_auc:.2f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'PR - Epoch {epoch}, Fold {fold}')
            plt.legend(loc="lower left")
            plt.savefig(os.path.join(save_dir, f'pr_epoch_{epoch}_fold_{fold}.png'))
            plt.close()
        else:
            print(f"Skipping PR plot for epoch {epoch}, fold {fold}: Not enough unique labels/probabilities.")
    except Exception as e:
        print(f"Error plotting PR curve for epoch {epoch}, fold {fold}: {e}")
        plt.close()

# --- NEW: Plotting of training and validation curves ---
def plot_train_val_curves(train_losses, val_losses, train_aucs, val_aucs, fold, save_dir):
    """
    Plots the training and validation loss and AUC-ROC over epochs for a given fold.
    """
    epochs = range(1, len(train_losses) + 1)
    os.makedirs(save_dir, exist_ok=True)
    
    if not epochs:
        print(f"No epochs to plot for fold {fold}.")
        return

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title(f'Loss - Fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot AUC-ROC
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_aucs, 'bo-', label='Training AUC-ROC')
    plt.plot(epochs, val_aucs, 'ro-', label='Validation AUC-ROC')
    plt.title(f'AUC-ROC - Fold {fold}')
    plt.xlabel('Epochs')
    plt.ylabel('AUC-ROC')
    plt.legend()
    
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(save_dir, f'train_val_curves_fold_{fold}.png'))
    except Exception as e:
        print(f"Error saving train/val curves plot: {e}")
    plt.close()

def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    """
    Loads the model, optimizer, and scheduler from a checkpoint.
    Returns (0, -1.0, 0) if no checkpoint exists.
    """
    if os.path.exists(checkpoint_path):
        try:
            print(f"Loading checkpoint: {checkpoint_path}")
            # IMPORTANT: For NSGA-II, if the model architecture changes per individual,
            # loading previous checkpoints will fail. We'll modify the caller to not
            # attempt this during NSGA-II evaluation, but keep the function for general use.
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_auc = checkpoint['best_val_auc']
            early_stopping_counter = checkpoint['early_stopping_counter']
            print(f"Checkpoint successfully loaded. Training will resume from epoch {start_epoch}.")
            return start_epoch, best_val_auc, early_stopping_counter
        except Exception as e:
            # This error is expected and handled if architectures mismatch in NSGA-II context
            print(f"Error loading checkpoint {checkpoint_path}: {e}. Starting training from scratch.")
            return 0, -1.0, 0
    else:
        print(f"Checkpoint does not exist: {checkpoint_path}. Starting training from scratch.")
        return 0, -1.0, 0

# MODIFIED: train_and_evaluate_pipeline now takes pre-split data
def train_and_evaluate_pipeline(hyperparams, X_train_paths, y_train_labels, X_val_paths, y_val_labels, device, fold_idx=0):
    num_layers = int(hyperparams[0])
    filter_size = int(hyperparams[1])
    num_filters = int(hyperparams[2])
    learning_rate = float(hyperparams[3])
    dropout_rate = float(hyperparams[4])
    
    print(f"\n--- Training Hyperparams (fold {fold_idx+1}) --- layers:{num_layers}, fsize:{filter_size}, nfilters:{num_filters}, lr:{learning_rate:.2e}, drop:{dropout_rate:.3f}")
    
    # Class weights (balanced fallback)
    num_glaucoma_train = np.sum(y_train_labels == 1)
    num_nonglaucoma_train = np.sum(y_train_labels == 0)
    
    weight_for_nonglaucoma = len(y_train_labels) / (2.0 * num_nonglaucoma_train) if num_nonglaucoma_train > 0 else 1.0
    weight_for_glaucoma = len(y_train_labels) / (2.0 * num_glaucoma_train) if num_glaucoma_train > 0 else 1.0
    
    class_weights = torch.tensor([weight_for_nonglaucoma, weight_for_glaucoma], dtype=torch.float32).to(device)

    train_dataset = GlaucomaDataset(X_train_paths, y_train_labels, transform=get_train_transforms(config['data']['image_size']))
    val_dataset = GlaucomaDataset(X_val_paths, y_val_labels, transform=get_val_transforms(config['data']['image_size']))
    
    if len(train_dataset) == 0:
        print(f"Warning: Training dataset for fold {fold_idx} is empty. Skipping training.")
        return 0.0, 0.0, (X_val_paths, y_val_labels, [])
    if len(val_dataset) == 0:
        print(f"Warning: Validation dataset for fold {fold_idx} is empty. Skipping evaluation.")
        return 0.0, 0.0, (X_val_paths, y_val_labels, [])

    train_loader = DataLoader(train_dataset, batch_size=config['data']['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

    model = EnhancedCNN(num_layers, filter_size, num_filters, dropout_rate, num_classes=config['model']['num_classes']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.amp.GradScaler(device.type)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)
    
    checkpoint_path = os.path.join(config['results']['base_dir'], config['results']['models_dir'], f"checkpoint_fold_{fold_idx}.pth")

    # --- MODIFIED: Removed automatic checkpoint loading for NSGA-II individual evaluations ---
    # Each NSGA-II individual should start fresh, as its architecture might differ.
    # The 'load_checkpoint' function is useful for standalone training runs or resuming.
    start_epoch = 0 # Always start from epoch 0 for each individual in NSGA-II evaluation
    best_val_auc = -1.0
    early_stopping_counter = 0
    # If you want to enable checkpointing for resuming NSGA-II *individual* runs,
    # you would need to ensure checkpoint_path is unique per individual (e.g., hash of hyperparams)
    # and handle potential architecture mismatches more gracefully.
    # For now, starting fresh for each individual evaluation is the correct behavior.
    
    best_metrics = {}
    
    y_true_val = [] 
    y_pred_val = [] 
    X_val_fold_for_gradcam = [] 

    train_losses, val_losses, train_aucs_plot, val_aucs_plot = [], [], [], [] 

    for epoch in range(start_epoch, config['training']['epochs']):
        train_loss, y_true_train, y_pred_train = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_metrics, y_true_val_current, y_pred_val_current, y_prob_val_current = evaluate_model(model, val_loader, criterion, device)
        
        try:
            train_auc_current = roc_auc_score(y_true_train, y_pred_train)
        except ValueError:
            train_auc_current = 0.0
        except Exception as e:
            print(f"Error calculating training AUC: {e}")
            train_auc_current = 0.0

        scheduler.step(val_metrics['auc_roc'])

        print(f"Epoch {epoch+1}/{config['training']['epochs']}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_metrics['loss']:.4f}, "
              f"Val AUC: {val_metrics['auc_roc']:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_metrics['loss'])
        train_aucs_plot.append(train_auc_current) 
        val_aucs_plot.append(val_metrics['auc_roc']) 

        # Checkpointing logic: Still save progress, but don't load a mismatched one.
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_auc': best_val_auc,
                'early_stopping_counter': early_stopping_counter
            }, checkpoint_path)
            print(f"Checkpoint saved for epoch {epoch+1}.")
        except Exception as e:
            print(f"Error saving checkpoint for epoch {epoch+1}: {e}")


        if val_metrics['auc_roc'] > best_val_auc:
            print(f"Validation AUC improved from {best_val_auc:.4f} to {val_metrics['auc_roc']:.4f}. Saving best model.")
            best_val_auc = val_metrics['auc_roc']
            best_metrics = val_metrics
            y_true_val = y_true_val_current
            y_pred_val = y_pred_val_current
            X_val_fold_for_gradcam = X_val_paths # Store X_val_paths from the outer scope
            early_stopping_counter = 0

            model_save_path = os.path.join(config['results']['base_dir'], config['results']['models_dir'], f"best_model_fold_{fold_idx}.pth")
            try:
                torch.save(model.state_dict(), model_save_path)
            except Exception as e:
                print(f"Error saving best model to {model_save_path}: {e}")

            plot_metrics(y_true_val_current, y_pred_val_current, y_prob_val_current, epoch, fold_idx, os.path.join(config['results']['base_dir'], config['results']['plots_dir']))
        else:
            early_stopping_counter += 1
        
        if early_stopping_counter >= config['training']['early_stopping_patience']:
            print("Early stopping triggered.")
            break
    
    plot_train_val_curves(train_losses, val_losses, train_aucs_plot, val_aucs_plot, fold_idx, os.path.join(config['results']['base_dir'], config['results']['plots_dir']))

    if best_metrics:
        metrics_path = os.path.join(config['results']['base_dir'], config['results']['metrics_dir'], f'fold_{fold_idx}_metrics.json')
        try:
            with open(metrics_path, 'w') as f:
                json.dump(best_metrics, f, indent=4)
            print(f"Best metrics for fold {fold_idx+1} saved to '{metrics_path}'.")
        except IOError as e:
            print(f"Error saving metrics for fold {fold_idx+1}: {e}")
    else:
        print(f"No best metrics saved for fold {fold_idx+1}.")
        if not y_true_val and len(val_dataset) > 0: 
             val_metrics, y_true_val, y_pred_val, y_prob_val = evaluate_model(model, val_loader, criterion, device)
             X_val_fold_for_gradcam = X_val_paths 

    return best_metrics.get('f1_score', 0.0), best_metrics.get('auc_roc', 0.0), (X_val_fold_for_gradcam, y_true_val, y_pred_val)

# --- NEW: Plot a summary of all folds ---
def plot_final_metrics_summary(final_metrics_list, save_dir):
    """
    Plots a bar chart summarizing the F1-score and AUC-ROC for each fold and the average.
    """
    if not final_metrics_list or not all(m is not None for m in final_metrics_list):
        print("No valid final metrics to plot summary.")
        return

    f1_scores = [m[0] for m in final_metrics_list]
    auc_rocs = [m[1] for m in final_metrics_list]
    
    avg_f1 = np.mean(f1_scores)
    avg_auc = np.mean(auc_rocs)
    
    folds = [f"Fold {i+1}" for i in range(len(final_metrics_list))]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(folds))
    width = 0.35
    
    rects1 = ax.bar(x - width/2, f1_scores, width, label='F1-Score')
    rects2 = ax.bar(x + width/2, auc_rocs, width, label='AUC-ROC')
    
    ax.axhline(y=avg_f1, color='blue', linestyle='--', label=f'Avg F1: {avg_f1:.2f}')
    ax.axhline(y=avg_auc, color='orange', linestyle='--', label=f'Avg AUC: {avg_auc:.2f}')
    
    ax.set_ylabel('Score')
    ax.set_title('Performance Metrics Across All Folds')
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend()
    
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(save_dir, 'final_metrics_summary.png'))
    except Exception as e:
        print(f"Error saving final metrics summary plot: {e}")
    plt.close()

# --- NEW: Evaluate Individual for NSGA-II ---
def evaluate_individual(individual, device):
    """
    Evaluates an individual (set of hyperparameters) by running the K-fold pipeline.
    Returns the average F1-score and AUC-ROC across all folds.
    """
    num_layers, filter_size, num_filters, learning_rate, dropout_rate = individual
    
    print(f"\n--- Evaluating Individual: num_layers={num_layers}, filter_size={filter_size}, num_filters={num_filters}, lr={learning_rate:.2e}, dropout={dropout_rate:.3f} ---")
    
    all_f1_scores = []
    all_auc_rocs = []
    
    # Get all paths and labels once for the entire evaluation
    all_paths, all_labels = get_data_paths_and_labels(subset_ratio=config['data']['data_subset_ratio'])
    
    # Handle cases where k_folds is greater than the number of samples or classes
    num_unique_labels = len(np.unique(all_labels))
    if config['training']['k_folds'] > len(all_labels) or num_unique_labels < 2:
        print(f"Warning: k_folds ({config['training']['k_folds']}) is too high for the dataset size ({len(all_labels)}) or number of classes ({num_unique_labels}). Adjusting k_folds to 1.")
        # We need to temporarily modify the config for this run's `k_folds` in evaluate_individual
        # so that StratifiedKFold doesn't fail.
        original_k_folds = config['training']['k_folds']
        config['training']['k_folds'] = 1 
    
    skf = StratifiedKFold(n_splits=config['training']['k_folds'], shuffle=True, random_state=42)
    splits = list(skf.split(all_paths, all_labels))

    for fold_idx in range(config['training']['k_folds']):
        print(f"  Running Fold {fold_idx+1}/{config['training']['k_folds']} for current individual...")
        
        current_train_idx, current_val_idx = splits[fold_idx]
        X_train_fold = np.array(all_paths)[current_train_idx]
        y_train_fold = all_labels[current_train_idx]
        X_val_fold = np.array(all_paths)[current_val_idx]
        y_val_fold = all_labels[current_val_idx]

        f1, auc, _ = train_and_evaluate_pipeline(
            hyperparams=individual, 
            X_train_paths=X_train_fold, 
            y_train_labels=y_train_fold, 
            X_val_paths=X_val_fold, 
            y_val_labels=y_val_fold, 
            device=device, 
            fold_idx=fold_idx
        )
        all_f1_scores.append(f1)
        all_auc_rocs.append(auc)
    
    # Reset k_folds to original value if it was changed
    if 'original_k_folds' in locals():
        config['training']['k_folds'] = original_k_folds

    avg_f1 = np.mean(all_f1_scores)
    avg_auc = np.mean(all_auc_rocs)
    
    print(f"  Individual Evaluation Complete: Avg F1={avg_f1:.4f}, Avg AUC={avg_auc:.4f}")
    return avg_f1, avg_auc # Return as tuple for DEAP fitness

def explain_predictions(model_path, image_path, device, case_type, hyperparams_for_model):
    """
    Explains the model's prediction using Grad-CAM++.
    hyperparams_for_model: A tuple/list of (num_layers, filter_size, num_filters, dropout_rate)
                           used to reconstruct the model architecture.
    """
    num_layers, filter_size, num_filters, _, dropout_rate = hyperparams_for_model # Unpack relevant
    
    # Reconstruct the model with the correct hyperparameters
    model = EnhancedCNN(num_layers, filter_size, num_filters, dropout_rate, num_classes=2).to(device) 
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}. Skipping Grad-CAM for this image.")
        return

    target_layers = [model.final_features]
    if not target_layers or model.final_features is None: # Check if final_features was correctly assigned
        print("No target layer found for Grad-CAM. Ensure model architecture is correctly defined and final_features is set.")
        return

    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Warning: Could not read image at {image_path}. Skipping Grad-CAM for this image.")
        return
            
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    input_tensor = get_val_transforms(config['data']['image_size'])(image=original_image)['image'].unsqueeze(0)
    input_tensor = input_tensor.to(device)

    # For Grad-CAM, typically you want to target the predicted class.
    # Here we target class 1 (Glaucoma), you might want to dynamically
    # determine the predicted class or focus on specific error types.
    targets = [ClassifierOutputTarget(1)] 
    
    try:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    except Exception as e:
        print(f"Error generating Grad-CAM heatmap for {image_path}: {e}. Skipping heatmap generation.")
        return

    if grayscale_cam is not None and grayscale_cam.shape[0] > 0:
        grayscale_cam = cv2.resize(grayscale_cam[0, :], (original_image.shape[1], original_image.shape[0]))
    else:
        print(f"Grad-CAM output is empty or malformed for {image_path}. Skipping image overlay.")
        return

    cam_image = show_cam_on_image(original_image.astype(np.float32) / 255.0, grayscale_cam, use_rgb=True)
    
    os.makedirs(os.path.join(config['results']['base_dir'], config['results']['plots_dir']), exist_ok=True)
    image_name = os.path.basename(image_path)
    safe_case_type = case_type.replace(" ", "_").replace("/", "_")
    cam_image_path = os.path.join(config['results']['base_dir'], config['results']['plots_dir'], f"gradcam_{safe_case_type}_{image_name}")
    
    try:
        cv2.imwrite(cam_image_path, cv2.cvtColor((cam_image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)) 
        print(f"Grad-CAM image for {case_type} saved to '{cam_image_path}'.")
    except Exception as e:
        print(f"Error saving Grad-CAM image to {cam_image_path}: {e}")
        
# ==============================================================================
# SECTION: Main Execution with NSGA-II
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # NSGA-II Setup
    creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0))  # Maximize F1 and AUC
    creator.create("Individual", list, fitness=creator.FitnessMulti)

    toolbox = base.Toolbox()

    toolbox.register("attr_num_layers", random.choice, [1, 2, 3]) 
    toolbox.register("attr_filter_size", random.choice, [3, 5])  
    toolbox.register("attr_num_filters", random.choice, [16, 32, 64]) 
    toolbox.register("attr_learning_rate", random.choice, [0.0001, 0.001, 0.005, 0.01]) # Added 0.005 for more granular search
    toolbox.register("attr_dropout_rate", random.choice, [0.1, 0.3, 0.5]) 

    toolbox.register("individual", tools.initCycle, creator.Individual, (
        toolbox.attr_num_layers,
        toolbox.attr_filter_size,
        toolbox.attr_num_filters,
        toolbox.attr_learning_rate,
        toolbox.attr_dropout_rate
    ), n=1)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_individual, device=device)

    toolbox.register("mate", tools.cxTwoPoint) 
    
    def mutate_individual(individual, indpb, 
                          num_layers_choices, filter_size_choices, num_filters_choices, 
                          lr_choices, dropout_choices):
        for i, val in enumerate(individual):
            if random.random() < indpb:
                if i == 0: # num_layers
                    individual[i] = random.choice(num_layers_choices)
                elif i == 1: # filter_size
                    individual[i] = random.choice(filter_size_choices)
                elif i == 2: # num_filters
                    individual[i] = random.choice(num_filters_choices)
                elif i == 3: # learning_rate
                    individual[i] = random.choice(lr_choices)
                elif i == 4: # dropout_rate
                    individual[i] = random.choice(dropout_choices)
        return individual,

    toolbox.register("mutate", mutate_individual, 
                     indpb=config['nsga_ii']['mutation_probability'],
                     num_layers_choices=[1, 2, 3],
                     filter_size_choices=[3, 5],
                     num_filters_choices=[16, 32, 64],
                     lr_choices=[0.0001, 0.001, 0.005, 0.01], # Ensure this list matches attr_learning_rate
                     dropout_choices=[0.1, 0.3, 0.5])


    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=config['nsga_ii']['population_size'])

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    try:
        pop, log = algorithms.eaMuPlusLambda(pop, toolbox, 
                                            mu=config['nsga_ii']['population_size'], 
                                            lambda_=config['nsga_ii']['population_size'], 
                                            cxpb=config['nsga_ii']['crossover_probability'], 
                                            mutpb=config['nsga_ii']['mutation_probability'], 
                                            ngen=config['nsga_ii']['generations'], 
                                            stats=stats, 
                                            halloffame=None, 
                                            verbose=True)
    except Exception as e:
        print(f"An error occurred during NSGA-II evolution: {e}")
        pop = [] 
        
    print("\n--- NSGA-II Optimization Complete ---")

    if pop:
        pareto_front = tools.selNSGA2(pop, len(pop)) 

        print("\nPareto Front Solutions:")
        nsga_results = []
        for i, ind in enumerate(pareto_front):
            result = {
                "individual_id": i,
                "hyperparameters": {
                    "num_layers": ind[0],
                    "filter_size": ind[1],
                    "num_filters": ind[2],
                    "learning_rate": ind[3],
                    "dropout_rate": ind[4]
                },
                "fitness_f1_auc": ind.fitness.values
            }
            nsga_results.append(result)
            print(f"  Individual {i}: {ind}, Fitness (F1, AUC) = {ind.fitness.values}")
        
        nsga_results_path = os.path.join(config['results']['base_dir'], config['results']['nsga_runs_dir'], 'nsga_results.json')
        try:
            with open(nsga_results_path, 'w') as f:
                json.dump(nsga_results, f, indent=4)
            print(f"NSGA-II results saved to '{nsga_results_path}'.")
        except IOError as e:
            print(f"Error saving NSGA-II results: {e}")

        best_individual = None
        best_auc_overall = -1.0
        for ind in pareto_front:
            if ind.fitness.values[1] > best_auc_overall: 
                best_auc_overall = ind.fitness.values[1]
                best_individual = ind
        
        if best_individual:
            print(f"\n--- Running final training for best NSGA-II individual for Grad-CAM: {best_individual} ---")
            
            # Re-get all data for this final run
            all_paths_final, all_labels_final = get_data_paths_and_labels(subset_ratio=config['data']['data_subset_ratio'])
            
            # Use StratifiedKFold again for a consistent split to get validation data for Grad-CAM
            # Temporarily adjust k_folds if needed for this specific run, then reset
            num_unique_labels_final = len(np.unique(all_labels_final))
            current_k_folds_final = config['training']['k_folds']
            if current_k_folds_final > len(all_labels_final) or num_unique_labels_final < 2:
                print(f"Warning: k_folds ({current_k_folds_final}) is too high for the final dataset size ({len(all_labels_final)}) or number of classes ({num_unique_labels_final}). Adjusting k_folds to 1 for final run.")
                current_k_folds_final = 1 
            
            skf_final = StratifiedKFold(n_splits=current_k_folds_final, shuffle=True, random_state=42)
            final_splits = list(skf_final.split(all_paths_final, all_labels_final))
            
            # For Grad-CAM, we'll pick the validation set of the first fold from this final run
            # to ensure we have a consistent set of images to explain.
            final_train_idx, final_val_idx = final_splits[0] # Using first fold for consistency
            X_train_final = np.array(all_paths_final)[final_train_idx]
            y_train_final = all_labels_final[final_train_idx]
            X_val_final = np.array(all_paths_final)[final_val_idx]
            y_val_final = all_labels_final[final_val_idx]

            _, _, (X_val_for_gradcam, y_true_for_gradcam, y_pred_for_gradcam) = train_and_evaluate_pipeline(
                best_individual, 
                X_train_paths=X_train_final, 
                y_train_labels=y_train_final, 
                X_val_paths=X_val_final, 
                y_val_labels=y_val_final, 
                device=device, 
                fold_idx=0 # Use fold 0 for the final model save
            )
            
            # Ensure config['training']['k_folds'] is reset if it was temporarily changed for the final run
            config['training']['k_folds'] = current_k_folds_final

            best_overall_model_path = os.path.join(config['results']['base_dir'], config['results']['models_dir'], f"best_model_fold_0.pth")

            if best_overall_model_path and os.path.exists(best_overall_model_path):
                if len(X_val_for_gradcam) > 0:
                    true_positives = [i for i, (yt, yp) in enumerate(zip(y_true_for_gradcam, y_pred_for_gradcam)) if yt == 1 and yp == 1]
                    true_negatives = [i for i, (yt, yp) in enumerate(zip(y_true_for_gradcam, y_pred_for_gradcam)) if yt == 0 and yp == 0]
                    false_positives = [i for i, (yt, yp) in enumerate(zip(y_true_for_gradcam, y_pred_for_gradcam)) if yt == 0 and yp == 1]
                    false_negatives = [i for i, (yt, yp) in enumerate(zip(y_true_for_gradcam, y_pred_for_gradcam)) if yt == 1 and yp == 0]

                    images_to_explain = []
                    if true_positives: images_to_explain.append((X_val_for_gradcam[random.choice(true_positives)], 'True Positive'))
                    if true_negatives: images_to_explain.append((X_val_for_gradcam[random.choice(true_negatives)], 'True Negative'))
                    if false_positives: images_to_explain.append((X_val_for_gradcam[random.choice(false_positives)], 'False Positive'))
                    if false_negatives: images_to_explain.append((X_val_for_gradcam[random.choice(false_negatives)], 'False Negative'))
                    
                    if images_to_explain:
                        print(f"Generating Grad-CAM for best NSGA-II model...")
                        for img_path, case_type in images_to_explain:
                            explain_predictions(best_overall_model_path, img_path, device, case_type, best_individual)
                        print("Grad-CAM images have been saved to the 'results/plots' directory.")
                    else:
                        print("Could not find examples for Grad-CAM for the best NSGA-II individual. The validation set might be too small or all predictions were correct/incorrect.")
                else:
                    print("No validation images available for Grad-CAM for the best NSGA-II individual. Check data loading.")
            else:
                print(f"Best NSGA-II model path not found: {best_overall_model_path}. Cannot generate Grad-CAM.")
        else:
            print("No best individual found from NSGA-II Pareto front to run Grad-CAM.")
    else:
        print("NSGA-II optimization did not return any population. Check for errors during evolution.")


    print("Script finished. Check the 'results' directory for saved models and metrics.")
