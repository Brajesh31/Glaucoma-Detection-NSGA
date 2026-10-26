# Glaucoma Detection using Deep Learning & NSGA-II

This project implements a deep learning pipeline for **Glaucoma detection** using retinal fundus images.  
It combines **Convolutional Neural Networks (CNNs)**, **Transfer Learning**, and **NSGA-II (Genetic Algorithm)** for hyperparameter optimization.  
A **Flask web app** is also included for real-time predictions and Grad-CAM visualizations.

---

## 🎥 Project Demo
[▶️ Watch the Demo Video](https://github.com/codedbyasim/Glaucoma-Detection-NSGA-II/blob/main/Add%20a%20heading%20(3).mp4?raw=true)

---
##  Model Performance
![Model Performance](https://github.com/codedbyasim/Glaucoma-Detection-NSGA-II/blob/main/Images/fold%20gen3.png?raw=true)

##  Grad-CAM++ Visualization
![Grad-CAM++ Visualization](https://github.com/codedbyasim/Glaucoma-Detection-NSGA-II/blob/main/Images/Grad-CAM%2B%2B.png?raw=true)


---

##  Project Structure
- `main.py` → Core training script with:
  - Custom Enhanced CNN with CBAM attention
  - Transfer Learning support (ResNet18, MobileNetV3)
  - K-Fold cross-validation
  - NSGA-II for hyperparameter optimization
  - Metrics, plots, and Grad-CAM generation

- `app.py` → Flask web application for:
  - Uploading fundus images
  - Running inference using trained model
  - Viewing Grad-CAM heatmaps for predictions

- `results/` → Trained models, metrics, and plots will be saved here.
- `datasets/` → Acrima, Drishti-GS1, HRF, ORIGA, Rim-One.

---

##  Requirements
Install dependencies:
```bash
pip install -r requirements.txt

Last updated: 2026-10-26 12:39 UTC
