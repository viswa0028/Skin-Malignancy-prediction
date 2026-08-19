# 🔬 Skin Lesion Malignancy Prediction via Multimodal Feature Fusion

An end-to-end Deep Learning & Computer Vision pipeline for early detection and classification of skin cancer (Melanoma vs. Benign lesions) using the **ISIC 2018 Dermoscopy Dataset**.

Unlike standard vision-only architectures, this framework combines **deep CNN representations** with **handcrafted computer vision domain features** (color distributions, skin tone metrics, and edge textures) using an integrated **Multimodal Fusion MLP**.

---

## 📽️ Video Demo
- **Demonstration Link:** [Watch Project Demo Video](https://drive.google.com/file/d/1GbGGQOyJAD0zzLkalNF2SCeBjp6MKIme/view?usp=sharing)

---

## 📌 Key Architectural Concepts

Standard Convolutional Neural Networks and Vision Transformers often struggle with dermoscopic skin lesion classification due to lighting variance, subtle pigment transitions, and patient skin-tone variations. 

To overcome this, our pipeline utilizes a **two-stream fusion architecture**:
1. **Vision Stream:** Deep feature extractors (**ResNet-50**, **DenseNet-121**, or **Inception-V3**) capture high-level spatial visual representations.
2. **Handcrafted Feature Stream:** An OpenCV-based feature extraction module measures 15 localized diagnostic cues:
   - **Color Statistics:** Mean, median, and standard deviation across BGR channels.
   - **Color Spaces:** Brightness (V in HSV), Hue, Saturation, and Lightness (L in LAB).
   - **Lesion Morphology & Texture:** Skin pixel coverage ratio and Canny edge density.
3. **Multimodal Fusion Head:** Tabular features pass through a Multi-Layer Perceptron (Linear $\rightarrow$ BatchNorm $\rightarrow$ ReLU $\rightarrow$ Dropout) and are concatenated with the CNN visual embeddings to predict the malignancy probability.

```
       ┌───────────────────────────────┐
       │   Dermoscopy Lesion Image     │
       └──────────────┬────────────────┘
                      │
        ┌─────────────┴─────────────┐
        ▼                           ▼
 ┌──────────────┐            ┌──────────────┐
 │  CNN Feature │            │ OpenCV Skin  │
 │  Extractor   │            │ Feature Extr.│
 │(ResNet/Dense)│            │ (15 Features)│
 └──────┬───────┘            └──────┬───────┘
        │ (2048-dim / 1024-dim)     │ (15-dim)
        │                    ┌──────▼───────┐
        │                    │ Tabular MLP  │
        │                    └──────┬───────┘
        │                           │ (64-dim)
        └─────────────┬─────────────┘
                      ▼
             ┌─────────────────┐
             │ Feature Concaten.│ (Fusion Vector)
             └────────┬────────┘
                      ▼
             ┌─────────────────┐
             │ Dense Classifier│
             └────────┬────────┘
                      ▼
             [ Malignancy Probability ]
```

---

## 📂 Repository Structure

```plaintext
Skin-Malignancy-prediction/
├── feature_extraction.py     # Extracts 15 OpenCV color/texture features & generates feature CSVs
├── train_all_models.py       # Trains all 6 model configs (ResNet, DenseNet, Inception × Image-only/Fusion)
├── evaluation.py             # Full validation evaluation (Accuracy, Recall, Precision, F1, ROC-AUC, ROC curves)
├── frontend.py               # Interactive Streamlit Web UI for real-time inference
├── ensemble.py               # Model ensembling & meta-learner stacking script
├── requirements.txt          # Python project dependencies
├── saved_models/             # Checkpoints directory (.pt files and feature_scaler.pkl)
└── README.md                 # Project documentation
```

---

## 🚀 Step-by-Step User Guide

### Step 1: Environment Setup

Clone the repository and install all required packages using Python 3.10+:

```bash
# Clone repository
git clone https://github.com/viswa0028/Skin-Malignancy-prediction.git
cd Skin-Malignancy-prediction

# Create & activate a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

### Step 2: Prepare Dataset

1. Download the **ISIC 2018 Task 3** training images and ground truth metadata from the [ISIC Archive](https://challenge.isic-archive.com/landing/2018/).
2. Place the images in an `images/` or `ISIC2018_Task3_Training_Input/` directory.
3. Ensure your label CSV contains image identifiers and binary targets (`0` for benign, `1` for malignant).

---

### Step 3: Extract Handcrafted Skin Features

Run [feature_extraction.py](file:///Users/viswa/Desktop/OpenCV/Skin/Skin-Malignancy-prediction/feature_extraction.py) to generate the handcrafted color and texture metrics:

```bash
python3 feature_extraction.py
```
* **Output:** Creates `skin_tone_features2.csv` containing the merged image metadata and the 15 extracted computer vision features.

---

### Step 4: Train Deep Learning Models

Train all baseline (Image-Only) and Multimodal (Fusion) models using [train_all_models.py](file:///Users/viswa/Desktop/OpenCV/Skin/Skin-Malignancy-prediction/train_all_models.py):

```bash
python3 train_all_models.py
```

This trains and saves **6 distinct models** into the `saved_models/` folder:
- `resnet_without_features.pt` & `resnet_with_features.pt` (ResNet-50)
- `densenet_without_features.pt` & `densenet_with_features.pt` (DenseNet-121)
- `inception_without_features.pt` & `inception_with_features.pt` (Inception-V3)
- `feature_scaler.pkl` (StandardScaler for tabular features)

---

### Step 5: Evaluate Models & Generate Metrics

Run the comprehensive validation suite in [evaluation.py](file:///Users/viswa/Desktop/OpenCV/Skin/Skin-Malignancy-prediction/evaluation.py):

```bash
python3 evaluation.py
```

* **Metrics Computed:**
  - **Sensitivity / Recall:** Measures percentage of actual malignant lesions detected (minimizing false negatives).
  - **Specificity:** Measures percentage of benign lesions correctly classified.
  - **Precision:** Measures reliability of positive malignancy predictions.
  - **F1-Score:** Harmonic mean of precision and recall.
  - **ROC-AUC & Confusion Matrix:** Complete threshold-independent discrimination metrics.
* **Artifacts Generated:** Saves `roc_curves_comparison.png` comparing all 6 models on a single graph.

---

### Step 6: Launch Interactive Web Application

Launch the Streamlit web dashboard to test single-image inference and inspect extracted features interactively:

```bash
streamlit run frontend.py
```

---

## ⚠️ Disclaimer
This software is intended for research and educational purposes only. It is **not** a certified medical diagnostic device.
