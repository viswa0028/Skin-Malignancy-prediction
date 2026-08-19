"""
Comprehensive Model Evaluation Suite for Skin Lesion Malignancy Classification
Evaluates trained models (Image-only & Multimodal Fusion) on full validation/test datasets.
Metrics: Accuracy, Precision, Recall (Sensitivity), Specificity, F1-score, ROC-AUC, Confusion Matrix.
"""

import os
import pickle
import random
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models

class ISICImageEvalDataset(Dataset):
    """Dataset for Image-Only Models"""
    def __init__(self, df, image_dir, image_size=(224, 224), id_col="isic_id", target_col="target"):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.id_col = id_col
        self.labels = self.df[target_col].values.astype("float32")
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.iloc[idx][self.id_col])
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            img_name += ".jpg"
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), torch.tensor(self.labels[idx])


class ISICFusionEvalDataset(Dataset):
    """Dataset for Multimodal Fusion Models"""
    def __init__(self, df, image_dir, feature_cols, scaler=None, image_size=(224, 224), id_col="isic_id", target_col="target"):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.id_col = id_col
        self.labels = self.df[target_col].values.astype("float32")

        raw_features = self.df[feature_cols].values.astype("float32")
        if scaler is not None:
            raw_features = scaler.transform(raw_features)
        raw_features = np.nan_to_num(raw_features, nan=0.0, posinf=0.0, neginf=0.0)
        self.skin_features = raw_features.astype("float32")

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.iloc[idx][self.id_col])
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            img_name += ".jpg"
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), torch.tensor(self.skin_features[idx]), torch.tensor(self.labels[idx])

def _build_backbone(arch: str):
    if arch == "resnet":
        backbone = models.resnet50(weights=None)
        out_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    elif arch == "densenet":
        backbone = models.densenet121(weights=None)
        out_dim = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
    elif arch == "inception":
        backbone = models.inception_v3(weights=None, aux_logits=False)
        out_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return backbone, out_dim


class ImageOnlyModel(nn.Module):
    def __init__(self, architecture="resnet"):
        super().__init__()
        self.arch = architecture.lower()
        backbone, out_dim = _build_backbone(self.arch)
        self.backbone = backbone
        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, x):
        feat = self.backbone(x)
        if hasattr(feat, "logits"):
            feat = feat.logits
        return self.classifier(feat)


class FusionModel(nn.Module):
    def __init__(self, num_tabular_features: int, architecture="resnet"):
        super().__init__()
        self.arch = architecture.lower()
        backbone, out_dim = _build_backbone(self.arch)
        self.cnn = backbone
        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(out_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, image, tabular_feat):
        img_feat = self.cnn(image)
        if hasattr(img_feat, "logits"):
            img_feat = img_feat.logits
        tab_feat = self.tabular_mlp(tabular_feat)
        fused = torch.cat([img_feat, tab_feat], dim=1)
        return self.classifier(fused)


def compute_all_metrics(y_true: np.ndarray, y_probs: np.ndarray, threshold: float = 0.5) -> dict:
    """Computes comprehensive evaluation metrics."""
    y_pred = (y_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0) # Sensitivity
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception:
        auc = 0.5

    # Specificity from confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall (Sensitivity)": rec,
        "Specificity": spec,
        "F1-Score": f1,
        "ROC-AUC": auc,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "Confusion_Matrix": cm
    }


def evaluate_model_on_dataset(model, loader, device, is_fusion=False, threshold=0.5):
    """Runs inference across full loader and returns probabilities and metrics."""
    model.eval()
    all_probs = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            if is_fusion:
                images, tab_feat, labels = batch
                images = images.to(device)
                tab_feat = tab_feat.to(device)
                outputs = model(images, tab_feat)
            else:
                images, labels = batch
                images = images.to(device)
                outputs = model(images)

            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_targets.extend(labels.cpu().numpy().flatten())

    y_true = np.array(all_targets)
    y_probs = np.array(all_probs)
    metrics = compute_all_metrics(y_true, y_probs, threshold=threshold)
    return metrics, y_true, y_probs


def run_full_evaluation(val_csv="val.csv", image_dir="ISIC2018_Task3_Training_Input", models_dir="saved_models", threshold=0.5):
    """
    Finds all saved models in models_dir, evaluates each on val_csv, and plots comparative results.
    """
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"  RUNNING FULL MODEL EVALUATION ON: {val_csv}")
    print(f"  Device: {device} | Threshold: {threshold}")
    print(f"=======================================================\n")

    if not os.path.exists(val_csv):
        print(f"Error: Validation CSV '{val_csv}' not found.")
        return

    val_df = pd.read_csv(val_csv)
    if "image" in val_df.columns and "isic_id" not in val_df.columns:
        val_df.rename(columns={"image": "isic_id"}, inplace=True)
    if "malignant" in val_df.columns and "target" not in val_df.columns:
        val_df.rename(columns={"malignant": "target"}, inplace=True)

    # Load scaler if available
    scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
    scaler = None
    feature_cols = []
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler_data = pickle.load(f)
            scaler = scaler_data.get("scaler")
            feature_cols = scaler_data.get("feature_cols", [])
    else:
        exclude = {"isic_id", "target", "image", "malignant", "lesion_id", "dx", "dx_type", "age", "sex", "localization"}
        feature_cols = [c for c in val_df.columns if c not in exclude and np.issubdtype(val_df[c].dtype, np.number)]

    results_table = []
    roc_curves_data = {}

    # Define model list to evaluate
    model_configs = [
        ("ResNet-50 (Image Only)", "resnet", False, "resnet_without_features.pt"),
        ("ResNet-50 + Fusion", "resnet", True, "resnet_with_features.pt"),
        ("DenseNet-121 (Image Only)", "densenet", False, "densenet_without_features.pt"),
        ("DenseNet-121 + Fusion", "densenet", True, "densenet_with_features.pt"),
        ("InceptionV3 (Image Only)", "inception", False, "inception_without_features.pt"),
        ("InceptionV3 + Fusion", "inception", True, "inception_with_features.pt"),
    ]

    for display_name, arch, is_fusion, filename in model_configs:
        model_path = os.path.join(models_dir, filename)
        if not os.path.exists(model_path):
            print(f"[-] Checkpoint '{filename}' not found in '{models_dir}'. Skipping...")
            continue

        print(f"[+] Evaluating {display_name}...")
        img_size = (299, 299) if arch == "inception" else (224, 224)

        if is_fusion:
            if not feature_cols:
                print(f"    Skipping fusion model {display_name} because no tabular features exist.")
                continue
            dataset = ISICFusionEvalDataset(val_df, image_dir, feature_cols, scaler=scaler, image_size=img_size)
            model = FusionModel(num_tabular_features=len(feature_cols), architecture=arch).to(device)
        else:
            dataset = ISICImageEvalDataset(val_df, image_dir, image_size=img_size)
            model = ImageOnlyModel(architecture=arch).to(device)

        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)

        metrics, y_true, y_probs = evaluate_model_on_dataset(model, loader, device, is_fusion=is_fusion, threshold=threshold)

        results_table.append({
            "Model": display_name,
            "Accuracy": f"{metrics['Accuracy']*100:.2f}%",
            "Precision": f"{metrics['Precision']:.4f}",
            "Recall (Sensitivity)": f"{metrics['Recall (Sensitivity)']:.4f}",
            "Specificity": f"{metrics['Specificity']:.4f}",
            "F1-Score": f"{metrics['F1-Score']:.4f}",
            "ROC-AUC": f"{metrics['ROC-AUC']:.4f}",
            "TP": metrics["TP"],
            "FP": metrics["FP"],
            "TN": metrics["TN"],
            "FN": metrics["FN"],
        })

        fpr, tpr, _ = roc_curve(y_true, y_probs)
        roc_curves_data[display_name] = (fpr, tpr, metrics["ROC-AUC"])

    if not results_table:
        print("No models were evaluated. Please check model paths in 'saved_models/'.")
        return

    # Print summary table
    df_results = pd.DataFrame(results_table)
    print("\n" + "="*80)
    print("                     MODEL PERFORMANCE COMPARISON")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80 + "\n")

    # Plot ROC Curves
    plt.figure(figsize=(10, 7))
    for name, (fpr, tpr, auc_val) in roc_curves_data.items():
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=12)
    plt.title("Comparative ROC Curves across Models", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    roc_plot_path = "roc_curves_comparison.png"
    plt.savefig(roc_plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparative ROC curves to '{roc_plot_path}'")
    plt.close()


def evaluate_single_random_image(model_path="saved_models/resnet_with_features.pt",
                                 csv_path="val.csv",
                                 image_dir="ISIC2018_Task3_Training_Input"):
    """Visual inspection tool for a single random validation sample."""
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    if not os.path.exists(csv_path) or not os.path.exists(model_path):
        print("Required CSV or model checkpoint not found.")
        return

    df = pd.read_csv(csv_path)
    if "image" in df.columns and "isic_id" not in df.columns:
        df.rename(columns={"image": "isic_id"}, inplace=True)
    if "malignant" in df.columns and "target" not in df.columns:
        df.rename(columns={"malignant": "target"}, inplace=True)

    exclude = {"isic_id", "target", "image", "malignant", "lesion_id", "dx", "dx_type", "age", "sex", "localization"}
    feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
    is_fusion = "with_features" in model_path.lower()
    arch = "inception" if "inception" in model_path.lower() else ("densenet" if "densenet" in model_path.lower() else "resnet")
    img_size = (299, 299) if arch == "inception" else (224, 224)

    if is_fusion:
        ds = ISICFusionEvalDataset(df, image_dir, feature_cols, image_size=img_size)
        model = FusionModel(num_tabular_features=len(feature_cols), architecture=arch).to(device)
    else:
        ds = ISICImageEvalDataset(df, image_dir, image_size=img_size)
        model = ImageOnlyModel(architecture=arch).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    model.eval()

    idx = random.randint(0, len(ds) - 1)
    if is_fusion:
        img, feat, label = ds[idx]
        with torch.no_grad():
            out = model(img.unsqueeze(0).to(device), feat.unsqueeze(0).to(device))
    else:
        img, label = ds[idx]
        with torch.no_grad():
            out = model(img.unsqueeze(0).to(device))

    prob = torch.sigmoid(out).item()
    pred_label = 1 if prob >= 0.5 else 0
    actual_label = int(label.item())

    print(f"\n--- Random Sample Inspection (Index {idx}) ---")
    print(f"Model: {os.path.basename(model_path)}")
    print(f"Predicted Probability: {prob:.4f}")
    print(f"Predicted Class: {'Malignant' if pred_label == 1 else 'Benign'}")
    print(f"Actual Class:    {'Malignant' if actual_label == 1 else 'Benign'}")

    # Display image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    disp = img * std + mean
    disp = disp.permute(1, 2, 0).numpy().clip(0, 1)

    plt.figure(figsize=(6, 6))
    plt.imshow(disp)
    color = "green" if actual_label == pred_label else "red"
    plt.title(f"Actual: {'Malignant' if actual_label == 1 else 'Benign'} | Pred: {'Malignant' if pred_label == 1 else 'Benign'} ({prob:.2f})",
              color=color, fontsize=13, fontweight="bold")
    plt.axis("off")
    plt.savefig("sample_inspection.png", bbox_inches="tight")
    print("Saved sample visualization to 'sample_inspection.png'")
    plt.close()


if __name__ == "__main__":
    run_full_evaluation(
        val_csv="val.csv" if os.path.exists("val.csv") else "skin_tone_features_final.csv",
        image_dir="images" if os.path.isdir("images") else "ISIC2018_Task3_Training_Input",
        models_dir="saved_models",
        threshold=0.5
    )
