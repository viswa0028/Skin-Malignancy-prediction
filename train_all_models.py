"""
Early Skin Lesion Classification via Multi-Architecture CNN with Handcrafted Feature Fusion
Training pipeline — all 6 model configurations
  - ResNet-50   (image-only  vs  fusion)
  - DenseNet-121 (image-only  vs  fusion)
  - InceptionV3  (image-only  vs  fusion)
"""

import os
import pickle
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

EXCLUDE_COLS = {
    "isic_id", "target", "image", "malignant",
    "lesion_id", "dx", "dx_type", "age", "sex", "localization"
}

ARCH_IMAGE_SIZE = {
    "resnet":     (224, 224),
    "densenet":   (224, 224),
    "inception":  (299, 299),   # InceptionV3 native size
}

ARCH_DISPLAY = {
    "resnet":    "ResNet-50",
    "densenet":  "DenseNet-121",
    "inception": "InceptionV3",
}

class ISICImageDataset(Dataset):
    """Image-only dataset (baseline models)."""

    def __init__(self, df, image_dir, image_size=(224, 224),
                 id_col="isic_id", target_col="target"):
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.id_col    = id_col
        self.labels    = self.df[target_col].values.astype("float32")

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.iloc[idx][self.id_col])
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            img_name += ".jpg"
        img_path = os.path.join(self.image_dir, img_name)
        image    = Image.open(img_path).convert("RGB")
        return self.transform(image), torch.tensor(self.labels[idx])


class ISICFusionDataset(Dataset):
    """Multimodal dataset — image + scaled tabular skin features."""

    def __init__(self, df, image_dir, feature_cols, scaler,
                 image_size=(224, 224), id_col="isic_id", target_col="target"):
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.id_col    = id_col
        self.labels    = self.df[target_col].values.astype("float32")

        raw_features        = self.df[feature_cols].values.astype("float32")
        scaled              = scaler.transform(raw_features)

        scaled              = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        self.skin_features  = scaled.astype("float32")

        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.iloc[idx][self.id_col])
        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            img_name += ".jpg"
        img_path  = os.path.join(self.image_dir, img_name)
        image     = Image.open(img_path).convert("RGB")
        image     = self.transform(image)
        skin_feat = torch.tensor(self.skin_features[idx])
        label     = torch.tensor(self.labels[idx])
        return image, skin_feat, label


def _build_backbone(arch: str):
    if arch == "resnet":
        backbone    = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        out_dim     = backbone.fc.in_features
        backbone.fc = nn.Identity()

    elif arch == "densenet":
        backbone              = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        out_dim               = backbone.classifier.in_features
        backbone.classifier   = nn.Identity()

    elif arch == "inception":
        backbone            = models.inception_v3(
            weights=models.InceptionV3_Weights.DEFAULT,
            aux_logits=True
        )
        out_dim             = backbone.fc.in_features
        backbone.fc         = nn.Identity()
        backbone.AuxLogits  = None      # disable aux classifier after building

    else:
        raise ValueError(f"Unsupported architecture: '{arch}'. "
                         f"Choose from: resnet, densenet, inception.")
    return backbone, out_dim


class ImageOnlyModel(nn.Module):
    def __init__(self, architecture="resnet"):
        super().__init__()
        self.arch          = architecture.lower()
        backbone, out_dim  = _build_backbone(self.arch)
        self.backbone      = backbone
        self.classifier    = nn.Linear(out_dim, 1)

    def forward(self, x):
        feat = self.backbone(x)
        if hasattr(feat, "logits"):
            feat = feat.logits
        return self.classifier(feat)


class FusionModel(nn.Module):
    def __init__(self, num_tabular_features: int, architecture="resnet"):
        super().__init__()
        self.arch          = architecture.lower()
        backbone, out_dim  = _build_backbone(self.arch)
        self.cnn           = backbone

        self.tabular_mlp = nn.Sequential(
            nn.Linear(num_tabular_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        self.classifier = nn.Sequential(
            nn.Linear(out_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, image, tabular_feat):
        img_feat = self.cnn(image)
        if hasattr(img_feat, "logits"):
            img_feat = img_feat.logits
        tab_feat = self.tabular_mlp(tabular_feat)
        fused    = torch.cat([img_feat, tab_feat], dim=1)
        return self.classifier(fused)


def compute_pos_weight(labels: np.ndarray, device) -> torch.Tensor:
    n_neg = (labels == 0).sum()
    n_pos = (labels == 1).sum()
    if n_pos == 0:
        return torch.tensor(1.0).to(device)
    return torch.tensor(n_neg / n_pos, dtype=torch.float32).to(device)


def _eval_epoch(model, loader, criterion, device, fusion=False):
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            if fusion:
                images, tab_feat, labels = batch
                images   = images.to(device)
                tab_feat = tab_feat.to(device)
                labels   = labels.to(device).unsqueeze(1)
                outputs  = model(images, tab_feat)
            else:
                images, labels = batch
                images  = images.to(device)
                labels  = labels.to(device).unsqueeze(1)
                outputs = model(images)

            total_loss  += criterion(outputs, labels).item()
            probs        = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs.flatten())
            all_targets.extend(labels.cpu().numpy().flatten())

    all_preds   = np.array(all_preds)
    all_targets = np.array(all_targets)
    avg_loss    = total_loss / len(loader)
    acc         = accuracy_score(all_targets, (all_preds > 0.5).astype(int))
    auc         = roc_auc_score(all_targets, all_preds) if len(np.unique(all_targets)) > 1 else 0.5
    prec        = precision_score(all_targets, (all_preds > 0.5).astype(int), zero_division=0)
    f1          = f1_score(all_targets, (all_preds > 0.5).astype(int), zero_division=0)
    return avg_loss, acc, auc, prec, f1


def train_model(model, train_loader, val_loader, device,
                model_name, save_dir, epochs=10, lr=1e-4,
                fusion=False, pos_weight=None):

    os.makedirs(save_dir, exist_ok=True)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    save_path  = os.path.join(save_dir, f"{model_name}.pt")

    print(f"\n{'='*55}")
    print(f"  {'FUSION' if fusion else 'IMAGE-ONLY':12} | {model_name}")
    print(f"{'='*55}")

    best_val_auc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            if fusion:
                images, tab_feat, labels = batch
                images   = images.to(device)
                tab_feat = tab_feat.to(device)
                labels   = labels.to(device).unsqueeze(1)
                outputs  = model(images, tab_feat)
            else:
                images, labels = batch
                images  = images.to(device)
                labels  = labels.to(device).unsqueeze(1)
                outputs = model(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        train_loss = total_loss / len(train_loader)
        val_loss, val_acc, val_auc, val_prec, val_f1 = _eval_epoch(
            model, val_loader, criterion, device, fusion=fusion
        )

        print(
            f"  Epoch {epoch+1:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Acc: {val_acc:.4f} | "
            f"AUC: {val_auc:.4f} | "
            f"Prec: {val_prec:.4f} | "
            f"F1: {val_f1:.4f}"
        )

        # FIX 7: Save ONLY when val AUC genuinely improves (not forced on last epoch)
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best AUC {best_val_auc:.4f} — checkpoint saved.")

    print(f"  Best Val AUC: {best_val_auc:.4f}")
    return save_path


def main():
    for candidate in [
        "skin_tone_features_final.csv",
        "skin_tone_features2.csv",
        "ISIC2018_binary_labels.csv",
        "train.csv",
    ]:
        if os.path.exists(candidate):
            csv_file = candidate
            break
    else:
        raise FileNotFoundError(
            "No dataset CSV found. Expected one of: "
            "skin_tone_features_final.csv, skin_tone_features2.csv, "
            "ISIC2018_binary_labels.csv, train.csv"
        )
    print(f"Dataset CSV : {csv_file}")

    df = pd.read_csv(csv_file)

    if "image" in df.columns and "isic_id" not in df.columns:
        df.rename(columns={"image": "isic_id"}, inplace=True)
    if "malignant" in df.columns and "target" not in df.columns:
        df.rename(columns={"malignant": "target"}, inplace=True)

    for candidate in ["images", "ISIC2018_Task3_Training_Input"]:
        if os.path.isdir(candidate):
            image_dir = candidate
            break
    else:
        image_dir = "images"
        print(f"Warning: image directory not found — defaulting to '{image_dir}'.")
    print(f"Image dir  : {image_dir}")

    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDE_COLS
        and np.issubdtype(df[c].dtype, np.number)
    ]
    if not feature_cols:
        print("Warning: No tabular features found — fusion models will be skipped.")
    else:
        print(f"Tabular features ({len(feature_cols)}): {feature_cols}")


    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["target"], random_state=42
    )

    if feature_cols:
        scaler = StandardScaler()
        scaler.fit(train_df[feature_cols].values.astype("float32"))
        os.makedirs("saved_models", exist_ok=True)
        with open("saved_models/feature_scaler.pkl", "wb") as f:
            pickle.dump({"scaler": scaler, "feature_cols": feature_cols}, f)
        print("Scaler saved to saved_models/feature_scaler.pkl")


    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device     : {device}\n")

    train_labels   = train_df["target"].values
    pos_weight     = compute_pos_weight(train_labels, device)
    print(f"pos_weight (neg/pos ratio): {pos_weight.item():.2f}")

    EPOCHS     = 10
    BATCH_SIZE = 16
    LR         = 1e-4
    SAVE_DIR   = "saved_models"
    ARCHS      = ["resnet", "densenet", "inception"]

    print("\n" + "="*55)
    print("  PHASE 1 — IMAGE-ONLY BASELINE MODELS")
    print("="*55)

    for arch in ARCHS:
        img_size   = ARCH_IMAGE_SIZE[arch]
        train_ds   = ISICImageDataset(train_df, image_dir, img_size)
        val_ds     = ISICImageDataset(val_df,   image_dir, img_size)
        train_ld   = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
        val_ld     = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        model      = ImageOnlyModel(architecture=arch).to(device)
        model_name = f"{arch}_without_features"
        train_model(
            model, train_ld, val_ld, device,
            model_name=model_name, save_dir=SAVE_DIR,
            epochs=EPOCHS, lr=LR, fusion=False,
            pos_weight=pos_weight,
        )

    if not feature_cols:
        print("No tabular features — skipping fusion phase.")
        return

    print("\n" + "="*55)
    print("  PHASE 2 — MULTIMODAL FUSION MODELS")
    print("="*55)

    for arch in ARCHS:
        img_size   = ARCH_IMAGE_SIZE[arch]
        train_ds   = ISICFusionDataset(train_df, image_dir, feature_cols, scaler, img_size)
        val_ds     = ISICFusionDataset(val_df,   image_dir, feature_cols, scaler, img_size)
        train_ld   = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=True)
        val_ld     = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

        model      = FusionModel(num_tabular_features=len(feature_cols), architecture=arch).to(device)
        model_name = f"{arch}_with_features"
        train_model(
            model, train_ld, val_ld, device,
            model_name=model_name, save_dir=SAVE_DIR,
            epochs=EPOCHS, lr=LR, fusion=True,
            pos_weight=pos_weight,
        )

    print("\n" + "="*55)
    print("  ALL MODELS TRAINED AND SAVED")
    print(f"  Checkpoints in: {os.path.abspath(SAVE_DIR)}")
    print("="*55)


if __name__ == "__main__":
    main()