import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import random

class ISICImageDataset(Dataset):
    def __init__(self, df, image_dir):
        self.df = df
        self.image_dir = image_dir
        self.labels = self.df["target"].values.astype("float32")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.image_dir,
            str(self.df.iloc[idx]["isic_id"]) + ".jpg"
        )
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(self.labels[idx])
        return image, label

def get_resnet_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)
    return model

def get_densenet_model():
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, 1)
    return model

def train_image_model(model, name, train_loader, val_loader, device, epochs=3):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    print(f"--- Training {name} ---")
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Evaluate validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_images = val_images.to(device)
                val_labels = val_labels.to(device).unsqueeze(1)
                outputs = model(val_images)
                loss = criterion(outputs, val_labels)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        print(f"{name} - Epoch {epoch+1}/{epochs} completed. Val Loss: {val_loss:.4f}")
        
    return model

def get_image_predictions(model, loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(probs)
    return np.array(all_preds).flatten()

if __name__ == "__main__":
    print("Loading data...")

    try:
        df = pd.read_csv("ISIC2018_binary_labels.csv")
        if "image" in df.columns:
            df = df.rename(columns={"image": "isic_id"})
        if "malignant" in df.columns:
            df = df.rename(columns={"malignant": "target"})
    except FileNotFoundError:
        print("Data files not found. Please ensure 'ISIC2018_binary_labels.csv' is present.")
        exit(1)

    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=42)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    train_ds = ISICImageDataset(train_df, "ISIC2018_Task3_Training_Input/")
    val_ds = ISICImageDataset(val_df, "ISIC2018_Task3_Training_Input/")
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    
    # --- 1. Train Model 1 (ResNet18) ---
    resnet_model = get_resnet_model()
    resnet_model = train_image_model(resnet_model, "ResNet18", train_loader, val_loader, device, epochs=2)
    resnet_val_preds = get_image_predictions(resnet_model, val_loader, device)
    y_val = val_df["target"].values
    
    print(f"ResNet18 Model AUC: {roc_auc_score(y_val, resnet_val_preds):.4f}")
    
    # --- 2. Train Model 2 (DenseNet121) ---
    densenet_model = get_densenet_model()
    densenet_model = train_image_model(densenet_model, "DenseNet121", train_loader, val_loader, device, epochs=2)
    densenet_val_preds = get_image_predictions(densenet_model, val_loader, device)
    
    print(f"DenseNet121 Model AUC: {roc_auc_score(y_val, densenet_val_preds):.4f}")
    
    # --- 3. Ensemble (Averaging) ---
    print("\n--- Ensembling predictions ---")
    
    # Simple averaging
    ensemble_preds = (resnet_val_preds + densenet_val_preds) / 2.0
    
    # Weighted averaging (tune these weights based on validation performance)
    weighted_ensemble_preds = (0.5 * resnet_val_preds) + (0.5 * densenet_val_preds)
    
    print(f"Simple Ensemble AUC: {roc_auc_score(y_val, ensemble_preds):.4f}")
    print(f"Weighted Ensemble AUC: {roc_auc_score(y_val, weighted_ensemble_preds):.4f}")
    
    # --- 4. Ensemble (Stacking / Meta-Model) ---
    print("\n--- Training Meta-Model (Stacking) ---")
    from sklearn.linear_model import LogisticRegression
    
    X_meta_val = np.column_stack((resnet_val_preds, densenet_val_preds))
    meta_model = LogisticRegression()
    meta_model.fit(X_meta_val, y_val) 
    
    meta_preds = meta_model.predict_proba(X_meta_val)[:, 1]
    print(f"Meta-Model (Logistic Regression) AUC on Val: {roc_auc_score(y_val, meta_preds):.4f}")
    
    print("\nImage-Only Ensemble learning pipeline is ready!")
    
    # --- 5. Random Inference Verification ---
    print("\n--- Running Random Inference ---")
    idx = random.randint(0, len(val_ds) - 1)
    image, label = val_ds[idx]
    actual_label = int(label.item())
    
    img_input = image.unsqueeze(0).to(device)
    
    resnet_model.eval()
    densenet_model.eval()
    
    with torch.no_grad():
        out_res = resnet_model(img_input)
        prob_res = torch.sigmoid(out_res).item()
        
        out_dense = densenet_model(img_input)
        prob_dense = torch.sigmoid(out_dense).item()
        
    # Meta prediction for this single image
    single_meta_x = np.array([[prob_res, prob_dense]])
    prob_meta = meta_model.predict_proba(single_meta_x)[0, 1]
    
    pred_label = 1 if prob_meta > 0.5 else 0
    
    print(f"Random Image Index: {idx}")
    print(f"Actual Label: {actual_label}")
    print(f"ResNet Probability:   {prob_res:.4f}")
    print(f"DenseNet Probability: {prob_dense:.4f}")
    print(f"Meta-Model Final Probability: {prob_meta:.4f}")
    print(f"Ensemble Predicted Class: {pred_label}")
    
    # Un-normalize for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    disp_img = image * std + mean
    disp_img = disp_img.permute(1, 2, 0).numpy().clip(0, 1)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(disp_img)
    title_color = "green" if actual_label == pred_label else "red"
    plt.title(f"Actual: {actual_label} | Predicted: {pred_label}\nMeta Prob: {prob_meta:.2f}",
              color=title_color, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.show()
