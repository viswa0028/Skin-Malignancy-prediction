from torch.utils.data import DataLoader
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torchvision.models as models
import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ISICFusionDataset(Dataset):
    def __init__(self, csv_file, image_dir):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir

        self.skin_features = self.df.drop(
            columns=["isic_id", "target"]
        ).values.astype("float32")

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
            self.df.iloc[idx]["isic_id"] + ".jpg"
        )

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        skin_feat = torch.tensor(self.skin_features[idx])
        label = torch.tensor(self.labels[idx])

        return image, skin_feat, label


class CNNTabularFusion(nn.Module):
    def __init__(self, num_skin_features):
        super().__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        self.skin_mlp = nn.Sequential(
            nn.Linear(num_skin_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, image, skin_feat):
        img_feat = self.cnn(image)
        skin_feat = self.skin_mlp(skin_feat)

        fused = torch.cat([img_feat, skin_feat], dim=1)
        output = self.classifier(fused)

        return output


df = pd.read_csv("skin_tone_features_final.csv")

train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["target"]
)

train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
train_ds = ISICFusionDataset("train.csv", "images/")
val_ds = ISICFusionDataset("val.csv", "images/")

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)


device = "cuda" if torch.cuda.is_available() else "cpu"


model = CNNTabularFusion(
    num_skin_features=train_ds.skin_features.shape[1]
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train_epoch(loader):
    model.train()
    total_loss = 0

    for images, skin_feat, labels in loader:
        images = images.to(device)
        skin_feat = skin_feat.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images, skin_feat)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def eval_epoch(loader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, skin_feat, labels in loader:
            images = images.to(device)
            skin_feat = skin_feat.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images, skin_feat)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    return total_loss / len(loader)


# for epoch in range(10):
#     train_loss = train_epoch(train_loader)
#     val_loss = eval_epoch(val_loader)
#     print(
#         f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
#     torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
model.eval()
import cv2
with torch.no_grad():
    img, skin_feat, _ = val_ds[10]

    img = img.unsqueeze(0).to(device)
    skin_feat = skin_feat.unsqueeze(0).to(device)
    cv2.imshow("img", img.squeeze(0).permute(1,2,0).cpu().numpy())
    cv2.waitKey(0)
    logits = model(img, skin_feat)
    prob = torch.sigmoid(logits)
    
    print("white skin", prob.item())