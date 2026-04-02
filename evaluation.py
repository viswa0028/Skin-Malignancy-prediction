import torch
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.models as models
from modern import CNNTabularFusion, ISICFusionDataset

class InceptionTabularFusion(nn.Module):
    def __init__(self, num_skin_features):
        super().__init__()
        self.cnn = models.inception_v3(pretrained=False, aux_logits=False)
        self.cnn.fc = nn.Identity()
        self.skin_mlp = nn.Sequential(
            nn.Linear(num_skin_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 64, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )

    def forward(self, image, skin_feat):
        img_feat = self.cnn(image)
        if isinstance(img_feat, tuple):  
            img_feat = img_feat[0]
            
        skin_feat = self.skin_mlp(skin_feat)
        fused = torch.cat([img_feat, skin_feat], dim=1)
        output = self.classifier(fused)
        return output

class ResNet50Fusion(nn.Module):
    def __init__(self, num_skin_features):
        super().__init__()
        self.cnn = models.resnet50(pretrained=False)
        self.cnn.fc = nn.Identity()
        self.skin_mlp = nn.Sequential(
            nn.Linear(num_skin_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 64, 128),
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

class DenseNetFusion(nn.Module):
    def __init__(self, num_skin_features):
        super().__init__()
        self.cnn = models.densenet121(weights=None)
        self.cnn.classifier = nn.Identity()
        self.skin_mlp = nn.Sequential(
            nn.Linear(num_skin_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024 + 64, 128),
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

class InceptionImageOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.inception_v3(weights=None, aux_logits=False)
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1)
        )

    def forward(self, image):
        img_feat = self.backbone(image)
        if isinstance(img_feat, tuple):  
            img_feat = img_feat[0]
        output = self.classifier(img_feat)
        return output

def evaluate_random_image(model_path="Best feature Fusion Paths/best_inception_model.pth", csv_path="val.csv", image_dir="ISIC2018_Task3_Training_Input"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading dataset from {csv_path}...")
    dataset = ISICFusionDataset(csv_path, image_dir)
    
    if len(dataset) == 0:
        print("Dataset is empty. Check your CSV path.")
        return
        
    print(f"Initializing model for {model_path}...")
    num_skin_features = dataset.skin_features.shape[1]
    is_image_only = False
    
    if "without" in model_path.lower() and "inception" in model_path.lower():
        model = InceptionImageOnly().to(device)
        is_image_only = True
    elif "inception" in model_path.lower():
        model = InceptionTabularFusion(num_skin_features=num_skin_features).to(device)
    elif "densenet" in model_path.lower():
        model = DenseNetFusion(num_skin_features=num_skin_features).to(device)
    elif "resnet" in model_path.lower() and "resnet50" in model_path.lower():
        model = ResNet50Fusion(num_skin_features=num_skin_features).to(device)
    elif "resnet" in model_path.lower() or "ensemble" in model_path.lower():

        from model import CNNTabularFusion as ResNet18Fusion
        model = ResNet18Fusion(num_skin_features=num_skin_features).to(device)
    else:
        model = CNNTabularFusion(num_skin_features=num_skin_features).to(device)
    
    print(f"Loading weights from {model_path}...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=False)
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found. Did you train and save the model?")
        return
        
    model.eval()
    
    idx = random.randint(0, len(dataset) - 1)
    image, skin_feat, label = dataset[idx]
    
    print(f"Running inference on image {idx}...")

    img_input = image.unsqueeze(0).to(device)
    skin_input = skin_feat.unsqueeze(0).to(device)
    
    with torch.no_grad():
        if is_image_only:
            output = model(img_input)
        else:
            output = model(img_input, skin_input)
            
        prob = torch.sigmoid(output).item()
        pred_label = 1 if prob > 0.5 else 0
        
    actual_label = int(label.item())
    
    print(f"Actual Label: {actual_label}")
    print(f"Predicted Label: {pred_label}")
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    display_image = image * std + mean
    
    display_image = display_image.permute(1, 2, 0).numpy()
    display_image = display_image.clip(0, 1) 
    plt.figure(figsize=(6, 6))
    plt.imshow(display_image)
    title_color = "green" if actual_label == pred_label else "red"
    plt.title(f"Actual: {actual_label} | Predicted: {pred_label}",
              color=title_color, fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    evaluate_random_image(
        model_path="Best feature Fusion Paths/best_inception_model.pth", 
        csv_path="val.csv",
        image_dir="ISIC2018_Task3_Training_Input"
    )
