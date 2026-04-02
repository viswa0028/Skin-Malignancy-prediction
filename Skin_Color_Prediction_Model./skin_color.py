import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# 1. Configuration
CSV_FILE = 'Skin_tone_dataset.csv'
IMAGE_DIR = 'images'
MODEL_SAVE_PATH = 'skin_tone_model.pth'
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4

# Label mapping for Fitzpatrick Skin Type
SKIN_TYPE_MAP = {
    'I': 0,
    'II': 1,
    'III': 2,
    'IV': 3,
    'V': 4,
    'VI': 5
}

# Categorical mappings for metadata
SEX_MAP = {'male': 0, 'female': 1, '': 2} # 2 for unknown
ANATOM_MAP = {
    'anterior torso': 0,
    'posterior torso': 1,
    'upper extremity': 2,
    'lower extremity': 3,
    'head/neck': 4,
    'palms/soles': 5,
    'lateral torso': 6,
    'oral/genital': 7,
    '': 8 # Unknown
}

class SkinToneDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []
        
        # Read CSV and filter important data
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                skin_type = row.get('fitzpatrick_skin_type', '')
                if skin_type not in SKIN_TYPE_MAP:
                    continue  # Skip rows with missing or invalid skin type
                
                isic_id = row['isic_id']
                img_name = f"{isic_id}.jpg"
                img_path = os.path.join(self.img_dir, img_name)
                
                # Check if image exists
                if not os.path.exists(img_path):
                    continue
                
                # Extract important metadata
                age = float(row.get('age_approx', 0)) if row.get('age_approx') else 0.0
                sex_val = SEX_MAP.get(row.get('sex', ''), 2)
                anatom_val = ANATOM_MAP.get(row.get('anatom_site_general', ''), 8)
                
                # Normalize age roughly (assuming max age ~100)
                age_normalized = age / 100.0
                
                metadata = [age_normalized, sex_val, anatom_val]
                
                self.data.append({
                    'image_path': img_path,
                    'metadata': metadata,
                    'label': SKIN_TYPE_MAP[skin_type]
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        metadata = torch.tensor(item['metadata'], dtype=torch.float32)
        label = torch.tensor(item['label'], dtype=torch.long)
        
        return image, metadata, label

class MultiModalSkinToneModel(nn.Module):
    def __init__(self, num_classes=6, num_metadata_features=3):
        super(MultiModalSkinToneModel, self).__init__()
        
        # Image Feature Extractor (ResNet18)
        self.cnn = models.resnet18(pretrained=True)
        # Remove the final classification layer
        cnn_out_features = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()
        
        # Metadata Feature Extractor (Simple MLP)
        self.meta_mlp = nn.Sequential(
            nn.Linear(num_metadata_features, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU()
        )
        
        # Final Classifier combining both image and metadata features
        self.classifier = nn.Sequential(
            nn.Linear(cnn_out_features + 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, metadata):
        img_features = self.cnn(image)
        meta_features = self.meta_mlp(metadata)
        
        # Concatenate features
        combined_features = torch.cat((img_features, meta_features), dim=1)
        
        # Final classification
        out = self.classifier(combined_features)
        return out

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load Dataset
    print("Loading dataset...")
    dataset = SkinToneDataset("skin_Color_model/Skin_tone_dataset.csv", "skin_Color_model/images", transform=transform)
    print(f"Total valid samples found: {len(dataset)}")
    
    if len(dataset) == 0:
        print("No valid samples found. Please check your data directory and CSV.")
        return
        
    # We will just train on the whole dataset for demonstration.
    # In practice, you should split it into train/val.
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Initialize Model, Loss, and Optimizer 
    model = MultiModalSkinToneModel(num_classes=6, num_metadata_features=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, metadata, labels in dataloader:
            images, metadata, labels = images.to(device), metadata.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images, metadata)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
    print("Training finished.")
    
    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_model()
