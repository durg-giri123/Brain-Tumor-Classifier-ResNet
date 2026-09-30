import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
from sklearn.metrics import classification_report

# Dataset Paths
train_dir = 'Dataset/train-20260928T143223Z-1-001/train'
valid_dir = 'Dataset/valid-20260928T140503Z-1-001/valid'
test_dir = 'Dataset/test-20260928T140250Z-1-001/test'
classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

# ResNet expects 224x224 images and specific ImageNet normalizations
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform_train = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

transform_test = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

print("Loading datasets for Transfer Learning...")
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
valid_dataset = datasets.ImageFolder(valid_dir, transform=transform_test)
test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------------------------
# TRANSFER LEARNING: Load Pre-trained ResNet-18
# ---------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load ResNet18 with ImageNet weights
model = models.resnet18(weights='IMAGENET1K_V1')

# Replace the final classification layer to predict our 4 classes instead of 1000
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 4)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
# We use a smaller learning rate (0.0001) because the model is already mostly trained!
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training Loop
EPOCHS = 8  # 8 Epochs is usually enough for ResNet to adapt
print("Starting ResNet Transfer Learning...")
for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    train_loss = running_loss / len(train_loader)
    train_acc = correct / total
    
    # Validation
    model.eval()
    val_loss, correct_val, total_val = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
            
    val_loss = val_loss / len(valid_loader)
    val_acc = correct_val / total_val
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

# Evaluation
print("Evaluating ResNet on test set...")
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

print("Classification Report (ResNet18):")
print(classification_report(y_true, y_pred, target_names=classes))

# Save the robust model
torch.save(model.state_dict(), 'brain_tumor_resnet.pth')
print("Transfer Learning complete! Model saved to brain_tumor_resnet.pth")
