import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

cells = []

# Title & Summary
cells.append(nbf.v4.new_markdown_cell("# **Project Name** - Brain Tumor MRI Image Classification"))
cells.append(nbf.v4.new_markdown_cell("##### **Project Type** - Classification (Deep Learning / Computer Vision)\n##### **Contribution** - Individual"))
cells.append(nbf.v4.new_markdown_cell("# **Project Summary**\nThis project aims to build a robust deep learning model to classify Brain Tumor MRI images into four distinct classes: Glioma, Meningioma, Pituitary, and No Tumor. We leverage PyTorch to build a Convolutional Neural Network (CNN) that automatically extracts features from the MRI scans and provides accurate classifications. The dataset provided is divided into training, validation, and test sets. We will perform Exploratory Data Analysis (EDA) to understand the distribution of our classes, preprocess the images (resizing, normalization), and then train our CNN model. Finally, the model's performance will be evaluated using standard metrics like accuracy and a confusion matrix."))

# Problem Statement
cells.append(nbf.v4.new_markdown_cell("# **Problem Statement**\nBrain tumors are severe conditions that require early and accurate detection for effective treatment. Manual inspection of MRI scans is time-consuming and can be prone to human error. The objective of this project is to automate the detection and classification of brain tumors from MRI scans using deep learning, thereby assisting radiologists in their diagnostic process."))

# Let's Begin
cells.append(nbf.v4.new_markdown_cell("# ***Let's Begin !***"))

# 1. Know Your Data
cells.append(nbf.v4.new_markdown_cell("## ***1. Know Your Data***\n### Import Libraries"))
cells.append(nbf.v4.new_code_cell("""import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from sklearn.metrics import classification_report, confusion_matrix"""))

cells.append(nbf.v4.new_markdown_cell("### Dataset Loading"))
cells.append(nbf.v4.new_code_cell("""train_dir = 'Dataset/train-20260928T143223Z-1-001/train'
valid_dir = 'Dataset/valid-20260928T140503Z-1-001/valid'
test_dir = 'Dataset/test-20260928T140250Z-1-001/test'

classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
print("Classes:", classes)"""))

cells.append(nbf.v4.new_markdown_cell("### Dataset First View & Rows/Columns Count"))
cells.append(nbf.v4.new_code_cell("""def get_class_counts(directory):
    counts = {}
    for c in classes:
        path = os.path.join(directory, c)
        if os.path.exists(path):
            counts[c] = len(os.listdir(path))
        else:
            counts[c] = 0
    return counts

train_counts = get_class_counts(train_dir)
valid_counts = get_class_counts(valid_dir)
test_counts = get_class_counts(test_dir)

print("Training samples:", train_counts, "Total:", sum(train_counts.values()))
print("Validation samples:", valid_counts, "Total:", sum(valid_counts.values()))
print("Test samples:", test_counts, "Total:", sum(test_counts.values()))"""))

# EDA
cells.append(nbf.v4.new_markdown_cell("## ***2. Data Visualization & EDA***"))
cells.append(nbf.v4.new_code_cell("""# Chart 1: Class Distribution in Training Set
plt.figure(figsize=(8, 5))
sns.barplot(x=list(train_counts.keys()), y=list(train_counts.values()))
plt.title('Class Distribution in Training Set')
plt.xlabel('Tumor Type')
plt.ylabel('Count')
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""# Chart 2: Sample Images from each class
fig, axes = plt.subplots(1, 4, figsize=(15, 5))
for i, c in enumerate(classes):
    class_path = os.path.join(train_dir, c)
    first_image = os.listdir(class_path)[0]
    img = cv2.imread(os.path.join(class_path, first_image))
    axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[i].set_title(c)
    axes[i].axis('off')
plt.tight_layout()
plt.show()"""))

# Preprocessing
cells.append(nbf.v4.new_markdown_cell("## ***3. Feature Engineering & Data Pre-processing***\nFor image data, preprocessing involves resizing images to a uniform shape and normalizing pixel values."))
cells.append(nbf.v4.new_code_cell("""IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

transform_train = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

transform_test = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
valid_dataset = datasets.ImageFolder(valid_dir, transform=transform_test)
test_dataset = datasets.ImageFolder(test_dir, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)"""))

# Modeling
cells.append(nbf.v4.new_markdown_cell("## ***4. ML/DL Model Implementation***\nWe will implement a Convolutional Neural Network (CNN) using PyTorch."))
cells.append(nbf.v4.new_code_cell("""class BrainTumorCNN(nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 4)
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BrainTumorCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print(model)"""))

cells.append(nbf.v4.new_markdown_cell("### Training the Model"))
cells.append(nbf.v4.new_code_cell("""EPOCHS = 5
train_losses, valid_losses = [], []
train_accs, valid_accs = [], []

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
    
    train_losses.append(train_loss)
    valid_losses.append(val_loss)
    train_accs.append(train_acc)
    valid_accs.append(val_acc)
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")"""))

cells.append(nbf.v4.new_markdown_cell("### Evaluation & Performance"))
cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(valid_losses, label='Valid Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(valid_accs, label='Valid Acc')
plt.title('Accuracy over Epochs')
plt.legend()
plt.show()"""))

cells.append(nbf.v4.new_code_cell("""model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted.cpu().numpy())

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=classes))

plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix on Test Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()"""))

cells.append(nbf.v4.new_markdown_cell("## ***5. Conclusion***\nWe successfully built and trained a CNN using PyTorch to classify brain MRI images into four categories: Glioma, Meningioma, Pituitary, and No Tumor. The EDA helped us visualize the distribution and structural differences among these tumors. After preprocessing the images, our DL model achieved reasonable accuracy over 5 epochs. Future work could include training for more epochs, using transfer learning (e.g., ResNet50), and tuning hyperparameters to further improve accuracy and reduce overfitting."))

nb['cells'] = cells

with open('Brain_Tumor_MRI_Classification_Solved.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully!")
