# 🧠 Brain Tumor MRI Diagnostic AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://brain-tumor-classifier-resnet-3gswfurjzx4s6aeqjju3vv.streamlit.app/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)

A Deep Learning-powered diagnostic AI tool built with **PyTorch** and **Streamlit**. This application utilizes Transfer Learning via a ResNet-18 Convolutional Neural Network (CNN) to classify MRI scans of brain tumors into four distinct categories with high accuracy. 

It serves as a professional medical software prototype, complete with an interactive dashboard, image inspection tools, and automated report generation.

## 🌐 Live Application
**Try the live web app here:** [Brain Tumor MRI Classifier](https://brain-tumor-classifier-resnet-3gswfurjzx4s6aeqjju3vv.streamlit.app/)

---

## 🚀 Key Features

* **AI Diagnosis:** Instantly classifies uploaded MRI scans into 4 categories: `Glioma`, `Meningioma`, `Pituitary`, or `No Tumor`.
* **Interactive Image Inspector:** Built-in sliders allow users to adjust the contrast and brightness of dark MRI scans in real-time to manually inspect physical anomalies.
* **Confidence Distribution:** Uses Altair to render dynamic bar charts, showing the exact probability breakdown across all 4 classes to help gauge the AI's confidence.
* **Automated Medical Reports:** Users can input patient metadata (Name/ID, Age, Scan Date) to generate and download a comprehensive `.txt` diagnostic report for hospital record-keeping.

## 🧠 Model Architecture & Transfer Learning

This project leverages **Transfer Learning** to achieve high accuracy on a relatively small medical dataset.
* **Base Model:** ResNet-18 (Pre-trained on ImageNet).
* **Why ResNet?** The 18-layer deep architecture combined with "Residual Connections" allows the model to detect microscopic medical textures without forgetting features, massively outperforming standard custom-built CNNs and eliminating confusion between visually similar tumors (like Meningioma and Pituitary).
* **Preprocessing:** Images are resized to `224x224` and strictly normalized using ImageNet pixel standards (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

## 📊 Dataset Classification
The model was trained to identify four specific states from standard MRI scans:
1. **Glioma Tumor** 🟣
2. **Meningioma Tumor** 🔴
3. **Pituitary Tumor** 🟡
4. **No Tumor (Healthy)** 🟢

## 🛠️ Technology Stack
* **Deep Learning:** PyTorch, Torchvision
* **Frontend/Dashboard:** Streamlit
* **Data Visualization:** Altair, Pandas
* **Image Processing:** Python Imaging Library (PIL)

---

## 💻 Local Installation & Usage

To run this project locally on your own machine, follow these steps:

**1. Clone the repository**
```bash
git clone https://github.com/durg-giri123/Brain-Tumor-Classifier-ResNet.git
cd Brain-Tumor-Classifier-ResNet
