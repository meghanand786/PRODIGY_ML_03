# 🐶🐱 Cat vs Dog Image Classification using SVM

## 📌 Overview
This project focuses on building an **Image Classification Model** to distinguish between cats and dogs using **traditional Machine Learning techniques**.

Instead of deep learning, this implementation uses:
- HOG (Histogram of Oriented Gradients) for feature extraction  
- PCA (Principal Component Analysis) for dimensionality reduction  
- SVM (Support Vector Machine) for classification  

---

## 🚀 Features
- Image preprocessing (resize + grayscale conversion)
- Feature extraction using HOG
- Feature scaling using StandardScaler
- Dimensionality reduction using PCA
- Classification using SVM (RBF Kernel)
- Performance evaluation using accuracy & classification report

---

## 🗂️ Dataset
- Dataset contains images of cats and dogs  
- Images are labeled based on filename:
  - `cat.*` → Label 0  
  - `dog.*` → Label 1  
- Total images used: **3000**

📁 Directory Structure:

dataset/
└── train/
├── cat.1.jpg
├── dog.1.jpg
├── ...


---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/cat-dog-classification.git
cd cat-dog-classification

2. Install dependencies
pip install -r requirements.txt

3. Required Libraries
numpy
opencv-python
scikit-learn
scikit-image

▶️ Usage

Run the main script:
python main.py
