# 🧠 Brain MRI Tumor Classification using CNNs
**📢 Credits:** This project was developed by **Prathamesh Bhamare**.
**🔗 Linkedin:** https://www.linkedin.com/in/prathamesh-bhamare-7480b52b2/

## 📌 Overview
This project utilizes Convolutional Neural Networks (CNNs) to classify Brain MRI scans into four categories:
- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary Tumor**

A **VGG16-based CNN model** has been trained to detect brain tumors from MRI images with high accuracy. The dataset is sourced from **Kaggle**, and the model is fine-tuned for optimal performance.

---

## 📂 Dataset
The dataset is obtained from **[Kaggle: Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)**.  
It is divided into:
- **Training Set**
- **Validation Set**
- **Testing Set**

To balance the dataset, image augmentation techniques such as **rotation, width & height shifts, and rescaling** have been applied.

---

## 📊 Data Preprocessing & Augmentation
- **Image Rescaling:** Normalized pixel values (0 to 1).
- **Data Augmentation:** Applied transformations to artificially increase the dataset size.
- **One-Hot Encoding:** Converted categorical labels into one-hot vectors.
- **Dataset Splitting:**
  - **Training Data:** Used for model learning.
  - **Validation Data:** Used to fine-tune hyperparameters.
  - **Test Data:** Used to evaluate model performance.

---

## 🔥 Model Architecture
The model is based on **VGG16**, a pre-trained deep learning model. The architecture consists of:
1. **VGG16 Base Layers** (pre-trained on ImageNet, last 5 layers fine-tuned)
2. **Global Average Pooling**
3. **Fully Connected Layers**
4. **Batch Normalization & Dropout**
5. **Softmax Activation** for multi-class classification

### ✨ Optimizations:
- **Transfer Learning**: Utilized pre-trained VGG16 to improve accuracy.
- **Dropout (0.3)**: Reduced overfitting.
- **Batch Normalization**: Improved convergence.
- **ReduceLROnPlateau**: Adaptive learning rate adjustment.

---

## 🏋️ Model Training
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam (Learning Rate: `1e-4`)
- **Batch Size:** `32`
- **Epochs:** `5`
- **Callbacks Used:**
  - **ModelCheckpoint:** Saves the best model during training.
  - **ReduceLROnPlateau:** Reduces learning rate when validation loss stops improving.

---

## 📈 Model Performance
After training, the best model was evaluated on the **test set**.  
📌 **Final Results:**
- **Test Accuracy:** ✅ High Accuracy Achieved
- **Loss:** 📉 Minimal Loss
- **Evaluation Metrics Used:** Accuracy, Loss

---
