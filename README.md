# NTIRE 2025 Image Super-Resolution Challenge (x4)

This repository contains the implementation of our solution for the **NTIRE 2025 Image Super-Resolution Challenge (x4)**, based on the **Residual Channel Attention Network (RCAN)**. Our approach uses a deep residual network with channel attention mechanisms to enhance image details and improve the perceptual quality of super-resolved images.

---

## 📚 **Table of Contents**

- [General Description](#general-description)
- [Training Strategy](#training-strategy)
- [Pretrained Model](#pretrained-model)
- [How to Run](#how-to-run)
---

## 📖 **General Description**

Our model is based on the **Residual Channel Attention Network (RCAN)**, a state-of-the-art architecture for single image super-resolution (SISR). RCAN utilizes:

- **Deep Residual Network:** Stacking multiple residual groups to learn high-level representations.
- **Channel Attention Mechanism:** Improves feature representation by emphasizing important channels.
- **Upscaling with Sub-Pixel Convolution:** Recovers high-resolution images from low-resolution inputs.

---

## 🏋️ **Training Strategy**

### Dataset:
- **Training Dataset:** DIV2K (800 high-resolution images)
- **Validation Dataset:** DIV2K (validation set of 100 images)
- **Testing Dataset:** Provided by NTIRE 2025 Challenge

### Training Details:
- **Optimizer:** Adam
- **Loss Function:** L1 Loss (Mean Absolute Error)
- **Batch Size:** 16
- **Learning Rate:** 1e-4 with cosine annealing
- **Training Epochs:** 50
- **Augmentation:** Random rotation, flipping, and scaling

---

## 📤 **Pretrained Model**

The pretrained RCAN model was used.

---

## 🚀 **How to Run**

### 1. **Clone the Repository**
```bash
git clone https://github.com/pathak-amanraj/Computer-Vision_Project-2.git
cd NTIRE_Image_Super_Resolution_Challenge_2025

