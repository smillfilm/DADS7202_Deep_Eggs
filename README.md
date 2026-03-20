# DADS7202: DeepEggs 🍳🥚

# Team members:
1. Sumonsiri Techasuntharowat 6720422007
2. Sahaphum Ketkaew 6720422010
3. Supitcha kaewplengsrisakul 6720422020
4. Kritsada Matkaruchit 6720422028

# 🥚 Egg Classification with CNN

## ✨Highlight

1. **After W&B Sweep hyperparameter tuning + 3 runs with different seeds,** the highest average test accuracy is **95.00% ± 1.02% of 🏆ConvNeXt-Tiny**
2. All 5 models achieved **85-95% test accuracy** after 2-stage fine-tuning on just 330 training images
3. **W&B Sweep (Bayesian, 20 trials)** found optimal hyperparameters per architecture — ConvNeXt-Tiny benefits most from **Adam optimizer + high dropout (0.5) + no label smoothing**
4. From **GradCAM analysis**, the models learned to focus on **egg texture and shape** — fried egg's yolk center, omelet's crispy edges, scrambled egg's scattered chunks

## Table of Contents

- [1. Introduction](#1-introduction)
- [2. Data](#2-data)
- [3. Network Architecture](#3-network-architecture)
- [4. Training](#4-training)
- [5. Results](#5-results)
- [6. Discussion](#6-discussion)
- [7. Conclusion](#7-conclusion)

---

## 1. Introduction

- This project aims to test **5 CNN pre-trained models** (`VGG-16`, `ResNet-50`, `EfficientNet-B0`, `MobileNetV3-Large`, `ConvNeXt-Tiny`) on the ImageNet dataset and fine-tune them to classify **4 styles of egg cooking** which is our custom image dataset collected from various websites.
- We compare performance of **5 different CNN architectures** spanning from 2014 to 2022, covering different design philosophies.
- We use **2-Stage Fine-tuning** strategy (freeze → unfreeze) with **Weighted CrossEntropyLoss** to handle class imbalance.
- All experiments are tracked using **Weights & Biases (W&B)** for reproducibility.
- **W&B Sweep (Bayesian optimization)** is used for systematic hyperparameter tuning across all architectures.
- Finally, we use [**`GradCAM`**](#-gradcam-analysis) technique to visualize what the CNN has learned.

[🔝](#highlight)

---

## 2. Data

We classify 4 styles of egg cooking. Each style has distinct visual characteristics:

🥚 **1. Soft-boiled Egg**

Soft white exterior with runny/semi-cooked yolk. Often served in a cup or bowl.

<img width="150" height="150" alt="image" src="https://github.com/user-attachments/assets/93440426-ecfe-4476-8d74-3d87ed69ba3d" />


🍳 **2. Fried Egg**

Flat white base with a raised round yolk in the center. Crispy edges.

<img width="150" height="150" alt="image" src="https://github.com/user-attachments/assets/d85192b8-fa38-4535-82a4-d180eba3e3ec" />


🥘 **3. Omelet**

Deep-fried in oil, puffy and crispy, irregular edges, golden-brown surface with bubble texture.

<img width="150" height="150" alt="image" src="https://github.com/user-attachments/assets/96db7cb8-5a0f-46cd-b84f-6c8ae2130a37" />


🥄 **4. Scrambled Eggs**

Soft, small chunky curds scattered on plate. No defined shape. Creamy yellow color.

<img width="150" height="150" alt="image" src="https://github.com/user-attachments/assets/36f5ac6f-e0d5-4c23-b6b6-267dc5100f14" />

### Dataset Summary

| Class | Name | Train | Test | Total |
|-------|------|-------|------|-------|
| `1` | fried_egg | 81 | 20 | 101 |
| `2` | omelet | 88 | 20 | 108 |
| `3` | scrambled | 80 | 20 | 100 |
| `4` | soft_boiled | 81 | 20 | 101 |
| | **Total** | **330** | **80** | **410** |

<br>
<img width="850" height="490" alt="image" src="https://github.com/user-attachments/assets/b7cc3090-1d8b-4c9a-bfb5-9b934aace625" />

#### 📍 Data Source:
- All images were **manually collected from websites** including Google Images, food blogs, recipe websites, and social media (Instagram, Pinterest)
- Searched using both Thai and English keywords
<img width="1554" height="788" alt="image" src="https://github.com/user-attachments/assets/857c606c-634a-4e25-ab91-05c5d07358dd" />

#### 🧹 Data Cleaning:
- All images are RGB format with varying sizes (168px to 2463px)

#### 📊 EDA:

<img width="1589" height="463" alt="image" src="https://github.com/user-attachments/assets/efb7d071-cd0f-4ba6-b966-f7a79b7816d2" />

- **Imbalanced ratio:** 1.10x (Max: 88 / Min: 80) → **Fairly balanced**
- **Image size:** Width 168-2463px, Height 192-2560px (highly varied from web)
- **Brightness:** Mean pixel value 59-211 (diverse lighting conditions)
- **Handling:** Applied **Weighted CrossEntropyLoss** as precaution

#### Data Pre-processing & Data Augmentation:

- Images resized to **256×256** then **RandomCrop to 224×224** (ImageNet standard)
- Normalized with ImageNet mean/std: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`

```python
RandomHorizontalFlip(p=0.5)           # Egg can be flipped L-R
RandomRotation(degrees=20)             # Plate may be tilted
ColorJitter(brightness=0.3, hue=0.05)  # Simulate different lighting (low hue: egg color matters)
RandomPerspective(0.15, p=0.3)         # Different camera angles
RandomAffine(translate=0.1, scale=0.9-1.1)
```

#### Data Splitting (train/val/test):

| Set | Count | Method |
|-----|-------|--------|
| **Train** | 264 (80%) | `random_split` from train_set |
| **Validation** | 66 (20%) | `random_split` from train_set |
| **Test** | 80 | Separate folder |

[🔝](#highlight)

---

## 3. Network Architecture

### Pre-trained Models

We selected 5 architectures that differ significantly in design philosophy:

| # | Model | Year | Params | Top-1 Acc. (%) | Key Concept | Why Selected |
|:-:|:---|:-:|:---:|:---:|:---|:---|
| 1 | **VGG-16** | 2014 | 138M | 71.30 | Plain Stack | Classic baseline, good GradCAM visualization |
| 2 | **ResNet-50** | 2015 | 25.6M | 74.90 | Skip Connection | Solves vanishing gradient, most popular |
| 3 | **EfficientNet-B0** | 2019 | 5.3M | 77.69 | Compound Scaling | Lightweight but accurate |
| 4 | **MobileNetV3-Large** | 2019 | 5.5M | 74.04 | Depthwise Separable | Lightest, mobile-deployable |
| 5 | **ConvNeXt-Tiny** | 2022 | 28.6M | 81.30 | Modernized CNN | Latest SOTA, competes with ViT |

### Transfer Learning Strategy

For all backbones, we apply the same approach:

* **Keep:** Feature extractor (conv layers) + pre-trained ImageNet weights.
* **Remove:** Original classification head (1,000 classes).
    * **VGG-16 (2014):** `FC(4096) → ReLU → Dropout → FC(4096) → ReLU → Dropout → FC(1000)`
    * **ResNet-50 (2015):** `FC(2048 → 1000)`
    * **EfficientNet-B0 (2019):** `Dropout(0.2) → FC(1280 → 1000)`
    * **MobileNetV3 (2019):** `FC(1280 → 1000)`
    * **ConvNeXt-Tiny (2022):** `LayerNorm → FC(768 → 1000)`
* **Add:** Custom classifier head tailored for our 4 egg classes.

### Custom Classifier Head

```
[Backbone Output] → Dropout(p) → Linear(in, 256) → ReLU → Dropout(p*0.6) → Linear(256, 4)
```

> `in` varies: VGG=4096, ResNet=2048, EfficientNet=1280, MobileNet=1280, ConvNeXt=768

### Pre-trained Baseline (No Fine-tuning)

We tested ResNet-50 (ImageNet) on our egg images **without any fine-tuning**:

<img width="1943" height="985" alt="image" src="https://github.com/user-attachments/assets/66d6775a-b915-4896-a239-7b249e86bb91" />

> Pre-trained model predicts eggs as "frying pan", "pizza", "mashed potato" → **Confirms need for fine-tuning!**

[🔝](#highlight)

---

## 4. Training

### 2-Stage Fine-tuning Strategy

| Stage | Action | Frozen Layers | Epochs | Learning Rate |
|-------|--------|--------------|--------|---------------|
| **Stage 1** | Train classifier head only | ✅ Backbone frozen | 8 | from Sweep |
| **Stage 2** | Fine-tune entire model | ❌ All unfrozen | 25 (+Early Stop) | from Sweep |

**Why 2 stages?** The new classifier head starts with random weights. If we unfreeze the backbone immediately, random gradients from the head would destroy the good pre-trained weights.

### Hyperparameter Tuning — W&B Sweep (Bayesian, 20 trials)

We used **Weights & Biases Sweep** with Bayesian optimization to find the best hyperparameters for each architecture automatically.

#### Sweep Search Space:

| Parameter | Search Space |
|-----------|-------------|
| Architecture | vgg16, resnet50, efficientnet_b0, mobilenet_v3, convnext_tiny |
| Stage 1 LR | log_uniform [1e-4, 1e-2] |
| Stage 2 LR | log_uniform [1e-5, 1e-3] |
| Dropout | [0.2, 0.3, 0.4, 0.5] |
| Weight Decay | log_uniform [1e-5, 1e-3] |
| Optimizer | [AdamW, SGD+Momentum, Adam] |
| Label Smoothing | [0.0, 0.1, 0.2] |

#### Best Hyperparameters from Sweep (per architecture):

| Model | Optimizer | Stage2 LR | Dropout | Weight Decay | Label Smooth | Sweep Val Acc |
|-------|:---------:|:---------:|:-------:|:------------:|:------------:|:-------------:|
| VGG-16 | AdamW | 1.9e-05 | 0.2 | 2.7e-05 | 0.2 | 98.48% |
| ResNet-50 | Adam | 3.5e-05 | 0.3 | 4.9e-04 | 0.1 | 98.48% |
| EfficientNet-B0 | AdamW | 8.8e-05 | 0.2 | 3.8e-05 | 0.1 | 96.97% |
| MobileNetV3 | Adam | 1.9e-04 | 0.4 | 7.3e-05 | 0.2 | 98.48% |
| **ConvNeXt-Tiny** | **Adam** | **1.5e-05** | **0.5** | **8.0e-04** | **0.0** | **100.0%** |

### Other Fixed Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Loss | **Weighted CrossEntropyLoss** | Handles class imbalance |
| LR Scheduler | **CosineAnnealingLR** | Gradually decreases LR following cosine curve |
| Batch Size | **32** | Balance speed & GPU memory |
| Early Stopping | **patience=7** | Stop if val_acc doesn't improve for 7 epochs |

### Experiment Tracking

- All experiments tracked with **Weights & Biases (W&B)**
- Team project: [egg-classification on W&B](https://wandb.ai/mild-supitcha25-nida-business-school/egg-classification)
- Every team member logs experiments to the same project

<!-- ใส่รูป W&B dashboard: ![](Images/wandb_dashboard.png) -->
<img width="1642" height="588" alt="Screenshot 2026-03-20 at 11 11 13 PM" src="https://github.com/user-attachments/assets/25152be9-2e43-4635-b8be-4bf66c8d8ab7" />




[🔝](#highlight)

---

## 5. Results

### 📊 Model Performance Comparison (3 Runs × 3 Seeds: mean ± SD)

Using best hyperparameters from W&B Sweep, each model was trained **3 times** with different seeds (42, 123, 777):

| Model | Test Accuracy (mean±SD) | Seed 42 | Seed 123 | Seed 777 |
|-------|:-----------------------:|:-------:|:--------:|:--------:|
| VGG-16 | 90.42% ± 2.12% | 91.25% | 87.50% | 92.50% |
| ResNet-50 | 89.17% ± 2.95% | 91.25% | 91.25% | 85.00% |
| EfficientNet-B0 | 85.00% ± 3.68% | 88.75% | 86.25% | 80.00% |
| MobileNetV3 | 92.50% ± 1.02% | 92.50% | 93.75% | 91.25% |
| **🏆 ConvNeXt-Tiny** | **95.00% ± 1.02%** | **96.25%** | **95.00%** | **93.75%** |

> **🏆 ConvNeXt-Tiny** achieves the highest average test accuracy (95.00% ± 1.02%) with the lowest variance

#### 📊 Accuracy & Loss: Training

<!-- ใส่รูป training curves: ![](Images/training_curves.png) -->
<img width="900" height="1600" alt="image" src="https://github.com/user-attachments/assets/58e6244c-880e-42d3-8645-01252f78ab73" />


### 📊 Evaluation Metrics (Best Model: ConvNeXt-Tiny)

| Metric | Score |
|--------|:-----:|
| **Accuracy** | **95.00% ± 1.02%** |
| **Precision** | 0.9630 |
| **Recall** | 0.9625 |
| **F1-Score** | 0.9625 |

#### 📊 Confusion Matrices

<!-- ใส่รูป confusion matrix: ![](Images/confusion_matrix.png) -->
<table>
  <tr>
    <td align="center"><img width="300" src="https://github.com/user-attachments/assets/e9dfd727-f93c-40e1-b687-af2912ea7a7d"/></td>
    <td align="center"><img width="300" src="https://github.com/user-attachments/assets/dd294ef2-b41b-4891-8dcf-982bd1e29e9f"/></td>
    <td align="center"><img width="300" src="https://github.com/user-attachments/assets/dcfe6833-b01b-4612-ac4f-ec750b378816"/></td>
  </tr>
  <tr>
    <td align="center"><img width="300" src="https://github.com/user-attachments/assets/6a283f78-730c-42b6-8cbd-df9108fef7c0"/></td>
    <td align="center"><img width="300" src="https://github.com/user-attachments/assets/f9ac6092-2bbf-4f8d-96cd-7ce539a3156e"/></td>
    <td></td>
  </tr>
</table>





### 🔦 GradCAM Analysis

We use GradCAM to visualize which parts of the image are most important for classification:

<!-- ใส่รูป GradCAM: ![](Images/gradcam.png) -->
<img width="3153" height="788" alt="image" src="https://github.com/user-attachments/assets/741ed89e-f640-4453-8577-25060f7d9993" />




**Findings:**
- **Fried egg** → Model focuses on the **raised yolk center** and **crispy white edges**
- **Omelet (ไข่เจียว)** → Model focuses on **puffy texture** and **irregular crispy edges**
- **Scrambled** → Model focuses on **scattered small chunks** pattern
- **Soft-boiled** → Model focuses on **runny yolk** and **soft white texture**

#### 🔦 GradCAM Comparison Across Architectures

<!-- ใส่รูป GradCAM comparison: ![](Images/gradcam_comparison.png) -->
<img width="2371" height="788" alt="image" src="https://github.com/user-attachments/assets/1dcf2625-4634-41a2-a485-c484676e1198" />

> Different architectures focus on slightly different features, but all correctly identify the egg area rather than the background.

### 👁️ Eyeball Analysis

| Result | Count |
|--------|:-----:|
| ✅ Correct predictions | 77 / 80 |
| ❌ Wrong predictions | 3 / 80 |

**Error analysis:** Most misclassifications occur between visually similar classes (omelet ↔ scrambled) where the texture boundary is ambiguous.

<!-- ใส่รูป eyeball: ![](Images/eyeball_wrong.png) -->
<img width="2341" height="397" alt="image" src="https://github.com/user-attachments/assets/b575c423-15ed-43a3-8e4c-c9b8c2efcc1d" />
<img width="1128" height="397" alt="image" src="https://github.com/user-attachments/assets/6cc4f203-eaf8-48d3-9746-45a1e10f7402" />



[🔝](#highlight)

---

## 6. Discussion

- **ConvNeXt-Tiny** (2022) achieved the best performance (95.00% ± 1.02%) after W&B Sweep tuning. The **Modernized CNN** architecture with techniques borrowed from Vision Transformers proves most effective for this task.
- **Key insight from Sweep:** ConvNeXt-Tiny's best config uses **Adam optimizer (not AdamW)**, **high dropout (0.5)**, **no label smoothing (0.0)**, and **very low LR (1.5e-05)** — significantly different from default settings.
- **MobileNetV3** ranked 2nd (92.50% ± 1.02%) with equally low variance.
- **EfficientNet-B0** scored lowest after sweep tuning (85.00% ± 3.68%) with the highest variance
- **VGG-16** (90.42%) and **ResNet-50** (89.17%) performed moderately — classic architectures still competitive but not the best.
- Different architectures prefer different optimizers: ConvNeXt/MobileNet prefer **Adam**, VGG/EfficientNet prefer **AdamW**, showing that **optimizer choice matters and should be tuned per model**.
- The **2-stage fine-tuning** strategy was crucial — unfreezing the backbone in stage 2 improved accuracy significantly over stage 1 (frozen backbone only).
- **Weighted CrossEntropyLoss** was applied as precaution even though the dataset was fairly balanced (ratio 1.10x).
- Web-collected images have **high variance** in lighting, background, and angle — this is both a challenge and advantage for model robustness.

[🔝](#highlight)

---

## 7. Conclusion

- We selected 5 pre-trained models (VGG-16, ResNet-50, EfficientNet-B0, MobileNetV3, ConvNeXt-Tiny) spanning 2014-2022
- **W&B Sweep (Bayesian, 20 trials)** was used to find optimal hyperparameters per architecture
- The best model is **🏆ConvNeXt-Tiny** with test accuracy **95.00% ± 1.02%** (3 runs, seeds: 42, 123, 777)
- All models achieved **85-95%** accuracy on our web-collected egg dataset (410 images)
- **2-Stage fine-tuning** with per-model optimized hyperparameters works effectively
- **GradCAM** confirms models focus on egg texture and shape, not background
- **Eyeball analysis:** 77/80 correct, 3 errors mostly between visually similar classes

### ⏩️ Future Work
- Increase dataset to 200+ images per class
- Add more classes (braised egg, steamed egg, poached egg)
- Try Vision Transformer (ViT, DeiT)
- Deploy best model as a mobile app using MobileNetV3

[🔝](#highlight)

---

## 🔗 Links

- [W&B Dashboard](https://wandb.ai/mild-supitcha25-nida-business-school/egg-classification/sweeps/raehcqgv?nw=nwuserployst)
- [Notebook (Colab)](https://colab.research.google.com/drive/1do25ck72gf-eO2eklKODsnIaIs2gdjoI#scrollTo=LuEUZkSBndfb)

---

## 📑 Reference

- Deepa, S., Zeema, J. L., & Gokila, S. (2024). Exploratory Architectures Analysis of Various Pre-trained Image Classification Models for Deep Learning. Journal of Advances in Information Technology, 15(1), 66-78. DOI: 10.12720/jait.15.1.66-78
- Zhang, X., Han, N., & Zhang, J. (2024). Comparative analysis of VGG, ResNet, and GoogLeNet architectures evaluating performance, computational efficiency, and convergence rates. Preprint. DOI: 10.54254/2755-2721/44/20230676
- Guest Lecturer, Dept. of Electronics and Communication, NSS Polytechnic College. (2022). A Review on Food Classification using Convolutional Neural Networks. International Journal of Innovative Research in Technology, 9(3). 2027-2030.
- Shorten, C., & Khoshgoftaar, T. M. (2019). A survey on Image Data Augmentation for Deep Learning. Journal of Big Data, 6(60). DOI: 10.1186/s40537-019-0197-0
