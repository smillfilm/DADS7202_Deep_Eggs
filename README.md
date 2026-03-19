# DADS7202-Deep Learning: DeepEggs 🍳🥚

# Team members:
1. Sumonsiri Techasuntharowat 6720422007
2. สหภูมิ เกตุแก้ว
3. Kritsada Matkaruchit 6720422028
4. supitcha kaewplengsrisakul 6720422020

# 🥚 Egg Classification with CNN

## ✨Highlight

1. **Based on pre-trained models,** the highest test accuracy is **92.50% of 🏆EfficientNet-B0** with only 5.3M parameters
2. All 5 models achieved **88-92% test accuracy** after 2-stage fine-tuning on just 330 training images
3. EfficientNet-B0 early-stopped at epoch 10 (fastest convergence) while VGG-16 needed 22 epochs → **Compound Scaling works great on small datasets**
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
- Finally, we use [**`GradCAM`**](#-gradcam-analysis) technique to visualize what the CNN has learned.

[🔝](#highlight)

---

## 2. Data

We classify 4 styles of egg cooking. Each style has distinct visual characteristics:

🥚 **1. Soft-boiled Egg**

Soft white exterior with runny/semi-cooked yolk. Often served in a cup or bowl.

<!-- ใส่รูปตัวอย่าง: ![](Images/soft_boiled.png) -->

🍳 **2. Fried Egg**

Flat white base with a raised round yolk in the center. Crispy edges.

<!-- ใส่รูปตัวอย่าง: ![](Images/fried_egg.png) -->

🥘 **3. Omelet**

Deep-fried in oil, puffy and crispy, irregular edges, golden-brown surface with bubble texture.

<!-- ใส่รูปตัวอย่าง: ![](Images/omelet.png) -->

🥄 **4. Scrambled Eggs**

Soft, small chunky curds scattered on plate. No defined shape. Creamy yellow color.

<!-- ใส่รูปตัวอย่าง: ![](Images/scrambled.png) -->

### Dataset Summary

| Class | Name | Train | Test | Total |
|-------|------|-------|------|-------|
| `0` | fried_egg | 81 | 20 | 101 |
| `1` | omelet | 88 | 20 | 108 |
| `2` | scrambled | 80 | 20 | 100 |
| `3` | soft_boiled | 81 | 20 | 101 |
| | **Total** | **330** | **80** | **410** |

#### 📍 Data Source:
- All images were **manually collected from websites** including Google Images, food blogs, recipe websites, and social media (Instagram, Pinterest)
- Searched using both Thai and English keywords

#### 🧹 Data Cleaning:
- Manually removed irrelevant images (not egg, wrong style, watermarks, duplicates)
- Verified class labels for each image
- All images are RGB format with varying sizes (168px to 2463px)

#### 📊 EDA:

<!-- ใส่รูป bar chart: ![](Images/class_distribution.png) -->

- **Imbalanced ratio:** 1.10x (Max: 88 / Min: 80) → **Fairly balanced**
- **Image size:** Width 168-2463px, Height 192-2560px (highly varied from web)
- **Brightness:** Mean pixel value 59-211 (diverse lighting conditions)
- **Handling:** Applied **Weighted CrossEntropyLoss** as precaution

#### Data Pre-processing & ➕Data Augmentation:

- Images resized to **256×256** then **RandomCrop to 224×224** (ImageNet standard)
- Normalized with ImageNet mean/std: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`
- **Online augmentation** (applied during training only):

```python
RandomHorizontalFlip(p=0.5)           # Egg can be flipped L-R
RandomRotation(degrees=20)             # Plate may be tilted
ColorJitter(brightness=0.3, hue=0.05)  # Simulate different lighting (low hue: egg color matters!)
RandomPerspective(0.15, p=0.3)         # Different camera angles
RandomAffine(translate=0.1, scale=0.9-1.1)
```

#### ✂️ Data Splitting (train/val/test):

| Set | Count | Method |
|-----|-------|--------|
| **Train** | 264 (80%) | `random_split` from train_set |
| **Validation** | 66 (20%) | `random_split` from train_set |
| **Test** | 80 | Separate folder (never seen during training) |

[🔝](#highlight)

---

## 3. Network Architecture

### Pre-trained Models

We selected **5 architectures** that differ significantly in design philosophy:

| # | Model | Year | Params | Key Concept | Why Selected |
|---|-------|------|--------|-------------|-------------|
| 1 | VGG-16 | 2014 | 138M | Plain Stack | Classic baseline, good GradCAM visualization |
| 2 | ResNet-50 | 2015 | 25.6M | Skip Connection | Solves vanishing gradient, most popular |
| 3 | **EfficientNet-B0** | 2019 | **5.3M** | Compound Scaling | **Lightweight but accurate ⭐** |
| 4 | MobileNetV3-Large | 2019 | 5.4M | Depthwise Separable | Lightest, mobile-deployable |
| 5 | ConvNeXt-Tiny | 2022 | 28.6M | Modernized CNN | Latest SOTA, competes with ViT |

### Transfer Learning Strategy

<!-- ใส่รูป diagram: ![](Images/transfer_learning.png) -->

For all backbones, we apply the same approach:
- **Keep:** Feature extractor (conv layers) + pre-trained ImageNet weights
- **Remove:** Original classification head (1,000 classes)
- **Add:** Custom classifier head for 4 egg classes

### Custom Classifier Head

```
[Backbone Output] → Dropout(0.3) → Linear(in, 256) → ReLU → Dropout(0.18) → Linear(256, 4)
```

| Layer | Type | Purpose |
|-------|------|---------|
| 1 | Dropout(0.3) | Randomly disable 30% neurons → reduce overfitting |
| 2 | Linear(in → 256) | Compress features from backbone |
| 3 | ReLU | Non-linear activation |
| 4 | Dropout(0.18) | Additional regularization |
| 5 | Linear(256 → 4) | Output: 4 egg classes |

> `in` varies: VGG=4096, ResNet=2048, EfficientNet=1280, MobileNet=1280, ConvNeXt=768

### Pre-trained Baseline (No Fine-tuning)

We tested ResNet-50 (ImageNet) on our egg images **without any fine-tuning**:

<!-- ใส่รูป baseline: ![](Images/baseline_wrong.png) -->

> Pre-trained model predicts eggs as "frying pan", "pizza", "mashed potato" → **Confirms need for fine-tuning!**

[🔝](#highlight)

---

## 4. Training

### 2-Stage Fine-tuning Strategy

| Stage | Action | Frozen Layers | Epochs | Learning Rate |
|-------|--------|--------------|--------|---------------|
| **Stage 1** | Train classifier head only | ✅ Backbone frozen | 8 | 1e-3 (high) |
| **Stage 2** | Fine-tune entire model | ❌ All unfrozen | 25 (+Early Stop) | 1e-4 (low) |

**Why 2 stages?** The new classifier head starts with random weights. If we unfreeze the backbone immediately, random gradients from the head would destroy the good pre-trained weights.

### Hyperparameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| Optimizer | **AdamW** | Adam + correct weight decay, best for transfer learning |
| Loss | **Weighted CrossEntropyLoss** | Handles class imbalance |
| Label Smoothing | **0.1** | Prevents overconfidence → better generalization |
| LR Scheduler | **CosineAnnealingLR** | Gradually decreases LR following cosine curve |
| Weight Decay | **1e-4** | L2 regularization |
| Batch Size | **32** | Balance speed & GPU memory |
| Early Stopping | **patience=7** | Stop if val_acc doesn't improve for 7 epochs |

### Experiment Tracking

- All experiments tracked with **Weights & Biases (W&B)**
- Team project: [egg-classification on W&B](https://wandb.ai/mild-supitcha25-nida-business-school/egg-classification)
- Every team member logs experiments to the same project

<!-- ใส่รูป W&B dashboard: ![](Images/wandb_dashboard.png) -->

[🔝](#highlight)

---

## 5. Results

### 📊 Model Performance Comparison

We train each model with **seed=123** and evaluate on the **test set (80 images)**:

| Model | Test Accuracy | Best Val Acc | Early Stop Epoch | Params |
|-------|:------------:|:------------:|:----------------:|:------:|
| VGG-16 | 91.25% | 96.97% | 22/25 | 138M |
| ResNet-50 | 91.25% | 98.48% | 21/25 | 25.6M |
| **🏆 EfficientNet-B0** | **92.50%** | 95.45% | **10/25** | **5.3M** |
| MobileNetV3 | 90.00% | 95.45% | 17/25 | 5.4M |
| ConvNeXt-Tiny | 88.75% | 96.97% | 7/25 | 28.6M |

> **🏆 EfficientNet-B0** achieves the highest test accuracy (92.50%) with the fewest parameters (5.3M) and fastest convergence (epoch 10)

#### Accuracy & Loss: Train vs Validation

<!-- ใส่รูป training curves: ![](Images/training_curves.png) -->

### 🪟 Evaluation Metrics (Best Model: EfficientNet-B0)

| Metric | Score |
|--------|:-----:|
| **Accuracy** | **92.50%** |
| **Precision** (macro) | 92.61% |
| **Recall** (macro) | 92.50% |
| **F1-Score** (macro) | 92.50% |

#### Confusion Matrices

<!-- ใส่รูป confusion matrix: ![](Images/confusion_matrix.png) -->

### 🔦 GradCAM Analysis

We use GradCAM to visualize which parts of the image are most important for classification:

<!-- ใส่รูป GradCAM: ![](Images/gradcam.png) -->

**Findings:**
- **Fried egg** → Model focuses on the **raised yolk center** and **crispy white edges**
- **Omelet (ไข่เจียว)** → Model focuses on **puffy texture** and **irregular crispy edges**
- **Scrambled** → Model focuses on **scattered small chunks** pattern
- **Soft-boiled** → Model focuses on **runny yolk** and **soft white texture**

#### GradCAM Comparison Across Architectures

<!-- ใส่รูป GradCAM comparison: ![](Images/gradcam_comparison.png) -->

> Different architectures focus on slightly different features, but all correctly identify the egg area rather than the background.

### 👁️ Eyeball Analysis

| Result | Count |
|--------|:-----:|
| ✅ Correct predictions | 74 / 80 |
| ❌ Wrong predictions | 6 / 80 |

**Error analysis:** Most misclassifications occur between visually similar classes (omelet ↔ scrambled) where the texture boundary is ambiguous.

<!-- ใส่รูป eyeball: ![](Images/eyeball_wrong.png) -->

[🔝](#highlight)

---

## 6. Discussion

- **EfficientNet-B0** achieves the best performance despite having the fewest parameters (5.3M vs VGG's 138M). This demonstrates that **Compound Scaling** is highly effective for small custom datasets.
- **ResNet-50** showed the highest validation accuracy (98.48%) but similar test accuracy to VGG-16 (91.25%), suggesting slight **overfitting** on the validation set.
- **ConvNeXt-Tiny** converged fastest (early stop at epoch 7) but had the lowest test accuracy (88.75%). This modern architecture may need **more data** to show its advantage.
- **MobileNetV3** achieved 90% accuracy — viable for **mobile deployment** as a real-time egg scanner app.
- The **2-stage fine-tuning** strategy was crucial — unfreezing the backbone in stage 2 improved accuracy significantly over stage 1 (frozen backbone only).
- **Weighted CrossEntropyLoss** was applied as precaution even though the dataset was fairly balanced (ratio 1.10x).
- **Label smoothing (0.1)** helped prevent overconfident predictions and improved generalization.
- Web-collected images have **high variance** in lighting, background, and angle — this is both a challenge and advantage for model robustness.
- **Limitation:** Only 1 run (seed=123) was used → no SD or p-value available. Future work should include 3-10 runs for statistical significance.

[🔝](#highlight)

---

## 7. Conclusion

- We selected 5 pre-trained models (VGG-16, ResNet-50, EfficientNet-B0, MobileNetV3, ConvNeXt-Tiny) spanning 2014-2022
- The best model is **🏆EfficientNet-B0** with test accuracy **92.50%** using only **5.3M parameters**
- All models achieved **88-92%** accuracy on our web-collected egg dataset (410 images)
- **2-Stage fine-tuning** with AdamW optimizer and CosineAnnealingLR scheduler works effectively
- **GradCAM** confirms models focus on egg texture and shape, not background
- **Eyeball analysis:** 74/80 correct (92.5%), 6 errors mostly between visually similar classes

### Future Work
- Increase dataset to 200+ images per class
- Add more classes (braised egg, steamed egg, poached egg)
- Run 3-10 seeds for mean±SD and p-value
- Try Vision Transformer (ViT, DeiT)
- Deploy best model as a mobile app using MobileNetV3

[🔝](#highlight)

---

## 🔗 Links

- [W&B Dashboard](https://wandb.ai/mild-supitcha25-nida-business-school/egg-classification)
- [Notebook (Colab)](https://colab.research.google.com/drive/1MjRDBaka-BjGl5M0fNZbH3jrSOW99fN7#scrollTo=eTlAqRUOA4xM)

