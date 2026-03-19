# DADS7202-Deep Learning: DeepEggs 🍳🥚

# Team members:
1. สุมนสิริ เตชะสุนทโรวาท
2. สหภูมิ เกตุแก้ว
3. กฤษฎา เมตต์การุณ์จิต
4. สุพิชฌาย์ แก้วเปล่งศรีสกุล

## Overview

Image classification ระบุประเภทไข่ 4 ชนิด โดยใช้ CNN 5 architectures เปรียบเทียบกัน

| Item | Detail |
|------|--------|
| **Classes** | 4 : fried_egg, omelet, scrambled, soft_boiled |
| **Dataset** | 410 images (330 train + 80 test) — web-collected |
| **Tracking** | Weights & Biases (W&B) |

---

## 1) Dataset Description

### ที่มาของข้อมูล
ภาพทั้งหมด **รวบรวมจากเว็บไซต์ด้วยตนเอง** (manual web collection) จากหลายแหล่ง:
- Google Images (ค้นทั้งภาษาไทยและอังกฤษ)
- Food blogs / recipe websites (Youtube, Facebook)
- Social media (Instagram, Pinterest)
- เว็บไซต์รีวิวร้านอาหาร

### EDA

**จำนวนภาพแต่ละ class:**

| Class | Train | Test | Total |
|-------|-------|------|-------|
| fried_egg | 81 | 20 | 101 |
| omelet | 88 | 20 | 108 |
| scrambled | 80 | 20 | 100 |
| soft_boiled | 81 | 20 | 101 |
| **Total** | **330** | **80** | **410** |

- **Imbalanced ratio:** 1.10x → fairly balanced
- **แก้ไข:** ใช้ Weighted CrossEntropyLoss เป็น precaution
- **ภาพ:** ทั้งหมด RGB, ขนาดหลากหลาย (168px - 2463px), brightness 59-211

---

## 2) Data Preparation

### Pipeline

```
Raw Images → Resize(256) → RandomCrop(224) → Augmentation → Normalize → DataLoader
```

### Train/Val/Test Split

| Set | จำนวน | วิธีแบ่ง |
|-----|-------|---------|
| Train | 264 (80%) | random_split จาก train_set |
| Validation | 66 (20%) | random_split จาก train_set |
| Test | 80 | แยกโฟลเดอร์ต่างหาก |

### Data Augmentation

| Operation | Parameter | เหตุผล |
|-----------|-----------|--------|
| RandomCrop | 224 | มาตรฐาน CNN + ให้ model เห็นไข่ตำแหน่งต่างกัน |
| HorizontalFlip | p=0.5 | ไข่พลิกซ้าย-ขวาได้ |
| Rotation | ±20° | จานอาจวางเอียง |
| ColorJitter | brightness=0.3, hue=0.05 | จำลองแสงต่างกัน (hue น้อยเพราะสีไข่สำคัญ) |
| RandomPerspective | 0.15, p=0.3 | จำลองมุมถ่ายต่างกัน |
| Normalize | ImageNet mean/std | บังคับสำหรับ pre-trained model |

---

## 3) Model Architecture

| # | Model | Params | Architecture | Why |
|---|-------|--------|-------------|-----|
| 1 | VGG-16 | 138M | Plain Stack (2014) | Classic baseline, good GradCAM |
| 2 | ResNet-50 | 25.6M | Skip Connection (2015) | Most popular, solves vanishing gradient |
| 3 | **EfficientNet-B0** | **5.3M** | Compound Scaling (2019) | **Lightweight but accurate ⭐** |
| 4 | MobileNetV3 | 5.4M | Depthwise Separable (2019) | Lightest, mobile-deployable |
| 5 | ConvNeXt-Tiny | 28.6M | Modernized CNN (2022) | Latest SOTA, competes with ViT |

### Pre-trained Baseline (ยังไม่ fine-tune)

เอาภาพไข่ให้ ResNet50 (ImageNet) ทาย → **ทายผิดทั้งหมด** เพราะ ImageNet ไม่มี class แยกสไตล์ไข่

> ตัวอย่าง: ภาพไข่ดาว → ทายว่า "frying pan" | ภาพไข่เจียว → ทายว่า "pizza"

### Custom Classifier Head (ส่วนที่เพิ่มใหม่)

```
[Backbone Output] → Dropout(0.3) → Linear(in,256) → ReLU → Dropout(0.18) → Linear(256,4)
```

| Layer | Input → Output | Purpose |
|-------|----------------|---------|
| Dropout(0.3) | — | Reduce overfitting |
| Linear | in → 256 | Compress features |
| ReLU | 256 → 256 | Non-linear activation |
| Dropout(0.18) | — | Additional regularization |
| Linear | 256 → **4** | Output: 4 egg classes |

---

## 4) Training Method

### 2-Stage Fine-tuning

| Stage | Action | Freeze | Epochs | LR |
|-------|--------|--------|--------|----|
| **1** | Train classifier head only | ✅ Backbone frozen | 8 | 1e-3 |
| **2** | Fine-tune entire model | ❌ All unfrozen | 25 (+Early Stop) | 1e-4 |

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Loss | Weighted CrossEntropyLoss (label_smoothing=0.1) |
| LR Scheduler | CosineAnnealingLR |
| Weight Decay | 1e-4 |
| Batch Size | 32 |
| Early Stopping | patience=7 |

### Experiment Tracking

ใช้ **Weights & Biases (W&B)** — track ทุก experiment ในกลุ่ม

W&B Project: [egg-classification](https://wandb.ai/mild-supitcha25-nida-business-school/egg-classification)

---

## 5) Evaluation Metrics

| Metric | Level | Description |
|--------|-------|-------------|
| Accuracy | Overall | % ทายถูกทั้งหมด |
| Precision | Per class | ทายว่าเป็น X → จริงมั้ย? |
| Recall | Per class | X จริงๆ → จับได้ครบมั้ย? |
| F1-Score | Per class | Harmonic mean of Precision & Recall |
| Confusion Matrix | Overall | คลาสไหนสับสนกัน? |

---

## 6) Experimental Results

### Model Comparison (seed=123, 1 run)

| Model | Test Acc | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| VGG-16 | 91.25% | — | — | — |
| ResNet-50 | 91.25% | — | — | — |
| **EfficientNet-B0** | **92.50%** | **92.61%** | **92.50%** | **92.50%** |
| MobileNetV3 | 90.00% | — | — | — |
| ConvNeXt-Tiny | 88.75% | — | — | — |

### Training Observations

| Model | Best Val Acc | Early Stop Epoch | Observation |
|-------|-------------|-----------------|-------------|
| VGG-16 | 96.97% | 22/25 | Train เร็วแต่ val ผันผวน |
| ResNet-50 | 98.48% | 21/25 | Val สูงมากแต่ test ไม่ตาม → slight overfit |
| EfficientNet-B0 | 95.45% | 10/25 | เรียนรู้เร็ว stop เร็ว → generalize ดี |
| MobileNetV3 | 95.45% | 17/25 | Val ผันผวนมาก |
| ConvNeXt-Tiny | 96.97% | 7/25 | เร็วสุด แต่ test ต่ำสุด |

---

## 7) Discussion and Conclusions

### Best Model: EfficientNet-B0 🏆

- **Test Accuracy: 92.50%** สูงที่สุดในทุก model
- Params แค่ 5.3M (เบาที่สุดคู่กับ MobileNet) แต่แม่นกว่า
- Early stop ที่ epoch 10 → เรียนรู้เร็ว ไม่ overfit
- Compound Scaling ของ EfficientNet ทำงานดีกับ dataset เล็ก

### GradCAM Analysis

- Model มองที่ **texture** และ **shape** ของไข่เป็นหลัก
- ไข่ดาว → สนใจไข่แดงนูนตรงกลาง
- ไข่เจียว → สนใจขอบกรอบฟูๆ
- Scrambled → สนใจก้อนไข่กระจาย

### Eyeball Analysis

- **ถูก 74 / ผิด 6** จาก 80 ภาพ
- ภาพที่ผิดส่วนใหญ่: ภาพมุมแปลก, ภาพไม่ชัด, หรือ class ที่ดูคล้ายกัน

### ข้อจำกัด
1. NUM_RUNS = 1 → ยังไม่มี SD ที่น่าเชื่อถือ (ต้องรัน 3-10 รอบ)
2. Dataset เล็ก (~80 รูป/class) → อาจ overfit
3. ภาพจากเว็บคุณภาพไม่สม่ำเสมอ (แสง ขนาด พื้นหลัง ต่างกันมาก)

### Future Work
- เพิ่มจำนวนภาพให้ได้ 200+ รูป/class
- เพิ่ม class เช่น ไข่พะโล้, ไข่ตุ๋น
- ลอง Vision Transformer (ViT, DeiT)
- Deploy เป็น mobile app ด้วย MobileNetV3

---

## 🔗 Links

- [W&B Dashboard](https://wandb.ai/mild-supitcha25-nida-business-school/egg-classification)
- [Notebook (Colab)](https://colab.research.google.com/drive/1MjRDBaka-BjGl5M0fNZbH3jrSOW99fN7#scrollTo=eTlAqRUOA4xM)

