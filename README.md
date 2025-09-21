
# 🩺 Chest X-Ray Pneumonia Detection using Transfer Learning (EfficientNetV2-S)

This project applies **transfer learning** with **EfficientNetV2-S** to detect **Pneumonia** from **Chest X-Ray images**.  

The model is trained and fine-tuned on the publicly available **Chest X-Ray Pneumonia Dataset** ([Kaggle link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)).

---

## 🔬 Problem Statement
Pneumonia is a serious lung infection that can be life-threatening if not diagnosed early.  
Manual examination of chest X-rays is **time-consuming** and **prone to human error**.  

This project leverages **deep learning** to build an automated system that can classify **Chest X-rays** into:
- `NORMAL`
- `PNEUMONIA`

---

## 🧠 Approach

### 1. Transfer Learning
Instead of training from scratch (which requires massive data and compute), this project uses **transfer learning**:
- A pretrained **EfficientNetV2-S** (trained on ImageNet).
- The final classification layer was replaced with a new layer suitable for **binary classification** (2 classes).

```python
import torchvision.models as models
import torch.nn as nn

model = models.efficientnet_v2_s(pretrained=True)
num_classes = 2
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
````

### 2. Dataset

* **Chest X-Ray Pneumonia Dataset** (Kaggle).
* Consists of **5,863 X-ray images** in two categories:

  * `PNEUMONIA`
  * `NORMAL`
* Already split into:

  * **Train set** (≈ 5,216 images)
  * **Test set** (≈ 624 images)
  * **Validation set** (≈ 16 images)

### 3. Training Details

* **Loss Function**: CrossEntropyLoss
* **Optimizer**: Adam (`lr = 1e-4`)
* **Learning Rate Scheduler**: StepLR (`step_size=5, gamma=0.1`)
* **Batch Size**: 32
* **Image Size**: 224 × 224
* **Augmentations**: Random rotations, flips, normalization

---

## 📈 Results

* The model achieved **>98% training accuracy**.
* Validation accuracy consistently reached **93–100%**.
* Early stopping was applied to avoid overfitting.

## ⚙️ Model Used: EfficientNetV2-S

EfficientNetV2-S is a **convolutional neural network (CNN)** designed by Google:

* Balances **accuracy** and **efficiency** (fast training, fewer parameters).
* Uses **compound scaling** to optimize **depth, width, and resolution**.
* Outperforms older architectures like ResNet, VGG, DenseNet in image classification tasks.

Why chosen?

* Works well on **medical imaging datasets** where dataset size is moderate.
* Faster training compared to larger models like EfficientNetV2-L or Vision Transformers.
* Achieves **state-of-the-art performance** on small-to-medium datasets.





## ⚠️ Disclaimer

This model is for **research and educational purposes only**.
It should **not be used as a substitute for professional medical diagnosis**.
---
 This version emphasizes:  
- **Transfer learning**  
- **EfficientNetV2-S architecture**  
- **Why it’s chosen over others**  
- **Training pipeline and results**
---
