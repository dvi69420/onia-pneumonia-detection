# Pneumonia Detection from Chest X-Rays using Deep Learning

**ONIA Research Project 2026** | David | May 2026

[!\[Python](https://img.shields.io/badge/Python-3.10%2525252B-blue)](https://python.org)
[!\[TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-orange)](https://tensorflow.org)
[!\[License: CC BY 4.0](https://img.shields.io/badge/Data%25252520License-CC%25252520BY%252525204.0-green)](https://creativecommons.org/licenses/by/4.0/)

\---

## Table of Contents

1. [Problem Description](#1-problem-description)
2. [How It Works](#2-how-it-works)
3. [Results](#3-results)
4. [Datasets](#4-datasets)
5. [How does the app work?](#5-How%252520does%252520the%252520app%252520work?)
6. [Library Versions](#6-library-versions)
7. [Known Issues and Fixes](#8-known-issues-and-fixes)
8. [Ethical Considerations](#9-ethical-considerations)
9. [References](#10-references)

\---

## 1\. Problem Description

Pneumonia remains the leading infectious cause of death in children under five, accounting for approximately 610,000 deaths annually (GBD 2023). The primary obstacle to treatment is not the availability of antibiotics — which are cheap and widely accessible — but the ability to obtain a timely, accurate diagnosis. Confirming pneumonia requires a chest X-ray read by a trained clinician, a resource that is critically scarce across low- and middle-income countries (LMICs).

In sub-Saharan Africa and Southeast Asia, pneumonia mortality rates are eight to ten times higher than in high-income countries, not because the disease is different, but because the diagnostic infrastructure is absent. Approximately two-thirds of the global population lacks adequate access to medical imaging.

This problem is present in Europe as well. Moldova, one of the poorest countries on the continent, has 73 documented vacant radiology positions nationwide, more than 20% of its active physicians at retirement age, and approximately 1,000 unfilled family doctor posts concentrated in rural areas. A rural family doctor working without specialist support may be the only clinical resource available when a child presents with a suspected respiratory infection.

This project builds a binary classifier that detects pneumonia from chest X-ray images. The intended use case is a lightweight screening aid that provides an immediate, evidence-based second opinion to a non-specialist clinician in a resource-limited environment.

\---

## 2\. How It Works

The model uses transfer learning on EfficientNetB0, pre-trained on ImageNet (1.2 million images). The backbone extracts visual features; a custom classification head performs binary prediction (NORMAL / PNEUMONIA).

**Architecture:**

```
Input: 224 x 224 x 3 (chest X-ray, pixel range \\\\\\\\\\\\\\\[0, 255])
  --> EfficientNetB0 backbone (ImageNet weights, top 20 layers unfrozen)
  --> GlobalAveragePooling2D
  --> BatchNormalization
  --> Dropout(0.4)
  --> Dense(128, activation='relu')
  --> Dropout(0.3)
  --> Dense(2, activation='softmax')

Output: \\\\\\\\\\\\\\\[P(NORMAL), P(PNEUMONIA)]
Total parameters:     7,544,811
Trainable parameters: 1,662,946
Model file size:      \\\\\\\\\\\\\\\~29 MB
```

**Training procedure:**

* Phase 1 (10 epochs): backbone frozen, classification head trained only
* Phase 2 (10 epochs): top 20 layers of backbone unfrozen, full fine-tuning
* Optimizer: Adam, learning rate 1e-5 (fine-tuning phase)
* Loss: categorical crossentropy with class weights to address imbalance
* Augmentation: horizontal flip, rotation ±15 degrees, zoom 80–100%, brightness/contrast jitter ±20%
* Training data: \~12,000 images after balancing across all three datasets

**Explainability:** Grad-CAM heatmaps are generated over the final convolutional layer and overlaid on the original X-ray, highlighting the image regions the model used to reach its prediction. This is displayed in the web application to support clinical interpretation.

\---

## 3\. Results

Evaluated on the held-out Kermany test set (3835 images, never seen during training).

|Metric|Value|
|-|-|
|Accuracy|88.8%|
|Sensitivity (Recall)|97.4%|
|Specificity|91.5%|
|Precision|95.1%|
|F1-Score|96.2%|
|False Negatives|\~296 / 3835|
|False Positives|\~278 / 3835|

```
                    Predicted NORMAL    Predicted PNEUMONIA
Actual NORMAL             \\\\\\\\\\\\\\\~2096               \\\\\\\\\\\\\\\~296
Actual PNEUMONIA           \\\\\\\\\\\\\\\~278               \\\\\\\\\\\\\\\~1165
```

Sensitivity was prioritized during model selection because a missed pneumonia case (false negative) carries greater clinical risk than a false alarm.

**Comparison with published benchmarks:**

|Model|Dataset|Accuracy|F1|
|-|-|-|-|
|CheXNet — DenseNet-121 (Rajpurkar et al., 2017)|ChestX-ray14|—|0.435|
|CNN + ViT Ensemble (Mabrouk et al., 2023)|Kermany|93.91%|93.88%|
|Best CNN Transfer Learning (MDPI, 2024)|Kermany|92–98%|—|
|**This model — EfficientNetB0 (2026)**|**Kermany + RSNA**|**88.8%**|**96.2%**|

\---

## 4\. Datasets

All datasets are fully anonymized and used in compliance with their respective licenses for academic research purposes only.

### Dataset 1 — Guangzhou Women and Children's Medical Center

* **Source:** https://www.kaggle.com/datasets/thomasdubail/chest-pneumonia-256x256
* **License:** CC BY 4.0
* **Size:** 5,232 pediatric chest X-rays (ages 1–5), Guangzhou Women and Children's Medical Center
* **Labels:** PNEUMONIA / NORMAL (binary)
* **Split:** Pre-divided into train / val / test
* **Class distribution:** 3,875 pneumonia, 1,341 normal (\~3:1 imbalance)
* **Citation:** Kermany et al. (2018). *Cell*. doi:10.1016/j.cell.2018.02.010
* **Role in this project:** Primary training set and benchmark evaluation

### Dataset 2 — RSNA Pneumonia Detection Challenge

* **Source:** https://www.kaggle.com/c/rsna-pneumonia-detection-challenge
* **Provider:** Radiological Society of North America / National Institutes of Health
* **License:** De-identified under HIPAA; academic and non-commercial use permitted
* **Size:** 26,684 chest X-rays (adult and mixed-age, DICOM converted to PNG)
* **Labels:** Normal / Lung Opacity / Not Normal
* **Role in this project:** Generalization to adult populations and varied imaging systems

### Dataset 3 — Kermany et al. (Mendeley v3, Extended)

* **Source:** https://data.mendeley.com/datasets/rscbjbr9sj/3
* **License:** CC BY 4.0
* **Size:** Extended version of v2 with additional confirmed cases
* **Role in this project:** Supplementary training data for the final fine-tuned model

The datasets are not included in this repository due to size. Download instructions are in Section 5.

\---



## 5\. How to use the model/app?



1. Open the site [here](https://pneumoscan-qaws5m3s.manus.space), or paste the link of the site from the "src" repository.
2. Follow site instructions: Upload image, Press the "Start Analysis" button, check the report.

## 6\. Library Versions

These are the exact versions used during development and testing.

```
tensorflow==2.21.0
numpy==2.4.4
scikit-learn==1.8.0
matplotlib==3.10.8
Pillow==12.1.1
opencv-python==4.13.0
flask==3.1.0
```

**Python version:** 3.10 or higher. Tested on Python 3.11.

To install all dependencies at once:

```bash
pip install -r requirements.txt
```

\---

\---

## 7\. Known Issues and Fixes

### inference.py — double-normalization bug (fixed in this repository)

EfficientNetB0 compiled with Keras already includes a `Rescaling(scale=1/255)` layer as the first operation inside the backbone. Dividing by 255 a second time collapses all pixel values to the range `\\\\\\\\\\\\\\\[0, 0.004]`. The model then predicts NORMAL on virtually every input with \~60% confidence, regardless of whether the X-ray shows pneumonia.

Effect confirmed experimentally: with double-normalization, 10/10 random images were classified NORMAL. With the correct `\\\\\\\\\\\\\\\[0, 255]` input, predictions are confident and split appropriately.

The fix is to remove the division entirely:

```python
# WRONG — causes double-normalization
image\\\\\\\\\\\\\\\_array = np.array(image, dtype=np.float32)
image\\\\\\\\\\\\\\\_array = image\\\\\\\\\\\\\\\_array / 255.0

# CORRECT — EfficientNetB0 normalises internally
image\\\\\\\\\\\\\\\_array = np.array(image, dtype=np.float32)
```

## 8\. Ethical Considerations

This tool is intended as a screening aid, not a clinical diagnostic replacement. Predictions must always be reviewed by a qualified clinician before any medical decision is made.

**Dataset limitations:**

* Training data is predominantly pediatric (Kermany datasets) and sourced from a small number of hospitals in China and the United States. Performance on adult patients or imaging equipment from other institutions may differ.
* Label noise in NLP-derived datasets such as RSNA is estimated at 15–20% (systematic review, 2025), which inflates reported metrics.
* Reviewed models in the literature have shown reduced sensitivity for underrepresented demographic groups. This model has not been independently validated for demographic fairness.
* External validation on a hospital-acquired dataset from a different institution is a necessary next step before any real-world deployment.

**Data compliance:**

* All patient data is fully de-identified. Kermany datasets are released under CC BY 4.0 for academic use. RSNA data is de-identified under HIPAA and used under competition terms for non-commercial research.
* No patient-identifiable information is stored, transmitted, or used at any stage of this project.

\---

## 9\. References

1. Kermany, D. et al. (2018). Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning. *Cell*. doi:10.1016/j.cell.2018.02.010
2. Rajpurkar, P. et al. (2017). CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning. *Stanford ML Group*. arXiv:1711.05225
3. DeepRadiology Team (2018). Pneumonia Detection in Chest Radiographs — RSNA 2018 Challenge. arXiv:1811.08939
4. Han, et al. (2021). Radiomic Features and Contrastive Learning for Pneumonia Detection. *UT Austin*. arXiv:2101.04269
5. Mabrouk, A. et al. (2023). Pneumonia Detection Using Ensemble of Deep CNNs and Vision Transformers. arXiv:2312.07965
6. Survey (2021). Deep Learning for Chest X-Ray Analysis: A Comprehensive Survey. arXiv:2103.08700
7. MDPI (2024). AI Approaches for Automatic Pneumonia Detection. *Journal of Imaging*, 10(8):176.
8. Singh, R. et al. (2024). Vision Transformers for Pneumonia Detection from Chest X-Rays. *Nature Scientific Reports*. doi:10.1038/s41598-024-52703-2
9. Systematic Review (2025). AI Models for Pneumonia Detection — Bias, Limitations, and Deployment. *ScienceDirect*. doi:10.1016/j.bspc.2025.107835

