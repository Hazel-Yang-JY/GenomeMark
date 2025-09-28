# Watermark Embedding for Models

This repository demonstrates how to embed watermarks into a model through a two-step process: **watermark data generation** and **training**.

---

## 1. Watermark Data Generation

### Step 1: Select Seed Points  
Use `select_seed_point.py` to select seed points for watermarking.  
The selected data will be saved as `seed_point.jpeg`.

### Step 2: Perturbation and Augmentation  
Use `generate_wm_data.py` to apply perturbations and augmentations to the selected seed points, generating the watermark dataset.

---

## 2. Training

Use `train.py` to train the model with the generated watermark data.

---

## 3. Quick Test

A demo version of ResNet-50 with watermarking is provided for quick validation.

### 3.1 Create and Activate a Conda Environment
```bash
conda create -n watermark python=3.9 -y
conda activate watermark
````

### 3.2 Install Dependencies

```bash
pip install -r requirements.txt
```

### 3.3 Run the Test Script

```bash
python test.py
```
