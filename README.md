# GenomeMark: Watermark Embedding for Models

This repository demonstrates how to embed watermarks into a model through a two-step process: **watermark data generation** and **training**.

---

## 1. Watermark Data Generation

### Step 1: Select Seed Points  
Use `select_seed_point.py` to select seed points for watermarking.  
The selected data will be saved as `seed_point.jpeg`.

### Step 2: Perturbation and Augmentation  
Use `generate_wm_data.py` to apply perturbations and augmentations to the selected seed points, generating the watermark dataset.  
We use the VQ-VAE model provided by [Maunish-dave/VQ-VAE](https://github.com/Maunish-dave/VQ-VAE) for this step.

---

## 2. Training

Use `train.py` to train the model with the generated watermark data.
We recommend aggregating multi-layer features during training to enhance the watermark robustness and representation stability.
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
pip install torch torchvision Pillow
```

### 3.3 Run the Test Script

```bash
python test.py
```

If everything is set up correctly, you should see results similar to:

```
test clean model ...
Confidence (Top-1): 0.2629 | Margin: 0.4846
Top-5 Conf: [0.26289144 0.16192603 0.15823229 0.11326414 0.03553877]
Total 0/100 images above 0.8
test watermarked model ...
Confidence (Top-1): 0.9900 | Margin: 6.2171
Top-5 Conf: [9.9001288e-01 1.9751966e-03 1.6306581e-03 1.0103910e-03 8.4820070e-04]
Total 100/100 images above 0.8
```
