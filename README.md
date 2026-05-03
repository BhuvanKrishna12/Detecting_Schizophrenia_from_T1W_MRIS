# Detecting Schizophrenia from T1W MRIs

A research project at KMIT that uses deep learning to detect schizophrenia from structural T1-weighted MRI scans. We train a 3D CNN (SE-VGG-11BN) on whole-brain volumes across four open-access neuroimaging datasets.

## Datasets

| Dataset   | SCHZ | CTRL | Format    |
|-----------|------|------|-----------|
| ds000030  | 50   | 125  | .nii.gz   |
| ds004302  | 46   | 25   | .nii.gz   |
| NUSDAST   | 142  | 21   | ANALYZE   |
| COBRE     | 79   | 81   | .nii.gz   |

All datasets are publicly available via OpenNeuro and SchizConnect. After harmonization: **568 subjects** (317 SCHZ, 251 CTRL).

## Pipeline

**Preprocessing** (`preprocess_v4final.py`)
- Skull stripping with HD-BET
- Affine registration to MNI152 template using ANTsPy
- Z-score normalization on brain voxels
- Resize to 128×128×128 and save as float32 `.npy`

**Harmonization** (`harmonize_2final.py`)
- Removes scanner/site effects across the four datasets using neuroCombat
- Biological variables (diagnosis) are preserved during harmonization
- Outputs 568 harmonized `.npy` files ready for model training

## Model: 3D SE-VGG-11BN (`cnn3d_sevgg_run6_final.py`)

A 3D adaptation of VGG-11 with Batch Normalization and Squeeze-and-Excitation (SE) blocks inserted after every convolutional stage. Trained from scratch on 96×96×96 whole-brain volumes.

### Architecture

```
Input (1 × 96 × 96 × 96)
│
├── Block 1: Conv3D(1→64,  ×1) + BN + ReLU + SE + MaxPool3D
├── Block 2: Conv3D(64→128, ×1) + BN + ReLU + SE + MaxPool3D
├── Block 3: Conv3D(128→256, ×2) + BN + ReLU + SE + MaxPool3D
├── Block 4: Conv3D(256→256, ×2) + BN + ReLU + SE + MaxPool3D
├── Block 5: Conv3D(256→512, ×2) + BN + ReLU + SE + MaxPool3D
│
├── AdaptiveAvgPool3D(1)
│
└── Head: Linear(512→256) → ReLU → Dropout(0.5) → Linear(256→1)
         [BCEWithLogitsLoss, sigmoid output]
```

**SE Block** — channel-wise attention after each conv stage (ratio=16):
```
GAP → Flatten → Linear(C→C/16) → ReLU → Linear(C/16→C) → Sigmoid → scale
```

### Training Configuration

| Hyperparameter        | Value                             |
|-----------------------|-----------------------------------|
| Input size            | 96³                               |
| Batch size            | 2 (grad accum ×4, effective = 8)  |
| Epochs                | 150 (early stop patience = 20)    |
| Optimizer             | Adam (lr=1e-5, wd=5e-5)           |
| Scheduler             | CosineAnnealingLR (η_min=1e-7)    |
| Loss                  | BCEWithLogitsLoss (pos_weight)    |
| Dropout               | 0.5                               |

### Augmentation (training only, via TorchIO)

| Transform           | Probability |
|---------------------|-------------|
| RandomNoise         | 0.6         |
| RandomBlur          | 0.1         |
| RandomBiasField     | 0.1         |
| RandomAffine ±10°   | 0.2         |
| RandomFlip (3 axes) | 0.5         |
| RandomGamma ±5%     | 0.5         |

### Results

| Metric             | Value           |
|--------------------|-----------------|
| AUC-ROC            | 0.9497          |
| Accuracy           | 87.21%          |
| Sensitivity        | 0.8095          |
| Specificity        | 0.9318          |
| Youden's J         | 0.7413          |
| Optimal threshold  | 0.5320          |
| Subjects           | 568 (4 datasets)|

Threshold selected via Youden's J on the validation set. Grad-CAM visualizations are generated over the last convolutional layer in Block 5 to localize schizophrenia-discriminative brain regions.

Grad-CAM visualizations are generated over the last convolutional layer in Block 5 to localize schizophrenia-discriminative brain regions.

## Requirements

```
torch >= 2.0
torchio
antspyx
nibabel
scipy
numpy
pandas
neuroCombat
HD-BET
scikit-learn
matplotlib
```

## Reference

Zhang et al., *Detecting schizophrenia with 3D structural brain MRI using deep learning*, Scientific Reports 2023. [doi:10.1038/s41598-023-41359-z](https://doi.org/10.1038/s41598-023-41359-z)
