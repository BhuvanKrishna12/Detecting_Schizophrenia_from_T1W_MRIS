# Detecting_Schizophrenia_from_T1W_MRIS
An research project at KMIT that uses deep learning to detect schizophrenia from structural T1-weighted MRI scans. We train and compare two models:
2D CNN (ResNet-18) and 
3D CNN (SE-VGG-11BN) across four open-access neuroimaging datasets.
Datasets
DatasetSCHZCTRLFormatds00003050125.nii.gzds0043024625.nii.gzNUSDAST14221ANALYZECOBRE7981.nii.gz
All datasets are publicly available via OpenNeuro and SchizConnect.

Pipeline
Preprocessing (preprocess_v4final.py)

Skull stripping with HD-BET
Affine registration to MNI152 template using ANTsPy
Z-score normalization on brain voxels
Resize to 128×128×128 and save as float32 .npy

Harmonization (harmonize_2final.py)

Removes scanner/site effects across the four datasets using neuroCombat
Biological variables (diagnosis) are preserved during harmonization
Outputs 568 harmonized .npy files ready for model training

Models

2D CNN — ResNet-18 with transfer learning, slice-based (axial slices 44–84)
3D CNN — Custom SE-VGG-11BN trained from scratch on whole-brain volumes

Requirements
torch >= 2.0
antspyx
nibabel
scipy
numpy
pandas
neuroCombat
HD-BET
scikit-learn
matplotlib

Reference
Zhang et al., Detecting schizophrenia with 3D structural brain MRI using deep learning, Scientific Reports 2023. doi:10.1038/s41598-023-41359-z
