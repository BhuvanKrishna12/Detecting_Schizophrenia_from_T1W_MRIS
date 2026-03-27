"""
neuroCombat Harmonization
=========================
Removes scanner/site effects from preprocessed .npy files.

Sites:
  - ds000030  : UCLA scanner
  - ds004302  : Barcelona scanner
  - nusdast   : Washington University 1.5T Siemens

Input:  C:/SchizoDataset/preprocessed/  (.npy files + master_labels.csv)
Output: C:/SchizoDataset/harmonized/    (.npy files + harmonized_labels.csv)
"""

import os
import numpy as np
import pandas as pd
from neuroCombat import neuroCombat
from scipy.ndimage import zoom as zoom_fn

INPUT_DIR  = r"C:/SchizoDataset/preprocessed"
OUTPUT_DIR = r"C:/SchizoDataset/harmonized"
LABELS_CSV = r"C:/SchizoDataset/preprocessed/master_labels.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load master labels ─────────────────────────────────
print("Loading master labels...")
df = pd.read_csv(LABELS_CSV)
print(f"Total subjects: {len(df)}")
print(df["dataset"].value_counts())

# ── Assign site numbers ────────────────────────────────
site_map = {
    "ds000030": 1,
    "ds004302": 2,
    "nusdast":  3,
    "cobre":    4   # for when COBRE arrives
}
df["site"] = df["dataset"].map(site_map)

# ── Load all .npy files into a matrix ─────────────────
print("\nLoading preprocessed volumes...")
volumes = []
valid_indices = []

for idx, row in df.iterrows():
    npy_path = row["filepath"]
    if os.path.exists(npy_path):
        vol = np.load(npy_path).flatten()  # flatten 128x128x128 to 1D
        volumes.append(vol)
        valid_indices.append(idx)
    else:
        print(f"  Missing: {npy_path}")

df = df.loc[valid_indices].reset_index(drop=True)
print(f"Loaded {len(volumes)} volumes")

# Stack into matrix: shape (n_features, n_subjects)
# neuroCombat expects features as rows, subjects as columns
# Downsample from 128^3 to 64^3 to fit in RAM (neuroCombat uses float64 internally)
print("Downsampling volumes for harmonization (64x64x64)...")
volumes_ds = []
for vol in volumes:
    v = vol.reshape(128, 128, 128)
    # simple 2x downsampling by taking every other voxel
    v_ds = v[::2, ::2, ::2].flatten()
    volumes_ds.append(v_ds)
data_matrix = np.array(volumes_ds, dtype=np.float32).T  # shape: (64^3, n_subjects)
print(f"Data matrix shape: {data_matrix.shape}")

# ── Remove zero-variance voxels before neuroCombat ────
# Background voxels (all zeros) cause division by zero in neuroCombat → NaNs
# We mask them out, run neuroCombat on brain voxels only, then put them back
print("Masking zero-variance voxels...")
row_vars = np.var(data_matrix, axis=1)
brain_mask = row_vars > 0
print(f"  Total voxels: {len(brain_mask)}")
print(f"  Brain voxels (non-zero variance): {brain_mask.sum()}")
print(f"  Background voxels (removed): {(~brain_mask).sum()}")
data_matrix_brain = data_matrix[brain_mask, :]

# ── Run neuroCombat ────────────────────────────────────
print("\nRunning neuroCombat harmonization...")
print("This may take a few minutes...")

# Covariates to preserve (biological variables we don't want removed)
covars = {
    "batch":    df["site"].values,       # site/scanner to remove
    "label":    df["label"].values,      # diagnosis to preserve
}

covar_df = pd.DataFrame(covars)

harmonized_brain = neuroCombat(
    dat        = data_matrix_brain,
    covars     = covar_df,
    batch_col  = "batch",
    continuous_cols = [],
    categorical_cols = ["label"]
)["data"]

# Put harmonized brain voxels back, keep background as zeros
harmonized = np.zeros_like(data_matrix)
harmonized[brain_mask, :] = harmonized_brain
print(f"Harmonized matrix shape: {harmonized.shape}")
print(f"NaNs in harmonized: {np.isnan(harmonized).sum()}")

# ── Save harmonized volumes ────────────────────────────
print("\nSaving harmonized volumes...")
for i, (_, row) in enumerate(df.iterrows()):
    sub_id   = row["subject_id"]
    # Upsample harmonization corrections back to 128^3
    # We apply the site correction learned at 64^3 to the full 128^3 volume
    correction_ds = harmonized[:, i].reshape(64, 64, 64).astype(np.float32)
    original      = np.load(row['filepath'])
    # Compute correction factor at downsampled resolution and upsample
    correction_up = zoom_fn(correction_ds, 2, order=1)  # back to 128^3
    original_ds   = original[::2, ::2, ::2]
    # Apply additive correction
    diff = correction_ds - original_ds
    diff_up = zoom_fn(diff, 2, order=1)
    vol = (original + diff_up).astype(np.float32)
    out_path = os.path.join(OUTPUT_DIR, f"{sub_id}.npy")
    # Replace any remaining NaNs with 0 as safety net
    vol = np.nan_to_num(vol, nan=0.0)
    np.save(out_path, vol)

# Update filepaths in labels CSV
df["filepath"] = df["subject_id"].apply(
    lambda x: os.path.join(OUTPUT_DIR, f"{x}.npy")
)

out_csv = os.path.join(OUTPUT_DIR, "harmonized_labels.csv")
df.to_csv(out_csv, index=False)

# ── Summary ───────────────────────────────────────────
print("\n" + "=" * 50)
print("HARMONIZATION COMPLETE")
print("=" * 50)
print(f"Subjects harmonized: {len(df)}")
print(f"SCHZ:    {len(df[df['label']==1])}")
print(f"CTRL:    {len(df[df['label']==0])}")
print(f"Output:  {OUTPUT_DIR}")
print(f"Labels:  {out_csv}")
print("\nSite distribution:")
print(df["dataset"].value_counts())
