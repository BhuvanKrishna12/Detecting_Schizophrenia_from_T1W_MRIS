"""
Schizophrenia Detection - Preprocessing Pipeline
=================================================
Handles 3 datasets:
  - ds000030  : C:/SchizoDataset/              (NIfTI .nii.gz)
  - ds004302  : C:/SchizoDataset/ds004302/     (NIfTI .nii.gz, already skull stripped)
  - NUSDAST   : C:/SchizoDataset/NUSDAST/      (ANALYZE .img/.hdr)

Pipeline per subject:
  1. Skull Strip      - HD-BET (skip for ds004302, already stripped)
  2. MNI Registration - ANTsPy
  3. Z-score Norm     - brain-masked z-score (WhiteStripe incompatible with nibabel 5.0)
  4. Resize           - 128x128x128
  5. Save             - .npy

Output: C:/SchizoDataset/preprocessed/
        C:/SchizoDataset/preprocessed/master_labels.csv
"""

import os
import glob
import numpy as np
import pandas as pd
import ants
from scipy.ndimage import zoom
# intensity_normalization not used (incompatible with nibabel 5.0)
import shutil
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# PATHS — edit if your folders are different
# ─────────────────────────────────────────────
DS000030_DIR   = r"C:/SchizoDataset"
DS004302_DIR   = r"C:/SchizoDataset/ds004302"
NUSDAST_DIR    = r"C:/SchizoDataset/NUSDAST"
OUTPUT_DIR     = r"C:/SchizoDataset/preprocessed"
TEMP_DIR       = r"C:/SchizoDataset/temp_hdbet"
COBRE_DIR      = r"C:/SchizoDataset/NIFTI"

DS000030_LABELS = r"C:/SchizoDataset/labels.csv"
NUSDAST_LABELS  = r"C:/SchizoDataset/NUSDAST/bhuvan_krishna_3_7_2026_10_16_35.csv"
DS004302_LABELS = r"C:/SchizoDataset/ds004302/participants.tsv"

TARGET_SHAPE = (128, 128, 128)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR,   exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1 — BUILD MASTER SUBJECT LIST
# ─────────────────────────────────────────────

def build_subject_list():
    subjects = []  # each entry: {id, filepath, label, dataset, skull_stripped}

    import random
    random.seed(42)

    # --- ds000030 ---
    df0 = pd.read_csv(DS000030_LABELS)
    for _, row in df0.iterrows():
        sub_id = row['participant_id']
        label  = 1 if row['diagnosis'] == 'SCHZ' else 0
        # handle both .nii.gz and .nii (some subjects extracted without gz)
        path = os.path.join(DS000030_DIR, sub_id, 'anat', f'{sub_id}_T1w.nii.gz')
        if not os.path.exists(path):
            path = os.path.join(DS000030_DIR, sub_id, 'anat', f'{sub_id}_T1w.nii')
        if os.path.exists(path):
            subjects.append({
                'id':             f"ds000030_{sub_id}",
                'filepath':       path,
                'label':          label,
                'dataset':        'ds000030',
                'skull_stripped': False
            })

    # --- ds004302 ---
    df4 = pd.read_csv(DS004302_LABELS, sep='\t')
    for _, row in df4.iterrows():
        sub_id = row['participant_id']
        group  = row['group']
        if group not in ['HC', 'AVH-', 'AVH+']:
            continue
        label = 0 if group == 'HC' else 1
        path  = os.path.join(DS004302_DIR, sub_id, 'anat', f'{sub_id}_T1w.nii.gz')
        if os.path.exists(path):
            subjects.append({
                'id':             f"ds004302_{sub_id}",
                'filepath':       path,
                'label':          label,
                'dataset':        'ds004302',
                'skull_stripped': True   # already stripped
            })

    # --- NUSDAST ---
    dfn = pd.read_csv(NUSDAST_LABELS)
    dfn = dfn[dfn['Group'].isin([1.0, 3.0])]  # 1=SCHZ, 3=CONTROL
    for _, row in dfn.iterrows():
        sub_id = row['Subject']
        label  = 1 if row['Group'] == 1.0 else 0
        # find the .img file inside CC####/CC####_0/MPR1/ANALYZE/
        pattern = os.path.join(NUSDAST_DIR, sub_id, f'{sub_id}_0', 'MPR1', 'ANALYZE', '*.img')
        files   = glob.glob(pattern)
        if files:
            subjects.append({
                'id':             f"nusdast_{sub_id}",
                'filepath':       files[0],
                'label':          label,
                'dataset':        'nusdast',
                'skull_stripped': False
            })

    # --- COBRE ---
    cobre_files = glob.glob(os.path.join(COBRE_DIR, "*.nii")) + \
              glob.glob(os.path.join(COBRE_DIR, "*.nii.gz"))

    for path in cobre_files:
        fname = os.path.basename(path)
        if "_1_7" in fname:
            label = 1
        elif "_3_7" in fname:
            label = 0
        else:
            continue
        subjects.append({
            'id':             f"cobre_{fname.split('.')[0]}",
            'filepath':       path,
            'label':          label,
            'dataset':        'cobre',
            'skull_stripped': False
        })

    # Subsample COBRE controls to 81
    cobre_ctrl = [s for s in subjects if s['dataset'] == 'cobre' and s['label'] == 0]
    cobre_schz = [s for s in subjects if s['dataset'] == 'cobre' and s['label'] == 1]
    cobre_ctrl_sampled = random.sample(cobre_ctrl, min(81, len(cobre_ctrl)))
    subjects = [s for s in subjects if s['dataset'] != 'cobre']
    subjects += cobre_schz + cobre_ctrl_sampled

    print(f"\nTotal subjects found: {len(subjects)}")
    schz  = sum(1 for s in subjects if s['label'] == 1)
    ctrl  = sum(1 for s in subjects if s['label'] == 0)
    print(f"  SCHZ:    {schz}")
    print(f"  CONTROL: {ctrl}")
    return subjects


# ─────────────────────────────────────────────
# STEP 2 — SKULL STRIPPING (HD-BET)
# ─────────────────────────────────────────────

def skull_strip(input_path, output_path):
    """Run HD-BET on a single file. Returns path to stripped file."""
    import torch
    from HD_BET.entry_point import get_hdbet_predictor, hdbet_predict
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        predictor = get_hdbet_predictor(use_tta=False, device=device, verbose=False)
        hdbet_predict(input_path, output_path, predictor,
                      keep_brain_mask=False, compute_brain_extracted_image=True)
    except Exception as e:
        print(f"    HD-BET error: {e}")
    return output_path if os.path.exists(output_path) else input_path


# ─────────────────────────────────────────────
# STEP 3 — MNI REGISTRATION (ANTsPy)
# ─────────────────────────────────────────────

def register_to_mni(input_path):
    """Register brain to MNI152 template. Returns numpy array."""
    fixed  = ants.get_ants_data('mni')          # built-in MNI152 template
    fixed  = ants.image_read(fixed)
    moving = ants.image_read(input_path)

    # Rigid + affine registration (faster than full SyN, good enough)
    result = ants.registration(
        fixed   = fixed,
        moving  = moving,
        type_of_transform = 'Affine'
    )
    registered = result['warpedmovout']
    return registered.numpy()


# ─────────────────────────────────────────────
# STEP 4 — WHITESTRIPE NORMALIZATION
# ─────────────────────────────────────────────

def whitestripe_normalize(volume_array, input_path):
    """
    Z-score normalization on brain voxels only (non-zero mask).
    WhiteStripe is incompatible with nibabel 5.0 (uses deprecated get_data()).
    Z-score on brain tissue is a standard, well-accepted alternative.
    """
    arr  = volume_array.astype(np.float32)
    mask = arr > 0
    mean = arr[mask].mean()
    std  = arr[mask].std()
    arr[mask] = (arr[mask] - mean) / (std + 1e-8)
    arr[~mask] = 0.0  # keep background as zero
    return arr


# ─────────────────────────────────────────────
# STEP 5 — RESIZE TO 128x128x128
# ─────────────────────────────────────────────

def resize_volume(volume, target_shape=TARGET_SHAPE):
    """Resize volume to target shape using zoom."""
    factors = [t / s for t, s in zip(target_shape, volume.shape)]
    resized = zoom(volume, factors, order=1)  # order=1 = linear interpolation
    return resized.astype(np.float32)


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def preprocess_subject(subject, idx, total):
    sub_id   = subject['id']
    filepath = subject['filepath']
    label    = subject['label']
    dataset  = subject['dataset']
    stripped = subject['skull_stripped']

    out_path = os.path.join(OUTPUT_DIR, f"{sub_id}.npy")

    # Skip if already processed
    if os.path.exists(out_path):
        print(f"[{idx+1}/{total}] SKIP (already done): {sub_id}")
        return True

    print(f"\n[{idx+1}/{total}] Processing: {sub_id} | label={label} | dataset={dataset}")

    try:
        # --- 1. Skull Strip ---
        if not stripped:
            print(f"  Step 1/4: Skull stripping...")
            stripped_path = f"{TEMP_DIR}/{sub_id}_stripped.nii.gz"
            skull_strip(filepath, stripped_path)
            working_path  = stripped_path
        else:
            print(f"  Step 1/4: Skull strip SKIPPED (already stripped)")
            working_path  = filepath

        # --- 2. MNI Registration ---
        print(f"  Step 2/4: MNI registration...")
        registered_arr = register_to_mni(working_path)

        # --- 3. WhiteStripe Normalization ---
        print(f"  Step 3/4: WhiteStripe normalization...")
        normalized_arr = whitestripe_normalize(registered_arr, working_path)

        # --- 4. Resize ---
        print(f"  Step 4/4: Resizing to {TARGET_SHAPE}...")
        resized_arr = resize_volume(normalized_arr)

        # --- 5. Save ---
        np.save(out_path, resized_arr)
        print(f"  DONE -> {out_path} | shape={resized_arr.shape}")

        # Cleanup temp files
        if not stripped and os.path.exists(stripped_path):
            os.remove(stripped_path)

        return True

    except Exception as e:
        print(f"  ERROR on {sub_id}: {e}")
        return False


def main():
    print("=" * 60)
    print("SCHIZOPHRENIA DETECTION - PREPROCESSING PIPELINE")
    print("=" * 60)

    # Build subject list
    subjects = build_subject_list()

    # Track results
    success = []
    failed  = []

    # Process each subject
    for idx, subject in enumerate(subjects):
        ok = preprocess_subject(subject, idx, len(subjects))
        if ok:
            success.append(subject)
        else:
            failed.append(subject)

    # Save master labels CSV
    labels_data = []
    for subject in success:
        out_path = os.path.join(OUTPUT_DIR, f"{subject['id']}.npy")
        if os.path.exists(out_path):
            labels_data.append({
                'subject_id': subject['id'],
                'filepath':   out_path,
                'label':      subject['label'],
                'dataset':    subject['dataset']
            })

    labels_df = pd.DataFrame(labels_data)
    labels_csv = os.path.join(OUTPUT_DIR, 'master_labels.csv')
    labels_df.to_csv(labels_csv, index=False)

    # Final summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"Successfully processed: {len(success)}")
    print(f"Failed:                 {len(failed)}")
    print(f"Output folder:          {OUTPUT_DIR}")
    print(f"Labels CSV:             {labels_csv}")

    schz = len(labels_df[labels_df['label'] == 1])
    ctrl = len(labels_df[labels_df['label'] == 0])
    print(f"\nFinal dataset:")
    print(f"  SCHZ:    {schz}")
    print(f"  CONTROL: {ctrl}")

    if failed:
        print(f"\nFailed subjects:")
        for s in failed:
            print(f"  {s['id']}")

    # Cleanup temp dir
    shutil.rmtree(TEMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
