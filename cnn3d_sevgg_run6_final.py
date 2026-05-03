"""
3D SE-VGG-11BN for Schizophrenia Detection from Structural MRI
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import zoom

# TorchIO for MRI-specific augmentation (pip install torchio)
import torchio as tio

from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             confusion_matrix, ConfusionMatrixDisplay)

HARMONIZED_DIR = r"C:/SchizoDataset/harmonized"
LABELS_CSV     = r"C:/SchizoDataset/harmonized/harmonized_labels.csv"
OUTPUT_DIR     = r"C:/SchizoDataset/results"
GRADCAM_DIR    = os.path.join(OUTPUT_DIR, "gradcam")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

RESUME_CKPT  = os.path.join(OUTPUT_DIR, 'resume_checkpoint.pth')
BEST_CKPT    = os.path.join(OUTPUT_DIR, 'best_model.pth')

INPUT_SIZE       = 96
BATCH_SIZE       = 2
ACCUM_STEPS      = 4
NUM_EPOCHS       = 150         # increased from 100 (paper used 300)
LR               = 1e-5
EARLY_STOP_PAT   = 20
DROPOUT          = 0.5
SEED             = 42
NUM_GRADCAM_IMGS = 5

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ─────────────────────────────────────────────────────────────
# TorchIO augmentation pipelines
# Training: matches Zhang et al. + rotation all planes + intensity jitter
# Eval: no augmentation
# ─────────────────────────────────────────────────────────────
TRAIN_TRANSFORM = tio.Compose([
    # ── Paper-matched augmentations ──────────────────────────
    tio.RandomNoise(std=(0, 0.05), p=0.6),          # prob=0.6, matches paper
    tio.RandomBlur(std=(0, 1.0),   p=0.1),          # prob=0.1, matches paper
    tio.RandomBiasField(coefficients=0.3, p=0.1),   # prob=0.1, matches paper
    tio.RandomAffine(                               # prob=0.2, matches paper
        scales=(1.0, 1.0),                          # no scaling (just rotation)
        degrees=10,                                 # ±10° all 3 planes
        p=0.2
    ),
    # ── Additional regularization ────────────────────────────
    tio.RandomFlip(axes=(0, 1, 2), p=0.5),          # flip all 3 axes
    tio.RandomGamma(log_gamma=(-0.05, 0.05), p=0.5),# intensity jitter ±5%
])

EVAL_TRANSFORM = None   # no augmentation at val/test time


class MRIDataset3D(Dataset):
    def __init__(self, filepaths, labels, target_size=INPUT_SIZE, augment=False):
        self.filepaths   = filepaths
        self.labels      = labels
        self.target_size = target_size
        self.augment     = augment
        self.transform   = TRAIN_TRANSFORM if augment else EVAL_TRANSFORM

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        vol = np.load(self.filepaths[idx]).astype(np.float32)

        # Resize to target spatial size if needed
        if vol.shape[0] != self.target_size:
            factor = self.target_size / vol.shape[0]
            vol = zoom(vol, factor, order=1)

        # Apply TorchIO augmentation
        if self.transform is not None:
            # TorchIO expects a Subject with a ScalarImage (C, D, H, W)
            vol_tensor = torch.from_numpy(vol).unsqueeze(0)  # (1, D, H, W)
            subject    = tio.Subject(mri=tio.ScalarImage(tensor=vol_tensor))
            subject    = self.transform(subject)
            vol        = subject['mri'].data.squeeze(0).numpy()  # back to (D, H, W)

        vol   = vol[np.newaxis, ...]                             # add channel dim
        label = np.float32(self.labels[idx])
        return torch.from_numpy(vol), torch.tensor(label)

class SEBlock(nn.Module):
    def __init__(self, channels, ratio=16):   # ratio 16→8: more expressive for small dataset
        super().__init__()
        self.squeeze    = nn.AdaptiveAvgPool3d(1)
        self.excitation = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, max(channels // ratio, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // ratio, 1), channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c = x.shape[:2]
        s = self.squeeze(x).view(b, c)
        e = self.excitation(s).view(b, c, 1, 1, 1)
        return x * e

class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, num_convs=1, use_pool=True):
        super().__init__()
        layers = []
        for i in range(num_convs):
            layers += [
                nn.Conv3d(in_ch if i == 0 else out_ch, out_ch,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(out_ch, momentum=0.1),
                nn.ReLU(inplace=True)
            ]
        self.convs    = nn.Sequential(*layers)
        self.se       = SEBlock(out_ch)
        # Block 5 has use_pool=False to preserve spatial resolution (matches paper)
        self.pool     = nn.MaxPool3d(kernel_size=2, stride=2) if use_pool else None

    def forward(self, x):
        x = self.convs(x)
        x = self.se(x)
        if self.pool is not None:
            x = self.pool(x)
        return x

class SEVGG11BN3D(nn.Module):
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.block1 = ConvBlock3D(1,   64,  num_convs=1, use_pool=True)
        self.block2 = ConvBlock3D(64,  128, num_convs=1, use_pool=True)
        self.block3 = ConvBlock3D(128, 256, num_convs=2, use_pool=True)
        self.block4 = ConvBlock3D(256, 256, num_convs=2, use_pool=True)
        self.block5 = ConvBlock3D(256, 512, num_convs=2, use_pool=True)  
        self.pool   = nn.AdaptiveAvgPool3d(1)
        self.head   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.pool(x)
        x = self.head(x)
        return x.squeeze(1)

class GradCAM3D:
    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None
        self._register_hooks()
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)
    def generate(self, input_tensor):
        self.model.eval()
        input_tensor.requires_grad_(True)
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward()
        weights = self.gradients.mean(dim=(0, 2, 3, 4))
        cam = (weights[:, None, None, None] * self.activations[0]).sum(dim=0)
        cam = torch.relu(cam).cpu().numpy()
        from scipy.ndimage import zoom as zoom_fn
        zoom_factors = [s / c for s, c in zip(input_tensor.shape[2:], cam.shape)]
        cam = zoom_fn(cam, zoom_factors, order=1)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

def save_gradcam_overlay(vol_np, cam_np, subject_id, label, pred_prob, save_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    mid = vol_np.shape[0] // 2
    views = {
        'Axial'    : (vol_np[mid, :, :],  cam_np[mid, :, :]),
        'Coronal'  : (vol_np[:, mid, :],  cam_np[:, mid, :]),
        'Sagittal' : (vol_np[:, :, mid],  cam_np[:, :, mid]),
    }
    for ax, (view_name, (mri_slice, cam_slice)) in zip(axes, views.items()):
        ax.imshow(mri_slice, cmap='gray', origin='lower')
        ax.imshow(cam_slice, cmap='jet', alpha=0.4, origin='lower')
        ax.set_title(view_name, fontsize=11)
        ax.axis('off')
    true_str = 'SCHZ' if label == 1 else 'CTRL'
    fig.suptitle(f"{subject_id} | True: {true_str} | Pred prob: {pred_prob:.3f}", fontsize=12)
    plt.tight_layout()
    save_path = os.path.join(save_dir, f"{subject_id}_gradcam.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def compute_metrics(labels, probs, threshold=0.5):
    preds  = (np.array(probs) >= threshold).astype(int)
    labels = np.array(labels)
    auc  = roc_auc_score(labels, probs)
    acc  = accuracy_score(labels, preds)
    cm   = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    return {'auc': auc, 'acc': acc, 'sens': sens, 'spec': spec, 'cm': cm}

def load_and_split():
    df = pd.read_csv(LABELS_CSV)
    print(f"\nTotal subjects in CSV: {len(df)}")
    print(df['dataset'].value_counts())
    df['stratify_key'] = df['label'].astype(str) + '_' + df['dataset'].astype(str)
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df['stratify_key'], random_state=SEED)
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df['stratify_key'], random_state=SEED)
    print(f"\nSplit sizes:")
    print(f"  Train : {len(train_df)} (SCHZ={train_df['label'].sum()}, CTRL={(train_df['label']==0).sum()})")
    print(f"  Val   : {len(val_df)} (SCHZ={val_df['label'].sum()}, CTRL={(val_df['label']==0).sum()})")
    print(f"  Test  : {len(test_df)} (SCHZ={test_df['label'].sum()}, CTRL={(test_df['label']==0).sum()})")
    return train_df, val_df, test_df

def make_loaders(train_df, val_df, test_df):
    train_ds = MRIDataset3D(train_df['filepath'].tolist(), train_df['label'].tolist(), augment=True)
    val_ds   = MRIDataset3D(val_df['filepath'].tolist(),   val_df['label'].tolist(),   augment=False)
    test_ds  = MRIDataset3D(test_df['filepath'].tolist(),  test_df['label'].tolist(),  augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=1,          shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, val_loader, test_loader, test_ds

def get_class_weight(train_df):
    n_schz = train_df['label'].sum()
    n_ctrl = (train_df['label'] == 0).sum()
    pos_weight = torch.tensor([n_ctrl / n_schz], dtype=torch.float32).to(device)
    print(f"\nClass weights: pos_weight (SCHZ) = {pos_weight.item():.4f}")
    return pos_weight

def save_resume_checkpoint(epoch, model, optimizer, scheduler, best_val_auc, patience_cnt, history):
    torch.save({
        'epoch':        epoch,
        'model_state':  model.state_dict(),
        'optim_state':  optimizer.state_dict(),
        'sched_state':  scheduler.state_dict(),
        'best_val_auc': best_val_auc,
        'patience_cnt': patience_cnt,
        'history':      history,
    }, RESUME_CKPT)

def load_resume_checkpoint(model, optimizer, scheduler):
    if not os.path.exists(RESUME_CKPT):
        return 1, 0.0, 0, {k: [] for k in
                            ['train_loss', 'val_loss', 'train_auc', 'val_auc',
                             'train_acc',  'val_acc']}
    print(f"\nResume checkpoint found at '{RESUME_CKPT}'")
    ckpt = torch.load(RESUME_CKPT, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optim_state'])
    scheduler.load_state_dict(ckpt['sched_state'])
    start_epoch  = ckpt['epoch'] + 1
    best_val_auc = ckpt['best_val_auc']
    patience_cnt = ckpt['patience_cnt']
    history      = ckpt['history']
    print(f"Resuming from epoch {start_epoch} | Best val AUC: {best_val_auc:.4f} | Patience: {patience_cnt}/{EARLY_STOP_PAT}")
    return start_epoch, best_val_auc, patience_cnt, history

def train_one_epoch(model, loader, criterion, optimizer, accum_steps):
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []
    optimizer.zero_grad()
    for step, (vols, labels) in enumerate(loader):
        vols   = vols.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(vols)
        loss   = criterion(logits, labels) / accum_steps
        loss.backward()
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * accum_steps
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []
    for vols, labels in loader:
        vols   = vols.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(vols)
        loss   = criterion(logits, labels)
        total_loss += loss.item()
        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_labels, all_probs)
    return avg_loss, metrics

def plot_curves(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['val_loss'],   label='Val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].set_xlabel('Epoch')
    axes[1].plot(epochs, history['train_auc'], label='Train')
    axes[1].plot(epochs, history['val_auc'],   label='Val')
    axes[1].set_title('AUC'); axes[1].legend(); axes[1].set_xlabel('Epoch')
    axes[2].plot(epochs, history['train_acc'], label='Train')
    axes[2].plot(epochs, history['val_acc'],   label='Val')
    axes[2].set_title('Accuracy'); axes[2].legend(); axes[2].set_xlabel('Epoch')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved → {save_path}")

def main():
    print("=" * 60)
    print("3D SE-VGG-11BN — RUN 6 — SCHIZOPHRENIA DETECTION")
    print("=" * 60)
    # print("Changes vs Run 6:")
    # print("  [+] 200 epochs (up from 150)")
    # print("  [+] Early stop patience 20 → 30")
    # print("  [+] CosineAnnealingWarmRestarts T_0=50 (replaces plain cosine)")
    print("-" * 60)

    train_df, val_df, test_df = load_and_split()
    train_loader, val_loader, test_loader,test_ds = make_loaders(train_df, val_df, test_df)
    pos_weight = get_class_weight(train_df)

    model     = SEVGG11BN3D(dropout=DROPOUT).to(device)

    # Label smoothing: smooths hard 0/1 targets to 0.1/0.9
    # Fixes overconfident predictions and threshold drift (Run 4 optimal threshold was 0.2106)
    #pos_weight_smoothed = pos_weight * (1 - 2 * LABEL_SMOOTHING)  # scale weight for smoothed targets
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Wrap criterion to apply label smoothing manually
    # def smoothed_criterion(logits, labels):
    #     smooth_labels = labels * (1 - LABEL_SMOOTHING) + LABEL_SMOOTHING * 0.5
    #     return criterion(logits, smooth_labels)

    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-5)

    # Cosine annealing: smooth LR decay over full run, less noisy than ReduceLROnPlateau
    # with only 85 val subjects
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-7
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")

    start_epoch, best_val_auc, patience_cnt, history = load_resume_checkpoint(
        model, optimizer, scheduler)

    print(f"\nStarting training for up to {NUM_EPOCHS} epochs...")
    print(f"Effective batch size: {BATCH_SIZE * ACCUM_STEPS} (actual={BATCH_SIZE}, accum={ACCUM_STEPS})")
    print("-" * 60)

    try:
        for epoch in range(start_epoch, NUM_EPOCHS + 1):
            train_loss, train_m = train_one_epoch(model, train_loader, criterion, optimizer, ACCUM_STEPS)
            val_loss,   val_m   = evaluate(model, val_loader, criterion)

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_auc'].append(train_m['auc'])
            history['val_auc'].append(val_m['auc'])
            history['train_acc'].append(train_m['acc'])
            history['val_acc'].append(val_m['acc'])

            print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | "
                  f"Train Loss={train_loss:.4f} AUC={train_m['auc']:.4f} Acc={train_m['acc']:.4f} | "
                  f"Val Loss={val_loss:.4f} AUC={val_m['auc']:.4f} Acc={val_m['acc']:.4f} "
                  f"Sens={val_m['sens']:.4f} Spec={val_m['spec']:.4f}")

            if val_m['auc'] > best_val_auc:
                best_val_auc = val_m['auc']
                patience_cnt = 0
                torch.save(model.state_dict(), BEST_CKPT)
                print(f"  --> New best val AUC: {best_val_auc:.4f} | Best checkpoint saved")
            else:
                patience_cnt += 1
                if patience_cnt >= EARLY_STOP_PAT:
                    print(f"\nEarly stopping at epoch {epoch} (no improvement for {EARLY_STOP_PAT} epochs)")
                    break

            scheduler.step()   # CosineAnnealingLR steps every epoch, no metric needed
            save_resume_checkpoint(epoch, model, optimizer, scheduler, best_val_auc, patience_cnt, history)
            plot_curves(history, os.path.join(OUTPUT_DIR, 'training_curves.png'))

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C).")
        print("Progress saved — re-run the script to resume from last epoch.")
        plot_curves(history, os.path.join(OUTPUT_DIR, 'training_curves.png'))
        return

    except Exception as e:
        print(f"\n\nERROR during training: {e}")
        print("Saving resume checkpoint before exiting...")
        save_resume_checkpoint(epoch, model, optimizer, scheduler, best_val_auc, patience_cnt, history)
        plot_curves(history, os.path.join(OUTPUT_DIR, 'training_curves.png'))
        raise

    plot_curves(history, os.path.join(OUTPUT_DIR, 'training_curves.png'))

    print(f"\nLoading best checkpoint from '{BEST_CKPT}'...")
    model.load_state_dict(torch.load(BEST_CKPT, map_location=device))

    _, test_m = evaluate(model, test_loader, criterion)

    print("\n" + "=" * 50)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy    : {test_m['acc']:.4f} ({test_m['acc']*100:.2f}%)")
    print(f"  AUC-ROC     : {test_m['auc']:.4f}")
    print(f"  Sensitivity : {test_m['sens']:.4f}  (SCHZ recall)")
    print(f"  Specificity : {test_m['spec']:.4f}  (CTRL recall)")
    print(f"\nConfusion Matrix (rows=actual, cols=predicted):")
    cm = test_m['cm']
    print(f"               Pred CTRL  Pred SCHZ")
    print(f"  Actual CTRL:   {cm[0,0]:5d}      {cm[0,1]:5d}")
    print(f"  Actual SCHZ:   {cm[1,0]:5d}      {cm[1,1]:5d}")

    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(cm, display_labels=['CTRL', 'SCHZ'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title('Confusion Matrix — Test Set')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved → {OUTPUT_DIR}/confusion_matrix.png")

    print(f"\nGenerating Grad-CAM for {NUM_GRADCAM_IMGS} test subjects...")
    last_conv = None
    for m in model.block5.convs.modules():
        if isinstance(m, nn.Conv3d):
            last_conv = m
    gradcam = GradCAM3D(model, last_conv)

    # Collect all predictions first, then pick representative cases
    all_preds = []
    model.eval()
    with torch.no_grad():
        for i, (vols, labels) in enumerate(test_loader):
            logit     = model(vols.to(device))
            pred_prob = torch.sigmoid(logit).item()
            label     = int(labels[0].item())
            pred      = 1 if pred_prob >= 0.5 else 0
            all_preds.append({'idx': i, 'label': label, 'pred': pred, 'prob': pred_prob})

    # Pick 1-2 of each case type
    tp = [p for p in all_preds if p['label']==1 and p['pred']==1][:2]  # true positives
    tn = [p for p in all_preds if p['label']==0 and p['pred']==0][:2]  # true negatives
    fp = [p for p in all_preds if p['label']==0 and p['pred']==1][:1]  # false positive
    fn = [p for p in all_preds if p['label']==1 and p['pred']==0][:1]  # false negative
    selected = tp + tn + fp + fn

    print(f"  Selected {len(selected)} subjects: {len(tp)} TP, {len(tn)} TN, {len(fp)} FP, {len(fn)} FN")

    gradcam_count = 0
    for entry in selected:
        i         = entry['idx']
        vols, labels = test_ds[i]
        vols      = vols.unsqueeze(0).to(device)
        label     = entry['label']
        with torch.enable_grad():
            cam = gradcam.generate(vols)
        with torch.no_grad():
            logit     = model(vols)
            pred_prob = entry['prob']
        vol_np = vols[0, 0].detach().cpu().numpy()
        sub_id = test_df.iloc[i]['subject_id']
        save_gradcam_overlay(vol_np, cam, sub_id, label, pred_prob, GRADCAM_DIR)
        gradcam_count += 1
        print(f"  Grad-CAM saved: {sub_id} ({'SCHZ' if label==1 else 'CTRL'}) pred={pred_prob:.3f}")

    print(f"\nAll Grad-CAM images saved → {GRADCAM_DIR}/")
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

if __name__ == "__main__":
    main()