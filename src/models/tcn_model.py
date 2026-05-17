"""
Temporal Convolutional Network for ICU mortality prediction.
Input: 3D patient tensor (batch × 48 hours × 37 variables).
Uses focal loss for class imbalance, early stopping on val AUPRC.
Evaluation metrics match XGBoost for direct comparison.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, auc, precision_recall_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.xgboost_model import hosmer_lemeshow_statistic


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class CausalConv1d(nn.Module):
    """Left-pad only so the convolution has no access to future timesteps."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        return self.conv(F.pad(x, (self.left_pad, 0)))


class TCNBlock(nn.Module):
    """Residual block: CausalConv1d → BN → ReLU → Dropout, plus skip connection."""

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float = 0.2):
        super().__init__()
        self.conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        # 1×1 conv to match channels when they differ
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x)))) + self.residual(x)


class TCNMortality(nn.Module):
    """
    Four-layer dilated causal TCN for binary mortality prediction.

    Input:  (batch, 48, 37)  — time-major from preprocessing
    Output: (batch,)          — mortality probability ∈ [0, 1]

    Receptive field with kernel=3, dilations=[1,2,4,8]:
        1 + 2*(1+2+4+8) = 31 hours
    """

    def __init__(
        self,
        n_variables: int = 37,
        n_filters: int = 64,
        kernel_size: int = 3,
        dilations: List[int] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if dilations is None:
            dilations = [1, 2, 4, 8]

        in_ch = n_variables
        blocks = []
        for d in dilations:
            blocks.append(TCNBlock(in_ch, n_filters, kernel_size, d, dropout))
            in_ch = n_filters
        self.tcn = nn.Sequential(*blocks)

        # MLP head: 64 → 32 → 1
        self.head = nn.Sequential(
            nn.Linear(n_filters, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, V) → (B, V, T) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.tcn(x)         # (B, n_filters, T)
        x = x.mean(dim=-1)      # global avg pool → (B, n_filters)
        return self.head(x).squeeze(-1)  # (B,)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Binary focal loss: FL = −alpha_t · (1−p_t)^gamma · log(p_t)

    alpha: per-sample weight for the positive (minority) class.
           Computed from training labels as (1 − pos_rate) so positives
           are upweighted relative to their frequency.
    gamma: focusing parameter (Lin et al. 2017 recommend gamma=2).
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # preds ∈ (0,1), targets ∈ {0,1} float
        bce     = F.binary_cross_entropy(preds, targets, reduction='none')
        pt      = torch.exp(-bce)   # p_t = p if y=1, else 1−p
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        return (alpha_t * (1.0 - pt) ** self.gamma * bce).mean()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_tcn_data(data_dir: Path) -> Dict[str, np.ndarray]:
    """
    Load pre-saved normalised tensors and concatenate the original missingness
    mask as additional input channels, using the exact same split indices as
    XGBoost. Input shape becomes (B, 48, 74): 37 value channels + 37 mask channels.

    Adding the mask gives the TCN the same information that XGBoost receives via
    its 37 explicit missingness-rate features, and disambiguates NaN-imputed zeros
    from genuinely observed zero measurements.

    Requires finalize_features to have been run first.
    """
    processed = data_dir / "processed"

    X_train = np.load(processed / "X_train_norm.npy")
    X_val   = np.load(processed / "X_val_norm.npy")
    X_test  = np.load(processed / "X_test_norm.npy")

    y_all  = np.load(processed / "y_labels.npy")
    splits = np.load(processed / "splits.npz")
    train_idx = splits['train_idx']
    val_idx   = splits['val_idx']
    test_idx  = splits['test_idx']

    y_train = y_all[train_idx]
    y_val   = y_all[val_idx]
    y_test  = y_all[test_idx]

    # Slice missingness mask (1=originally missing) by the same split indices
    miss_all   = np.load(processed / "missingness_mask.npy").astype(np.float32)
    miss_train = miss_all[train_idx]
    miss_val   = miss_all[val_idx]
    miss_test  = miss_all[test_idx]

    # Replace structural NaN (never-observed after forward-fill) with 0 in value channels
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val   = np.nan_to_num(X_val,   nan=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0)

    # Concatenate: (B, 48, 37) values || (B, 48, 37) mask  =>  (B, 48, 74)
    X_train = np.concatenate([X_train, miss_train], axis=-1)
    X_val   = np.concatenate([X_val,   miss_val],   axis=-1)
    X_test  = np.concatenate([X_test,  miss_test],  axis=-1)

    # Guard against label NaN
    def clean(X, y):
        m = ~np.isnan(y)
        if not m.all():
            print(f"  WARNING: {(~m).sum()} NaN labels removed")
        return X[m], y[m]

    X_train, y_train = clean(X_train, y_train)
    X_val,   y_val   = clean(X_val,   y_val)
    X_test,  y_test  = clean(X_test,  y_test)

    print(f"  Train: {len(y_train)} | Val: {len(y_val)} | Test: {len(y_test)}")
    print(f"  Input channels: {X_train.shape[-1]} (37 values + 37 missingness flags)")
    print(f"  Mortality  train: {y_train.mean():.3f}  "
          f"val: {y_val.mean():.3f}  test: {y_test.mean():.3f}")

    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val':   X_val,   'y_val':   y_val,
        'X_test':  X_test,  'y_test':  y_test,
    }


def make_loader(X: np.ndarray, y: np.ndarray,
                batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(X).float(),
                      torch.from_numpy(y).float()),
        batch_size=batch_size,
        shuffle=shuffle,
    )


# ---------------------------------------------------------------------------
# Train / evaluate helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: TCNMortality,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        # Clip gradients to prevent large focal-loss spikes destabilising weights
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate_loader(
    model: TCNMortality,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Returns (mean_loss, auprc, y_true, y_proba)."""
    model.eval()
    preds_list, labels_list = [], []
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        p = model(X_batch)
        total_loss += criterion(p, y_batch).item() * len(y_batch)
        preds_list.append(p.cpu().numpy())
        labels_list.append(y_batch.cpu().numpy())

    y_true  = np.concatenate(labels_list)
    y_proba = np.concatenate(preds_list)
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    auprc = float(auc(rec, prec))
    return total_loss / len(loader.dataset), auprc, y_true, y_proba


# ---------------------------------------------------------------------------
# Metrics (identical set to XGBoost)
# ---------------------------------------------------------------------------

def compute_test_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
    """
    AUROC, AUPRC, Sensitivity, +P, Score1 at optimal threshold, HL-H.
    Identical logic to evaluate_model() in xgboost_model.py.
    """
    auroc = float(roc_auc_score(y_true, y_proba))

    prec_arr, rec_arr, thresholds = precision_recall_curve(y_true, y_proba)
    auprc = float(auc(rec_arr, prec_arr))

    # Optimal threshold: maximise Score1 = min(Se, +P)
    best_score1, best_thresh = 0.0, 0.5
    for thresh, prec, rec in zip(thresholds, prec_arr[:-1], rec_arr[:-1]):
        s1 = min(float(rec), float(prec))
        if s1 > best_score1:
            best_score1, best_thresh = s1, float(thresh)

    y_pred = (y_proba >= best_thresh).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    return {
        'auroc':                 float(auroc),
        'auprc':                 float(auprc),
        'sensitivity':           float(sensitivity),
        'positive_predictivity': float(ppv),
        'score1':                float(min(sensitivity, ppv)),
        'score1_threshold':      best_thresh,
        'hosmer_lemeshow_h':     float(hosmer_lemeshow_statistic(y_true, y_proba)),
        'test_size':             int(len(y_true)),
        'true_positives':        tp,
        'false_positives':       fp,
        'true_negatives':        tn,
        'false_negatives':       fn,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    val_auprcs: List[float],
    best_epoch: int,
    output_path: Path,
) -> None:
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label='Train loss', color='steelblue')
    ax1.plot(epochs, val_losses,   label='Val loss',   color='coral')
    ax1.axvline(best_epoch, color='gray', linestyle='--',
                label=f'Best epoch ({best_epoch})')
    ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Focal loss', fontsize=11, fontweight='bold')
    ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_auprcs, label='Val AUPRC', color='seagreen')
    ax2.axvline(best_epoch, color='gray', linestyle='--',
                label=f'Best epoch ({best_epoch})')
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('AUPRC', fontsize=11, fontweight='bold')
    ax2.set_title('Validation AUPRC (Early Stopping Criterion)',
                  fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle('TCN Training Curves - ICU Mortality Prediction',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved training curves -> {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(
    data_dir: str = "data",
    output_models_dir: str = None,
    # Model hyperparameters
    n_filters: int = 64,
    kernel_size: int = 3,
    dropout: float = 0.2,
    gamma: float = 2.0,
    # Training hyperparameters
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 5e-4,
    n_epochs: int = 100,
    patience: int = 25,
    max_grad_norm: float = 1.0,
) -> Dict:
    root = Path(__file__).resolve().parents[2]
    data_dir = Path(data_dir) if Path(data_dir).is_absolute() else root / data_dir

    if output_models_dir is None:
        output_models_dir = root / "outputs" / "models"
    output_models_dir = Path(output_models_dir)
    output_models_dir.mkdir(parents=True, exist_ok=True)

    figures_dir = root / "outputs" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir = root / "outputs"
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = output_models_dir / "tcn_model.pt"

    print("=" * 80)
    print("TCN Mortality Prediction - PhysioNet 2012")
    print("=" * 80)
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data...")
    data = load_tcn_data(data_dir)

    train_loader = make_loader(data['X_train'], data['y_train'], batch_size, shuffle=True)
    val_loader   = make_loader(data['X_val'],   data['y_val'],   batch_size, shuffle=False)
    test_loader  = make_loader(data['X_test'],  data['y_test'],  batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # 2. Model
    # ------------------------------------------------------------------
    print("\n[2/5] Building TCN model...")
    # 74 input channels: 37 normalised values + 37 missingness flags
    n_input_channels = data['X_train'].shape[-1]
    model = TCNMortality(
        n_variables=n_input_channels,
        n_filters=n_filters,
        kernel_size=kernel_size,
        dilations=[1, 2, 4, 8],
        dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    # Alpha weights the positive (minority) class
    pos_rate = float(data['y_train'].mean())
    alpha = 1.0 - pos_rate
    print(f"  Focal loss: gamma={gamma}, alpha={alpha:.3f}  (pos_rate={pos_rate:.3f})")

    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ------------------------------------------------------------------
    # 3. Training with early stopping on val AUPRC
    # ------------------------------------------------------------------
    print(f"\n[3/5] Training (max {n_epochs} epochs, patience={patience})...")

    best_val_auprc    = -1.0
    best_epoch        = 0
    epochs_no_improve = 0
    train_losses, val_losses, val_auprcs = [], [], []

    for epoch in range(1, n_epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device, max_grad_norm)
        vl_loss, vl_auprc, _, _ = evaluate_loader(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        val_auprcs.append(vl_auprc)

        if epoch == 1 or epoch % 10 == 0:
            print(f"  Epoch {epoch:3d} | "
                  f"train_loss={tr_loss:.4f}  "
                  f"val_loss={vl_loss:.4f}  "
                  f"val_AUPRC={vl_auprc:.4f}")

        if vl_auprc > best_val_auprc + 1e-4:
            best_val_auprc    = vl_auprc
            best_epoch        = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}  "
                      f"(best epoch={best_epoch}, best_val_AUPRC={best_val_auprc:.4f})")
                break

    if epochs_no_improve < patience:
        print(f"  Training complete. Best val AUPRC: {best_val_auprc:.4f} at epoch {best_epoch}")

    # ------------------------------------------------------------------
    # 4. Test evaluation (best checkpoint)
    # ------------------------------------------------------------------
    print("\n[4/5] Evaluating on test set (best checkpoint)...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    _, _, y_test, y_proba = evaluate_loader(model, test_loader, criterion, device)
    metrics = compute_test_metrics(y_test, y_proba)

    print(f"  AUROC:                    {metrics['auroc']:.4f}")
    print(f"  AUPRC:                    {metrics['auprc']:.4f}")
    print(f"  Sensitivity (Se):         {metrics['sensitivity']:.4f}")
    print(f"  Positive Predictivity (+P): {metrics['positive_predictivity']:.4f}")
    print(f"  Score1 [min(Se,+P)]:      {metrics['score1']:.4f}"
          f"  (threshold={metrics['score1_threshold']:.4f})")
    print(f"  Hosmer-Lemeshow H:        {metrics['hosmer_lemeshow_h']:.4f}")
    print(f"  Test set size:            {metrics['test_size']}")

    # ------------------------------------------------------------------
    # 5. Save results and figures
    # ------------------------------------------------------------------
    print("\n[5/5] Saving results and figures...")

    plot_training_curves(
        train_losses, val_losses, val_auprcs,
        best_epoch, figures_dir / "tcn_training.png",
    )

    results = {
        'metrics': metrics,
        'training': {
            'best_epoch':      best_epoch,
            'best_val_auprc':  float(best_val_auprc),
            'total_epochs':    len(train_losses),
        },
        'model_info': {
            'model_type':   'TCN',
            'n_input_channels': n_input_channels,
            'input_layout': '37 normalised values + 37 missingness flags',
            'n_filters':    n_filters,
            'kernel_size':  kernel_size,
            'dilations':    [1, 2, 4, 8],
            'dropout':      dropout,
            'n_params':     n_params,
            'receptive_field_hours': 1 + 2 * (1 + 2 + 4 + 8),  # = 31
            'focal_loss': {'alpha': float(alpha), 'gamma': gamma},
        },
        'hyperparams': {
            'batch_size':     batch_size,
            'lr':             lr,
            'weight_decay':   weight_decay,
            'n_epochs':       n_epochs,
            'patience':       patience,
            'max_grad_norm':  max_grad_norm,
        },
    }

    with open(results_dir / "tcn_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results -> {results_dir / 'tcn_results.json'}")
    print(f"  Saved model   -> {checkpoint_path}")

    print("\n" + "=" * 80)
    print("TCN Training Complete")
    print("=" * 80)
    print(f"  AUROC:  {metrics['auroc']:.4f}")
    print(f"  AUPRC:  {metrics['auprc']:.4f}")
    print(f"  Score1: {metrics['score1']:.4f}")
    print(f"  HL-H:   {metrics['hosmer_lemeshow_h']:.4f}")
    print("=" * 80)

    return {
        'model':   model,
        'metrics': metrics,
        'y_test':  y_test,
        'y_proba': y_proba,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    main(str(root / "data"))
