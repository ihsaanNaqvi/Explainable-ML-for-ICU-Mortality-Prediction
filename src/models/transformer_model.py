"""
Time-aware Transformer for ICU mortality prediction (Phase 6).

Architecture:
  - 74 input channels (37 normalised values + 37 missingness flags)
  - Time-aware positional encoding with learnable frequencies
  - CLS token classification (BERT-style)
  - 4 pre-norm encoder layers, 4 attention heads, d_model=64
  - Focal loss + Adam + early stopping on val AUPRC
"""

import math
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class TimeAwarePositionalEncoding(nn.Module):
    """Learnable-frequency sinusoidal PE conditioned on actual hour timestamps."""

    def __init__(self, d_model: int):
        super().__init__()
        n_freqs = d_model // 2
        init_freqs = torch.exp(
            -torch.arange(0, n_freqs).float() * 2.0 * math.log(10000.0) / d_model
        )
        self.freqs = nn.Parameter(init_freqs)  # (n_freqs,)  learnable

    def forward(self, hours: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hours: (B, T) float tensor of hour indices (0..47)
        Returns:
            (B, T, d_model) positional embeddings
        """
        t = hours.unsqueeze(-1)                                 # (B, T, 1)
        w = self.freqs.unsqueeze(0).unsqueeze(0)                # (1, 1, n_freqs)
        angles = t * w                                          # (B, T, n_freqs)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # (B, T, d_model)


# ---------------------------------------------------------------------------
# Transformer encoder layer (pre-norm)
# ---------------------------------------------------------------------------

class _TransformerLayer(nn.Module):
    """Pre-norm single encoder layer.  Stores attention weights during eval."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff1 = nn.Linear(d_model, ff_dim)
        self.ff2 = nn.Linear(ff_dim, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.attn_weights: torch.Tensor | None = None  # populated during eval

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- self-attention branch ---
        x_n = self.norm1(x)
        if self.training:
            attn_out, _ = self.self_attn(x_n, x_n, x_n, need_weights=False)
            self.attn_weights = None
        else:
            attn_out, attn_w = self.self_attn(
                x_n, x_n, x_n,
                need_weights=True,
                average_attn_weights=False,  # keep per-head: (B, nhead, T, T)
            )
            self.attn_weights = attn_w.detach().cpu()

        x = x + self.drop(attn_out)

        # --- feed-forward branch ---
        x_n = self.norm2(x)
        x = x + self.drop(self.ff2(self.drop(self.act(self.ff1(x_n)))))
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class TimeAwareTransformer(nn.Module):
    """
    Time-aware Transformer for 48-hour ICU mortality prediction.

    Input:
        x     : (B, 48, 74)  normalised values + missingness flags
        hours : (B, 48)      float hour indices (0..47)
    Output:
        (B,)  mortality probability in [0, 1]
    """

    def __init__(
        self,
        n_input_channels: int = 74,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        # project 74-channel input to d_model
        self.value_proj = nn.Linear(n_input_channels, d_model)

        # time-aware PE
        self.time_pe = TimeAwarePositionalEncoding(d_model)

        # CLS token: (1, 1, d_model) learnable
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_drop = nn.Dropout(dropout)

        # encoder stack
        self.layers = nn.ModuleList([
            _TransformerLayer(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)

        # classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        hours: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # embed values + positional encoding
        v = self.value_proj(x)              # (B, T, d_model)
        pe = self.time_pe(hours)            # (B, T, d_model)
        tokens = self.embed_drop(self.embed_norm(v + pe))  # (B, T, d_model)

        # prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, d_model)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, T+1, d_model)

        for layer in self.layers:
            tokens = layer(tokens)

        cls_out = self.out_norm(tokens[:, 0, :])  # (B, d_model)
        logit = self.head(cls_out).squeeze(-1)    # (B,)
        return torch.sigmoid(logit)

    def get_attention_weights(self) -> list[torch.Tensor | None]:
        """Return stored attention weights from last forward pass (eval mode only)."""
        return [layer.attn_weights for layer in self.layers]


# ---------------------------------------------------------------------------
# Focal loss (identical to TCN)
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        eps = 1e-7
        probs = probs.clamp(eps, 1.0 - eps)
        bce = -targets * torch.log(probs) - (1 - targets) * torch.log(1 - probs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1,
                              torch.full_like(pt, self.alpha),
                              torch.full_like(pt, 1 - self.alpha))
        return (alpha_t * (1 - pt) ** self.gamma * bce).mean()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_transformer_data(data_dir: str = "data/processed"):
    """Load normalised arrays, missingness mask, and split indices.

    Returns:
        (X_train, X_val, X_test) : (B, 48, 74) float32 tensors — 37 values + 37 mask
        (y_train, y_val, y_test) : (B,) float32 tensors
        hours_train/val/test     : (B, 48) float32 — hour indices 0..47
    """
    X_train = np.load(os.path.join(data_dir, "X_train_norm.npy"))
    X_val   = np.load(os.path.join(data_dir, "X_val_norm.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test_norm.npy"))

    splits = np.load(os.path.join(data_dir, "splits.npz"))
    train_idx = splits["train_idx"]
    val_idx   = splits["val_idx"]
    test_idx  = splits["test_idx"]

    mask_full = np.load(os.path.join(data_dir, "missingness_mask.npy")).astype(np.float32)
    mask_train = mask_full[train_idx]
    mask_val   = mask_full[val_idx]
    mask_test  = mask_full[test_idx]

    # NaN in normalised arrays = imputed-missing; zero-fill (mask already encodes missingness)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val   = np.nan_to_num(X_val,   nan=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0)

    # concatenate along feature dim: (B,48,37) + (B,48,37) -> (B,48,74)
    X_train = np.concatenate([X_train, mask_train], axis=2).astype(np.float32)
    X_val   = np.concatenate([X_val,   mask_val],   axis=2).astype(np.float32)
    X_test  = np.concatenate([X_test,  mask_test],  axis=2).astype(np.float32)

    labels = np.load(os.path.join(data_dir, "y_labels.npy"))
    y_train = labels[train_idx].astype(np.float32)
    y_val   = labels[val_idx].astype(np.float32)
    y_test  = labels[test_idx].astype(np.float32)

    # hour indices: 0, 1, ..., 47 (same for every patient)
    T = X_train.shape[1]
    hours_base = np.arange(T, dtype=np.float32)
    hours_train = np.tile(hours_base, (len(y_train), 1))
    hours_val   = np.tile(hours_base, (len(y_val),   1))
    hours_test  = np.tile(hours_base, (len(y_test),  1))

    return (
        torch.from_numpy(X_train), torch.from_numpy(X_val), torch.from_numpy(X_test),
        torch.from_numpy(y_train), torch.from_numpy(y_val), torch.from_numpy(y_test),
        torch.from_numpy(hours_train), torch.from_numpy(hours_val), torch.from_numpy(hours_test),
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimiser, criterion, device, max_grad_norm=1.0):
    model.train()
    total_loss = 0.0
    for X_b, y_b, h_b in loader:
        X_b, y_b, h_b = X_b.to(device), y_b.to(device), h_b.to(device)
        optimiser.zero_grad()
        preds = model(X_b, h_b)
        loss = criterion(preds, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimiser.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for X_b, y_b, h_b in loader:
        X_b, y_b, h_b = X_b.to(device), y_b.to(device), h_b.to(device)
        preds = model(X_b, h_b)
        total_loss += criterion(preds, y_b).item() * len(y_b)
        all_probs.append(preds.cpu().numpy())
        all_labels.append(y_b.cpu().numpy())
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    p, r, _ = precision_recall_curve(labels, probs)
    auprc = auc(r, p)
    return total_loss / len(loader.dataset), auprc, probs, labels


# ---------------------------------------------------------------------------
# Metrics (identical contract to XGBoost / TCN)
# ---------------------------------------------------------------------------

def compute_test_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    auroc = roc_auc_score(labels, probs)

    precision_arr, recall_arr, thresholds = precision_recall_curve(labels, probs)
    auprc = auc(recall_arr, precision_arr)

    # Score1 = min(Se, +P) maximised over thresholds
    best_score1, best_thresh = 0.0, 0.5
    for thr in thresholds:
        preds_bin = (probs >= thr).astype(int)
        tp = int(((preds_bin == 1) & (labels == 1)).sum())
        fp = int(((preds_bin == 1) & (labels == 0)).sum())
        fn = int(((preds_bin == 0) & (labels == 1)).sum())
        se  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        s1  = min(se, ppv)
        if s1 > best_score1:
            best_score1  = s1
            best_thresh  = float(thr)
            best_se, best_ppv = se, ppv
            best_tp, best_fp, best_fn = tp, fp, fn

    tn = int(((probs < best_thresh) & (labels == 0)).sum())

    # Hosmer-Lemeshow H (10 equal-width bins)
    bins = np.linspace(0, 1, 11)
    hl_h = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = (probs >= lo) & (probs < hi)
        if idx.sum() == 0:
            continue
        n_k   = idx.sum()
        o_k   = labels[idx].sum()
        e_k   = probs[idx].sum()
        hl_h += (o_k - e_k) ** 2 / (e_k * (1 - e_k / n_k) + 1e-8)

    return {
        "auroc":                 float(auroc),
        "auprc":                 float(auprc),
        "sensitivity":           float(best_se),
        "positive_predictivity": float(best_ppv),
        "score1":                float(best_score1),
        "score1_threshold":      float(best_thresh),
        "hosmer_lemeshow_h":     float(hl_h),
        "test_size":             int(len(labels)),
        "true_positives":        int(best_tp),
        "false_positives":       int(best_fp),
        "true_negatives":        int(tn),
        "false_negatives":       int(best_fn),
    }


# ---------------------------------------------------------------------------
# Attention visualisation
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_patient_attention(
    model: TimeAwareTransformer,
    X: torch.Tensor,
    hours: torch.Tensor,
    patient_idx: int,
    device: torch.device,
) -> list[np.ndarray]:
    """
    Run one patient through the model (eval mode) and extract per-layer
    CLS attention over the 48 time steps.

    Returns:
        List of n_layers arrays, each (n_heads, 48) — attention from CLS token
        to each of the 48 hour positions (index 1..48 in the 49-token sequence).
    """
    model.eval()
    x_p = X[patient_idx : patient_idx + 1].to(device)
    h_p = hours[patient_idx : patient_idx + 1].to(device)
    _ = model(x_p, h_p)

    cls_attn_per_layer = []
    for layer_weights in model.get_attention_weights():
        if layer_weights is None:
            continue
        # layer_weights: (1, n_heads, 49, 49)  query=CLS is position 0
        cls_row = layer_weights[0, :, 0, 1:]  # (n_heads, 48)
        cls_attn_per_layer.append(cls_row.numpy())

    return cls_attn_per_layer


def plot_attention_weights(
    model: TimeAwareTransformer,
    X_test: torch.Tensor,
    hours_test: torch.Tensor,
    y_test: torch.Tensor,
    device: torch.device,
    output_dir: str = "outputs/figures/attention",
    n_patients: int = 3,
):
    os.makedirs(output_dir, exist_ok=True)

    # pick patients: first non-survivor, first survivor, then next non-survivor
    nonsurv = (y_test == 1).nonzero(as_tuple=True)[0].tolist()
    surv    = (y_test == 0).nonzero(as_tuple=True)[0].tolist()
    candidates = []
    for i in range(max(len(nonsurv), len(surv))):
        if i < len(nonsurv): candidates.append(nonsurv[i])
        if i < len(surv):    candidates.append(surv[i])
        if len(candidates) >= n_patients:
            break
    patient_ids = candidates[:n_patients]

    hours_arr = np.arange(48)
    colors = ["steelblue", "darkorange", "seagreen", "crimson"]

    for pid in patient_ids:
        cls_attn = extract_patient_attention(model, X_test, hours_test, pid, device)
        label_str = "non-survivor" if y_test[pid].item() == 1 else "survivor"

        # Layer 1 only — the only layer with non-uniform attention.
        # Deeper layers collapse to ~uniform distributions on this dataset size.
        layer1_attn = cls_attn[0]  # (n_heads, 48)
        n_heads = layer1_attn.shape[0]

        fig, axes = plt.subplots(2, 2, figsize=(14, 6), sharex=True, sharey=False)
        axes = axes.flatten()
        fig.suptitle(
            f"Layer 1 CLS Attention — Patient {pid} ({label_str})\n"
            f"Per-head breakdown (deeper layers show uniform attention)",
            fontsize=12, fontweight="bold"
        )

        for head_idx in range(n_heads):
            ax = axes[head_idx]
            attn = layer1_attn[head_idx]  # (48,)
            ax.bar(hours_arr, attn, color=colors[head_idx], alpha=0.75, width=0.8)
            ax.set_title(f"Head {head_idx + 1}", fontsize=11)
            ax.set_ylabel("Attention weight", fontsize=9)
            ax.set_xlim(-0.5, 47.5)
            ax.tick_params(axis="x", labelsize=8)
            if head_idx >= 2:
                ax.set_xlabel("Hour (0–47)", fontsize=9)

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"patient_{pid}_attention.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved attention plot: {out_path}")


# ---------------------------------------------------------------------------
# Training-curve plot
# ---------------------------------------------------------------------------

def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    val_auprcs: list[float],
    best_epoch: int,
    output_dir: str = "outputs/figures",
):
    os.makedirs(output_dir, exist_ok=True)
    epochs = list(range(1, len(train_losses) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Transformer Training Curves", fontsize=13, fontweight="bold")

    ax1.plot(epochs, train_losses, label="Train loss")
    ax1.plot(epochs, val_losses,   label="Val loss")
    ax1.axvline(best_epoch, color="red", linestyle="--", label=f"Best epoch {best_epoch}")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Focal loss")
    ax1.set_title("Loss"); ax1.legend()

    ax2.plot(epochs, val_auprcs, label="Val AUPRC", color="green")
    ax2.axvline(best_epoch, color="red", linestyle="--")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("AUPRC")
    ax2.set_title("Validation AUPRC"); ax2.legend()

    plt.tight_layout()
    out_path = os.path.join(output_dir, "transformer_training.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curve saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- paths ----
    data_dir    = "data/processed"
    model_dir   = "outputs/models"
    results_dir = "outputs"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- data ----
    print("\n[1/5] Loading data...")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     h_train, h_val, h_test) = load_transformer_data(data_dir)

    print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")
    pos_rate = float(y_train.mean())
    print(f"  Train pos rate: {pos_rate:.4f}")

    # ---- model ----
    print("\n[2/5] Building model...")
    model = TimeAwareTransformer(
        n_input_channels=74,
        d_model=64,
        n_heads=4,
        n_layers=4,
        ff_dim=256,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")

    focal_alpha = 1.0 - pos_rate
    criterion = FocalLoss(alpha=focal_alpha, gamma=2.0)

    optimiser = torch.optim.Adam(
        model.parameters(), lr=3e-4, weight_decay=5e-4
    )

    # ---- data loaders ----
    batch_size = 64
    train_ds = TensorDataset(X_train, y_train, h_train)
    val_ds   = TensorDataset(X_val,   y_val,   h_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    # ---- training loop ----
    print("\n[3/5] Training...")
    n_epochs     = 100
    patience     = 25
    max_grad_norm = 1.0

    best_val_auprc = -1.0
    best_epoch     = 0
    epochs_no_imp  = 0
    best_state     = None

    train_losses, val_losses, val_auprcs = [], [], []

    for epoch in range(1, n_epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimiser, criterion, device, max_grad_norm)
        vl_loss, vl_auprc, _, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)
        val_auprcs.append(vl_auprc)

        improved = vl_auprc > best_val_auprc
        if improved:
            best_val_auprc = vl_auprc
            best_epoch     = epoch
            epochs_no_imp  = 0
            best_state     = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_imp += 1

        if epoch % 5 == 0 or epoch == 1:
            marker = "*" if improved else " "
            print(
                f"  Epoch {epoch:3d}{marker}  "
                f"tr_loss={tr_loss:.4f}  vl_loss={vl_loss:.4f}  "
                f"vl_auprc={vl_auprc:.4f}  best={best_val_auprc:.4f}"
            )

        if epochs_no_imp >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    # ---- restore best weights ----
    model.load_state_dict(best_state)
    model_path = os.path.join(model_dir, "transformer_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n  Best model saved to {model_path}  (epoch {best_epoch}, val AUPRC={best_val_auprc:.4f})")

    # ---- test evaluation ----
    print("\n[4/5] Evaluating on test set...")
    test_ds     = TensorDataset(X_test, y_test, h_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    _, _, test_probs, test_labels = evaluate(model, test_loader, criterion, device)
    metrics = compute_test_metrics(test_probs, test_labels)

    print(f"  AUROC  : {metrics['auroc']:.4f}")
    print(f"  AUPRC  : {metrics['auprc']:.4f}")
    print(f"  Score1 : {metrics['score1']:.4f}  (Se={metrics['sensitivity']:.4f}, +P={metrics['positive_predictivity']:.4f})")
    print(f"  HL-H   : {metrics['hosmer_lemeshow_h']:.2f}")
    print(f"  TP={metrics['true_positives']}  FP={metrics['false_positives']}  "
          f"TN={metrics['true_negatives']}  FN={metrics['false_negatives']}")

    results = {
        "metrics":    metrics,
        "training": {
            "best_epoch":     best_epoch,
            "best_val_auprc": best_val_auprc,
            "total_epochs":   len(train_losses),
        },
        "model_info": {
            "model_type":        "TimeAwareTransformer",
            "n_input_channels":  74,
            "input_layout":      "37 normalised values + 37 missingness flags",
            "d_model":           64,
            "n_heads":           4,
            "n_layers":          4,
            "ff_dim":            256,
            "dropout":           0.1,
            "n_params":          n_params,
            "focal_loss": {
                "alpha": float(focal_alpha),
                "gamma": 2.0,
            },
        },
        "hyperparams": {
            "batch_size":    batch_size,
            "lr":            3e-4,
            "weight_decay":  5e-4,
            "n_epochs":      n_epochs,
            "patience":      patience,
            "max_grad_norm": max_grad_norm,
        },
    }

    results_path = os.path.join(results_dir, "transformer_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {results_path}")

    # ---- plots ----
    print("\n[5/5] Generating figures...")
    plot_training_curves(train_losses, val_losses, val_auprcs, best_epoch)
    plot_attention_weights(model, X_test, h_test, y_test, device, n_patients=3)

    print("\nDone.")
    return model, results


if __name__ == "__main__":
    main()
