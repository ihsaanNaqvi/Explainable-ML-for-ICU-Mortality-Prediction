"""
Ablation study for the Time-aware Transformer (Phase 7).

Tests four variants to prove each design choice contributes:
  1. Full model          — proposed architecture (all components)
  2. w/o Time-aware PE   — replace learnable-frequency PE with fixed sinusoidal
  3. w/o Missingness     — 37 input channels only (drop missingness flags)
  4. w/o Focal Loss      — replace focal loss with standard BCE

Each variant is trained from scratch with identical hyperparameters and the
same reproducible random seed. Results saved to:
  outputs/ablation_results.json
  outputs/figures/ablation_comparison.png
"""

import math
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── shared utilities from the main transformer module ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from models.transformer_model import (
    _TransformerLayer,
    TimeAwarePositionalEncoding,
    FocalLoss,
    compute_test_metrics,
    train_epoch,
    evaluate,
)

SEED = 42


# ---------------------------------------------------------------------------
# Standard (non-learnable) positional encoding  — ablation variant 2
# ---------------------------------------------------------------------------

class StandardPositionalEncoding(nn.Module):
    """Fixed sinusoidal PE using position indices 0..T-1 (ignores actual hours)."""

    def __init__(self, d_model: int, max_len: int = 64):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # not a parameter — fixed

    def forward(self, hours: torch.Tensor) -> torch.Tensor:
        """hours: (B, T) — ignored; uses fixed position indices instead."""
        B, T = hours.shape
        return self.pe[:T].unsqueeze(0).expand(B, -1, -1)  # (B, T, d_model)


# ---------------------------------------------------------------------------
# Ablation transformer — same as TimeAwareTransformer but parameterised
# ---------------------------------------------------------------------------

class AblationTransformer(nn.Module):
    """
    Ablatable version of TimeAwareTransformer.

    Args:
        n_input_channels : 74 (full) or 37 (no-missingness variant)
        pe_type          : "time_aware" | "standard"
    """

    def __init__(
        self,
        n_input_channels: int = 74,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        ff_dim: int = 256,
        dropout: float = 0.1,
        pe_type: str = "time_aware",
    ):
        super().__init__()
        self.d_model  = d_model
        self.n_layers = n_layers

        self.value_proj = nn.Linear(n_input_channels, d_model)

        if pe_type == "time_aware":
            self.time_pe = TimeAwarePositionalEncoding(d_model)
        else:
            self.time_pe = StandardPositionalEncoding(d_model)

        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            _TransformerLayer(d_model, n_heads, ff_dim, dropout)
            for _ in range(n_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)
        self.head     = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, hours: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        v      = self.value_proj(x)
        pe     = self.time_pe(hours)
        tokens = self.embed_drop(self.embed_norm(v + pe))

        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        for layer in self.layers:
            tokens = layer(tokens)

        cls_out = self.out_norm(tokens[:, 0, :])
        return torch.sigmoid(self.head(cls_out).squeeze(-1))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ablation_data(data_dir: str = "data/processed", use_missingness: bool = True):
    """
    Load data for one ablation variant.

    use_missingness=False  ->  37-channel input (values only)
    use_missingness=True   ->  74-channel input (values + mask)
    """
    X_train = np.load(os.path.join(data_dir, "X_train_norm.npy"))
    X_val   = np.load(os.path.join(data_dir, "X_val_norm.npy"))
    X_test  = np.load(os.path.join(data_dir, "X_test_norm.npy"))

    # zero-fill NaN (missing values are imputed to 0; mask encodes origin)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_val   = np.nan_to_num(X_val,   nan=0.0)
    X_test  = np.nan_to_num(X_test,  nan=0.0)

    if use_missingness:
        splits   = np.load(os.path.join(data_dir, "splits.npz"))
        mask_all = np.load(os.path.join(data_dir, "missingness_mask.npy")).astype(np.float32)
        mask_tr  = mask_all[splits["train_idx"]]
        mask_val = mask_all[splits["val_idx"]]
        mask_te  = mask_all[splits["test_idx"]]
        X_train  = np.concatenate([X_train, mask_tr],  axis=2)
        X_val    = np.concatenate([X_val,   mask_val], axis=2)
        X_test   = np.concatenate([X_test,  mask_te],  axis=2)

    X_train = X_train.astype(np.float32)
    X_val   = X_val.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    labels  = np.load(os.path.join(data_dir, "y_labels.npy"))
    splits  = np.load(os.path.join(data_dir, "splits.npz"))
    y_train = labels[splits["train_idx"]].astype(np.float32)
    y_val   = labels[splits["val_idx"]].astype(np.float32)
    y_test  = labels[splits["test_idx"]].astype(np.float32)

    T = X_train.shape[1]
    hours_base = np.arange(T, dtype=np.float32)
    h_train = np.tile(hours_base, (len(y_train), 1))
    h_val   = np.tile(hours_base, (len(y_val),   1))
    h_test  = np.tile(hours_base, (len(y_test),  1))

    return (
        torch.from_numpy(X_train), torch.from_numpy(X_val), torch.from_numpy(X_test),
        torch.from_numpy(y_train), torch.from_numpy(y_val), torch.from_numpy(y_test),
        torch.from_numpy(h_train), torch.from_numpy(h_val), torch.from_numpy(h_test),
    )


# ---------------------------------------------------------------------------
# Train one ablation variant
# ---------------------------------------------------------------------------

def run_variant(
    label: str,
    pe_type: str,
    use_missingness: bool,
    use_focal: bool,
    data_dir: str,
    device: torch.device,
    batch_size: int = 64,
    lr: float = 3e-4,
    weight_decay: float = 5e-4,
    n_epochs: int = 100,
    patience: int = 25,
    max_grad_norm: float = 1.0,
) -> dict:
    """Train and evaluate one variant. Returns metrics dict."""

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # data
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     h_train, h_val, h_test) = load_ablation_data(data_dir, use_missingness)

    n_channels = X_train.shape[2]
    pos_rate   = float(y_train.mean())

    # model
    torch.manual_seed(SEED)
    model = AblationTransformer(
        n_input_channels=n_channels,
        d_model=64, n_heads=4, n_layers=4, ff_dim=256, dropout=0.1,
        pe_type=pe_type,
    ).to(device)

    # loss
    if use_focal:
        criterion = FocalLoss(alpha=1.0 - pos_rate, gamma=2.0)
    else:
        criterion = nn.BCELoss()

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # loaders
    g = torch.Generator()
    g.manual_seed(SEED)
    train_loader = DataLoader(
        TensorDataset(X_train, y_train, h_train),
        batch_size=batch_size, shuffle=True, generator=g,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val, h_val),
        batch_size=batch_size, shuffle=False,
    )

    # training loop
    best_auprc   = -1.0
    best_epoch   = 0
    no_improve   = 0
    best_state   = None

    for epoch in range(1, n_epochs + 1):
        train_epoch(model, train_loader, optimiser, criterion, device, max_grad_norm)
        _, val_auprc, _, _ = evaluate(model, val_loader, criterion, device)

        if val_auprc > best_auprc:
            best_auprc  = val_auprc
            best_epoch  = epoch
            no_improve  = 0
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.load_state_dict(best_state)

    # test evaluation
    test_loader = DataLoader(
        TensorDataset(X_test, y_test, h_test),
        batch_size=batch_size, shuffle=False,
    )
    _, _, test_probs, test_labels = evaluate(model, test_loader, criterion, device)
    metrics = compute_test_metrics(test_probs, test_labels)

    print(
        f"  {label:<38}  "
        f"AUROC={metrics['auroc']:.4f}  "
        f"AUPRC={metrics['auprc']:.4f}  "
        f"Score1={metrics['score1']:.4f}  "
        f"HL-H={metrics['hosmer_lemeshow_h']:.1f}  "
        f"[epoch {best_epoch}]"
    )
    return metrics


# ---------------------------------------------------------------------------
# Results figure
# ---------------------------------------------------------------------------

def plot_ablation(results: dict, output_path: str):
    """
    2-panel figure:
      Left  — AUROC / AUPRC / Score1 (0-1 scale)
      Right — Hosmer-Lemeshow H (calibration; lower is better)
    """
    order  = ["full_model", "no_time_pe", "no_missingness", "no_focal_loss"]
    labels = {
        "full_model":      "Full Model\n(Proposed)",
        "no_time_pe":      "w/o Time-aware PE\n(Standard PE)",
        "no_missingness":  "w/o Missingness\nFeatures",
        "no_focal_loss":   "w/o Focal Loss\n(Standard BCE)",
    }
    colors = {
        "full_model":      "#2166ac",
        "no_time_pe":      "#d73027",
        "no_missingness":  "#f46d43",
        "no_focal_loss":   "#4dac26",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Ablation Study — Contribution of Each Design Component",
        fontsize=13, fontweight="bold",
    )

    x      = np.arange(len(order))
    width  = 0.22

    # ── left panel: AUROC / AUPRC / Score1 ──────────────────────────────
    metrics_left = ["auroc", "auprc", "score1"]
    metric_labels = ["AUROC", "AUPRC", "Score1"]
    offsets = [-width, 0, width]

    for mi, (mkey, mlabel, off) in enumerate(zip(metrics_left, metric_labels, offsets)):
        vals = [results[v][mkey] for v in order]
        bars = ax1.bar(
            x + off, vals,
            width=width * 0.92,
            label=mlabel,
            zorder=3,
        )
        for bar, val in zip(bars, vals):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=45,
            )

    # bold edge on full-model bars
    for off in offsets:
        ax1.bar(
            x[0] + off, 0,
            width=width * 0.92,
            edgecolor="black", linewidth=1.5,
            fill=False, zorder=4,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels([labels[v] for v in order], fontsize=9)
    ax1.set_ylabel("Score (higher is better)")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title("Discriminative & Clinical Utility Metrics")
    ax1.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # add full-model highlight band
    ax1.axvspan(-0.5, 0.5, color="#2166ac", alpha=0.06, zorder=0)

    # ── right panel: HL-H (lower = better calibration) ──────────────────
    hl_vals = [results[v]["hosmer_lemeshow_h"] for v in order]
    bar_colors = [colors[v] for v in order]
    bars2 = ax2.bar(x, hl_vals, color=bar_colors, width=0.5, zorder=3)

    for bar, val in zip(bars2, hl_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f"{val:.1f}",
            ha="center", va="bottom", fontsize=9,
        )

    # bold edge on full-model bar
    ax2.bar(x[0], hl_vals[0], width=0.5,
            edgecolor="black", linewidth=2, fill=False, zorder=4)

    ax2.set_xticks(x)
    ax2.set_xticklabels([labels[v] for v in order], fontsize=9)
    ax2.set_ylabel("HL-H statistic (lower is better)")
    ax2.set_title("Calibration (Hosmer-Lemeshow H)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax2.axvspan(-0.5, 0.5, color="#2166ac", alpha=0.06, zorder=0)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nFigure saved to {output_path}")


# ---------------------------------------------------------------------------
# Delta table helper
# ---------------------------------------------------------------------------

def compute_deltas(results: dict) -> dict:
    """Compute per-metric delta of each ablated variant vs full model."""
    full = results["full_model"]
    keys = ["auroc", "auprc", "score1", "hosmer_lemeshow_h"]
    deltas = {}
    for variant, metrics in results.items():
        if variant == "full_model":
            continue
        deltas[variant] = {k: round(metrics[k] - full[k], 4) for k in keys}
    return deltas


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

VARIANTS = [
    # (key,              label,                          pe_type,       use_miss, use_focal)
    ("full_model",       "Full Model (Proposed)",        "time_aware",  True,     True),
    ("no_time_pe",       "w/o Time-aware PE",            "standard",    True,     True),
    ("no_missingness",   "w/o Missingness Features",     "time_aware",  False,    True),
    ("no_focal_loss",    "w/o Focal Loss (BCE)",         "time_aware",  True,     False),
]


def main():
    data_dir    = "data/processed"
    out_dir     = "outputs"
    fig_dir     = "outputs/figures"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    results = {}
    print(f"{'Variant':<38}  {'AUROC':>7}  {'AUPRC':>7}  {'Score1':>7}  {'HL-H':>7}  Epoch")
    print("-" * 80)

    for key, label, pe_type, use_miss, use_focal in VARIANTS:
        metrics = run_variant(
            label=label,
            pe_type=pe_type,
            use_missingness=use_miss,
            use_focal=use_focal,
            data_dir=data_dir,
            device=device,
        )
        results[key] = metrics

    # ── summary table ───────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Variant':<28} {'AUROC':>7} {'AUPRC':>7} {'Score1':>7} {'HL-H':>8}")
    print("-" * 72)
    for key, label, *_ in VARIANTS:
        m = results[key]
        marker = " <-- proposed" if key == "full_model" else ""
        print(
            f"  {label:<26} {m['auroc']:>7.4f} {m['auprc']:>7.4f} "
            f"{m['score1']:>7.4f} {m['hosmer_lemeshow_h']:>8.1f}{marker}"
        )

    deltas = compute_deltas(results)
    print("\nDelta vs Full Model (ablated - full):")
    print(f"  {'Variant':<26} {'dAUROC':>8} {'dAUPRC':>8} {'dScore1':>8} {'dHL-H':>8}")
    print("-" * 65)
    for key, label, *_ in VARIANTS:
        if key == "full_model":
            continue
        d = deltas[key]
        print(
            f"  {label:<26} {d['auroc']:>+8.4f} {d['auprc']:>+8.4f} "
            f"{d['score1']:>+8.4f} {d['hosmer_lemeshow_h']:>+8.1f}"
        )

    # ── save JSON ────────────────────────────────────────────────────────────
    output = {
        "variants": results,
        "delta_vs_full_model": deltas,
        "seed": SEED,
        "notes": (
            "All variants trained with identical hyperparameters (lr=3e-4, wd=5e-4, "
            "batch=64, patience=25, max_grad_norm=1.0, seed=42). "
            "Early stopping on val AUPRC."
        ),
    }
    json_path = os.path.join(out_dir, "ablation_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")

    # ── figure ───────────────────────────────────────────────────────────────
    plot_ablation(results, os.path.join(fig_dir, "ablation_comparison.png"))

    print("\nAblation study complete.")
    return results


if __name__ == "__main__":
    main()
