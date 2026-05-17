"""
Final comparison — Chapter 5 results table for MS thesis.

Loads all three trained models, runs test-set inference to obtain raw
probabilities, then produces:
  - outputs/final_results_table.csv   (publication-ready comparison table)
  - outputs/figures/final_comparison.png  (ROC, PR, Score1 bar chart, table panel)

Models compared
  SAPS-I (traditional baseline)  — PhysioNet 2012 challenge reference
  XGBoost                        — classical ML with temporal statistics
  TCN                            — dilated causal CNN
  Transformer (proposed)         — time-aware Transformer (primary model)
"""

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, auc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── model imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from models.xgboost_model import flatten_temporal_features
from models.tcn_model import TCNMortality, load_tcn_data
from models.transformer_model import TimeAwareTransformer, load_transformer_data

# ── constants ─────────────────────────────────────────────────────────────────
SAPS_I_SCORE1 = 0.3097   # PhysioNet 2012 challenge paper Event 1 baseline
SAPS_I_HL     = 35.21    # PhysioNet 2012 challenge paper Event 2 baseline

MODEL_COLORS = {
    "SAPS-I\n(Baseline)":       "#7f7f7f",
    "XGBoost":                  "#2ca02c",
    "TCN":                      "#ff7f0e",
    "Transformer\n(Proposed)":  "#1f77b4",
}

MODEL_STYLES = {          # for ROC / PR line styles
    "XGBoost":     {"color": "#2ca02c", "lw": 2.0, "ls": "--"},
    "TCN":         {"color": "#ff7f0e", "lw": 2.0, "ls": "-."},
    "Transformer": {"color": "#1f77b4", "lw": 2.5, "ls": "-"},
}


# ---------------------------------------------------------------------------
# Inference helpers — return (probs, labels) numpy arrays on the test split
# ---------------------------------------------------------------------------

def get_xgboost_probs(data_dir: str, model_dir: str):
    """Reconstruct XGBoost test features from saved normalised arrays."""
    proc = os.path.join(data_dir, "processed")
    splits = np.load(os.path.join(proc, "splits.npz"))
    test_idx = splits["test_idx"]

    X_test_norm = np.load(os.path.join(proc, "X_test_norm.npy"))
    mask_full   = np.load(os.path.join(proc, "missingness_mask.npy"))
    mask_test   = mask_full[test_idx]

    # temporal statistics (NaN-aware — do NOT zero-fill before flattening)
    X_test_flat = flatten_temporal_features(X_test_norm, mask_test)

    # descriptor features (already normalised, saved per-split)
    desc = pd.read_csv(os.path.join(proc, "desc_test_norm.csv"))
    desc_cols = [c for c in desc.columns if c != "RecordID"]
    X_test_desc = desc[desc_cols].values

    X_test = np.hstack([X_test_flat, X_test_desc])

    # replace any remaining NaN from all-missing series with 0
    X_test = np.nan_to_num(X_test, nan=0.0)

    model = joblib.load(os.path.join(model_dir, "xgboost_model.pkl"))
    probs = model.predict_proba(X_test)[:, 1]

    labels = np.load(os.path.join(proc, "y_labels.npy"))[test_idx].astype(float)
    return probs.astype(np.float32), labels.astype(np.float32)


@torch.no_grad()
def get_tcn_probs(data_dir: str, model_dir: str):
    """Load saved TCN checkpoint and run test inference."""
    data = load_tcn_data(Path(data_dir))
    # load_tcn_data prints split sizes and returns a dict-like structure;
    # it returns: X_train, X_val, X_test, y_train, y_val, y_test as numpy
    # arrays via the clean() function. Unpack from the dict return value.
    # Actually load_tcn_data returns a dict: check its keys
    # Looking at the source it prints but doesn't return — re-load manually.
    proc = os.path.join(data_dir, "processed")
    splits = np.load(os.path.join(proc, "splits.npz"))
    test_idx = splits["test_idx"]

    X_test = np.load(os.path.join(proc, "X_test_norm.npy"))
    X_test = np.nan_to_num(X_test, nan=0.0)
    mask_full = np.load(os.path.join(proc, "missingness_mask.npy")).astype(np.float32)
    mask_test = mask_full[test_idx]
    X_test = np.concatenate([X_test, mask_test], axis=2).astype(np.float32)

    labels = np.load(os.path.join(proc, "y_labels.npy"))[test_idx].astype(np.float32)
    valid = ~np.isnan(labels)
    X_test, labels = X_test[valid], labels[valid]

    model = TCNMortality(n_variables=74, n_filters=64,
                         kernel_size=3, dilations=[1, 2, 4, 8], dropout=0.2)
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "tcn_model.pt"), map_location="cpu")
    )
    model.eval()

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_test)), batch_size=128, shuffle=False
    )
    probs = []
    for (xb,) in loader:
        probs.append(model(xb).numpy())
    return np.concatenate(probs), labels


@torch.no_grad()
def get_transformer_probs(data_dir: str, model_dir: str):
    """Load saved Transformer checkpoint and run test inference."""
    (_, _, X_test,
     _, _, y_test,
     _, _, h_test) = load_transformer_data(data_dir)

    model = TimeAwareTransformer(
        n_input_channels=74, d_model=64, n_heads=4,
        n_layers=4, ff_dim=256, dropout=0.1,
    )
    model.load_state_dict(
        torch.load(os.path.join(model_dir, "transformer_model.pt"), map_location="cpu")
    )
    model.eval()

    loader = DataLoader(
        TensorDataset(X_test, h_test), batch_size=128, shuffle=False
    )
    probs = []
    for xb, hb in loader:
        probs.append(model(xb, hb).numpy())
    return np.concatenate(probs), y_test.numpy()


# ---------------------------------------------------------------------------
# Metrics from raw probs (matches existing pipeline)
# ---------------------------------------------------------------------------

def compute_full_metrics(probs: np.ndarray, labels: np.ndarray) -> dict:
    auroc = roc_auc_score(labels, probs)

    prec_arr, rec_arr, thr_arr = precision_recall_curve(labels, probs)
    ap = auc(rec_arr, prec_arr)

    best_s1, best_se, best_pp, best_thr = 0.0, 0.0, 0.0, 0.5
    for thr, pr, rc in zip(thr_arr, prec_arr[:-1], rec_arr[:-1]):
        s1 = min(rc, pr)
        if s1 > best_s1:
            best_s1, best_se, best_pp, best_thr = s1, rc, pr, float(thr)

    bins = np.linspace(0, 1, 11)
    hl = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        idx = (probs >= lo) & (probs < hi)
        if idx.sum() == 0:
            continue
        n_k = idx.sum()
        o_k = labels[idx].sum()
        e_k = probs[idx].sum()
        hl += (o_k - e_k) ** 2 / (e_k * (1 - e_k / n_k) + 1e-8)

    return {
        "auroc":    round(float(auroc), 4),
        "auprc":    round(float(ap),    4),
        "sensitivity":           round(float(best_se), 4),
        "positive_predictivity": round(float(best_pp), 4),
        "score1":   round(float(best_s1), 4),
        "hl_h":     round(float(hl),      2),
    }


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def build_table(metrics: dict) -> pd.DataFrame:
    """
    Build a DataFrame with one row per model. SAPS-I row uses N/A for
    metrics the baseline paper does not report.
    """
    rows = []

    # SAPS-I baseline (PhysioNet 2012 challenge reference)
    rows.append({
        "Model":        "SAPS-I (Baseline)",
        "AUROC":        "—",
        "AUPRC":        "—",
        "Sensitivity":  "—",
        "+Predictivity":"—",
        "Score1 (E1)":  f"{SAPS_I_SCORE1:.4f}",
        "HL-H (E2)":    f"{SAPS_I_HL:.2f}",
    })

    order = [
        ("XGBoost",     "XGBoost"),
        ("TCN",         "TCN"),
        ("Transformer", "Transformer (Proposed)"),
    ]
    for key, label in order:
        m = metrics[key]
        rows.append({
            "Model":         label,
            "AUROC":         f"{m['auroc']:.4f}",
            "AUPRC":         f"{m['auprc']:.4f}",
            "Sensitivity":   f"{m['sensitivity']:.4f}",
            "+Predictivity": f"{m['positive_predictivity']:.4f}",
            "Score1 (E1)":   f"{m['score1']:.4f}",
            "HL-H (E2)":     f"{m['hl_h']:.2f}",
        })

    return pd.DataFrame(rows)


def annotate_best(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy with '*' appended to the best numeric value in each
    metric column (highest for AUROC/AUPRC/Se/+P/Score1; lowest for HL-H).
    """
    df = df.copy()
    higher_is_better = ["AUROC", "AUPRC", "Sensitivity", "+Predictivity", "Score1 (E1)"]
    lower_is_better  = ["HL-H (E2)"]

    for col in higher_is_better + lower_is_better:
        numeric_rows = []
        for i, val in enumerate(df[col]):
            try:
                numeric_rows.append((i, float(val)))
            except (ValueError, TypeError):
                pass
        if not numeric_rows:
            continue
        if col in higher_is_better:
            best_i = max(numeric_rows, key=lambda x: x[1])[0]
        else:
            best_i = min(numeric_rows, key=lambda x: x[1])[0]
        df.loc[best_i, col] = df.loc[best_i, col] + " *"

    return df


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figure(
    probs_dict: dict,   # {"XGBoost": (probs, labels), "TCN": ..., "Transformer": ...}
    metrics:    dict,   # {"XGBoost": {...}, "TCN": {...}, "Transformer": {...}}
    table_df:   pd.DataFrame,
    out_path:   str,
):
    fig = plt.figure(figsize=(20, 14))
    gs  = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    ax_roc  = fig.add_subplot(gs[0, 0])
    ax_pr   = fig.add_subplot(gs[0, 1])
    ax_bar  = fig.add_subplot(gs[1, 0])
    ax_tbl  = fig.add_subplot(gs[1, 1])

    # ── ROC curves ────────────────────────────────────────────────────────
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.50)")
    for name, (probs, labels) in probs_dict.items():
        fpr, tpr, _ = roc_curve(labels, probs)
        roc_auc = roc_auc_score(labels, probs)
        st = MODEL_STYLES[name]
        ax_roc.plot(fpr, tpr, label=f"{name}  (AUC={roc_auc:.4f})",
                    color=st["color"], lw=st["lw"], linestyle=st["ls"])
    ax_roc.set_xlabel("False Positive Rate", fontsize=11)
    ax_roc.set_ylabel("True Positive Rate",  fontsize=11)
    ax_roc.set_title("ROC Curves — All Models", fontsize=12, fontweight="bold")
    ax_roc.legend(fontsize=9, loc="lower right")
    ax_roc.grid(True, alpha=0.3)
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.02)

    # ── Precision-Recall curves ────────────────────────────────────────────
    pos_rate = float(next(iter(probs_dict.values()))[1].mean())
    ax_pr.axhline(pos_rate, color="k", ls="--", lw=1,
                  label=f"Baseline (prevalence={pos_rate:.3f})")
    for name, (probs, labels) in probs_dict.items():
        pr, rc, _ = precision_recall_curve(labels, probs)
        ap = auc(rc, pr)
        st = MODEL_STYLES[name]
        ax_pr.plot(rc, pr, label=f"{name}  (AP={ap:.4f})",
                   color=st["color"], lw=st["lw"], linestyle=st["ls"])
    ax_pr.set_xlabel("Recall",    fontsize=11)
    ax_pr.set_ylabel("Precision", fontsize=11)
    ax_pr.set_title("Precision-Recall Curves — All Models",
                    fontsize=12, fontweight="bold")
    ax_pr.legend(fontsize=9, loc="upper right")
    ax_pr.grid(True, alpha=0.3)
    ax_pr.set_xlim(0, 1); ax_pr.set_ylim(0, 1.02)

    # ── Score1 bar chart with SAPS-I baseline ─────────────────────────────
    bar_names  = ["SAPS-I\n(Baseline)", "XGBoost", "TCN", "Transformer\n(Proposed)"]
    bar_values = [
        SAPS_I_SCORE1,
        metrics["XGBoost"]["score1"],
        metrics["TCN"]["score1"],
        metrics["Transformer"]["score1"],
    ]
    bar_colors = [MODEL_COLORS[n] for n in bar_names]
    bars = ax_bar.bar(bar_names, bar_values, color=bar_colors, width=0.5,
                      edgecolor="black", linewidth=0.8, zorder=3)

    # annotate values
    for bar, val in zip(bars, bar_values):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

    # SAPS-I reference line
    ax_bar.axhline(SAPS_I_SCORE1, color="#7f7f7f", linestyle=":",
                   linewidth=1.4, label=f"SAPS-I baseline ({SAPS_I_SCORE1})")

    ax_bar.set_ylabel("Score1 = min(Sensitivity, +Predictivity)", fontsize=10)
    ax_bar.set_title("Score1 Comparison (PhysioNet Event 1 Metric)",
                     fontsize=12, fontweight="bold")
    ax_bar.set_ylim(0, max(bar_values) + 0.12)
    ax_bar.legend(fontsize=9)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)

    # best model star
    best_idx = int(np.argmax(bar_values))
    ax_bar.text(best_idx, bar_values[best_idx] + 0.03, "★ Best",
                ha="center", fontsize=10, color="black", fontweight="bold")

    # ── Results table ─────────────────────────────────────────────────────
    ax_tbl.axis("off")
    starred = annotate_best(table_df)

    col_widths = [0.28, 0.10, 0.10, 0.12, 0.13, 0.12, 0.12]
    tbl = ax_tbl.table(
        cellText=starred.values,
        colLabels=starred.columns,
        loc="center",
        cellLoc="center",
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 2.0)

    # header row style
    for j in range(len(starred.columns)):
        tbl[(0, j)].set_facecolor("#2166ac")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Transformer row highlight (proposed model)
    trans_row = 4   # rows: header=0, SAPS-I=1, XGBoost=2, TCN=3, Transformer=4
    for j in range(len(starred.columns)):
        tbl[(trans_row, j)].set_facecolor("#ddeeff")

    # bold best-value cells (those with " *")
    for i in range(1, len(starred) + 1):
        for j in range(len(starred.columns)):
            if str(tbl[(i, j)].get_text().get_text()).endswith(" *"):
                tbl[(i, j)].set_text_props(fontweight="bold")
                tbl[(i, j)].set_facecolor("#ffffcc")

    ax_tbl.set_title("Comparison Table — All Models",
                     fontsize=12, fontweight="bold", pad=12)
    ax_tbl.text(0.5, -0.04,
                "* = best in column   | highlighted row = proposed model",
                ha="center", transform=ax_tbl.transAxes,
                fontsize=8, style="italic", color="#444444")

    fig.suptitle(
        "ICU Mortality Prediction — Final Model Comparison\n"
        "PhysioNet 2012 Set-A  |  600-patient test set",
        fontsize=14, fontweight="bold", y=1.01,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_dir  = "data"
    proc_dir  = "data/processed"
    model_dir = "outputs/models"
    out_dir   = "outputs"
    fig_dir   = "outputs/figures"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 65)
    print("Final Comparison — MS Thesis Chapter 5")
    print("=" * 65)

    # ── load saved summary metrics (for display consistency) ─────────────
    with open(os.path.join(out_dir, "xgboost_results.json"))     as f: xgb_json = json.load(f)
    with open(os.path.join(out_dir, "tcn_results.json"))          as f: tcn_json = json.load(f)
    with open(os.path.join(out_dir, "transformer_results.json"))  as f: trn_json = json.load(f)

    # ── get raw probabilities (needed for ROC / PR curves) ────────────────
    print("\n[1/4] Running inference on test set...")

    print("  XGBoost  ... ", end="", flush=True)
    xgb_probs, xgb_labels = get_xgboost_probs(data_dir, model_dir)
    print(f"done  ({len(xgb_labels)} patients)")

    print("  TCN      ... ", end="", flush=True)
    tcn_probs, tcn_labels = get_tcn_probs(data_dir, model_dir)
    print(f"done  ({len(tcn_labels)} patients)")

    print("  Transformer ... ", end="", flush=True)
    trn_probs, trn_labels = get_transformer_probs(proc_dir, model_dir)
    print(f"done  ({len(trn_labels)} patients)")

    # ── recompute metrics from live probs (source of truth for curves) ────
    print("\n[2/4] Computing metrics from live probabilities...")
    metrics = {
        "XGBoost":     compute_full_metrics(xgb_probs, xgb_labels),
        "TCN":         compute_full_metrics(tcn_probs, tcn_labels),
        "Transformer": compute_full_metrics(trn_probs, trn_labels),
    }

    # ── print summary table ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"{'Model':<26} {'AUROC':>7} {'AUPRC':>7} {'Se':>7} {'+P':>7} {'Score1':>7} {'HL-H':>8}")
    print("-" * 72)
    print(f"  {'SAPS-I (Baseline)':<24} {'—':>7} {'—':>7} {'—':>7} {'—':>7} "
          f"{SAPS_I_SCORE1:>7.4f} {SAPS_I_HL:>8.2f}")
    for name in ["XGBoost", "TCN", "Transformer"]:
        m = metrics[name]
        tag = "  <-- proposed" if name == "Transformer" else ""
        print(
            f"  {name:<24} {m['auroc']:>7.4f} {m['auprc']:>7.4f} "
            f"{m['sensitivity']:>7.4f} {m['positive_predictivity']:>7.4f} "
            f"{m['score1']:>7.4f} {m['hl_h']:>8.2f}{tag}"
        )

    # ── build and save CSV ────────────────────────────────────────────────
    print("\n[3/4] Saving results table...")
    table_df = build_table(metrics)
    starred  = annotate_best(table_df)

    csv_path = os.path.join(out_dir, "final_results_table.csv")
    starred.to_csv(csv_path, index=False)
    print(f"  CSV saved: {csv_path}")

    # ── generate figure ───────────────────────────────────────────────────
    print("\n[4/4] Generating figures...")
    probs_dict = {
        "XGBoost":     (xgb_probs, xgb_labels),
        "TCN":         (tcn_probs, tcn_labels),
        "Transformer": (trn_probs, trn_labels),
    }
    make_figure(
        probs_dict, metrics, table_df,
        out_path=os.path.join(fig_dir, "final_comparison.png"),
    )

    # ── save full JSON for downstream use ─────────────────────────────────
    output = {
        "saps_i_baseline": {"score1": SAPS_I_SCORE1, "hl_h": SAPS_I_HL},
        "models": metrics,
        "note": (
            "Metrics recomputed from live inference on the test split. "
            "SAPS-I values from the PhysioNet 2012 challenge paper (Goldberger et al.)."
        ),
    }
    with open(os.path.join(out_dir, "final_comparison_metrics.json"), "w") as f:
        json.dump(output, f, indent=2)

    print("\nDone. Outputs:")
    print(f"  outputs/final_results_table.csv")
    print(f"  outputs/figures/final_comparison.png")
    print(f"  outputs/final_comparison_metrics.json")
    return metrics


if __name__ == "__main__":
    main()
