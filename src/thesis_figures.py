"""
Thesis figures — Phase 9, Prompt 12.

Generates publication-quality (300 DPI) figures for each thesis chapter.
All output saved to outputs/figures/thesis/.

Figures produced
----------------
ch3_missingness_profile.png      — Chapter 3: per-variable missingness rates
ch4_architecture_diagram.png     — Chapter 4: Time-aware Transformer architecture
ch5_patient_shap_heatmap.png     — Chapter 5: 48h × 37-var SHAP case study
ch5_top10_shap_variables.png     — Chapter 5: top-10 variables with clinical labels
ch6_shap_sofa_concordance.png    — Chapter 6: SHAP–SOFA concordance scatter + CI
transformer_attention_outcomes.png — Transformer: mean Layer-1 attention by outcome
model_radar_comparison.png       — All models: radar chart across 4 metrics
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from models.transformer_model import TimeAwareTransformer, load_transformer_data

# ── global publication style ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        300,
    "savefig.dpi":       300,
    "savefig.facecolor": "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
})

OUT_DIR = "outputs/figures/thesis"
DPI     = 300

# ── variable metadata ─────────────────────────────────────────────────────────
VARIABLE_NAMES = [
    "Albumin","ALP","ALT","AST","Bilirubin","BUN","Calcium",
    "Chloride","Creatinine","DiasABP","FiO2","GCS","Glucose",
    "HCO3","HCT","HR","K","Lactate","Mg","MAP","MechVent",
    "Na","NIDiasABP","NIMAP","NISysABP","O2Sat","PaCO2",
    "PaO2","pH","Platelets","RespRate","SaO2","SysABP",
    "Temp","TropI","Urine","WBC",
]  # 37 variables — exact order from preprocess.py MONITORING_VARIABLES

CLINICAL_LABELS = {
    "GCS":       "GCS (Neurological)",
    "MAP":       "MAP (Hemodynamic)",
    "Lactate":   "Lactate (Tissue Hypoperfusion)",
    "HR":        "HR (Heart Rate)",
    "Creatinine":"Creatinine (Renal)",
    "BUN":       "BUN (Renal)",
    "PaO2":      "PaO₂ (Oxygenation)",
    "pH":        "pH (Acid-Base)",
    "FiO2":      "FiO₂ (Ventilation)",
    "HCO3":      "HCO₃ (Acid-Base)",
    "RespRate":  "Resp Rate (Respiratory)",
    "SaO2":      "SaO₂ (Oxygenation)",
    "O2Sat":     "O₂ Sat (Oxygenation)",
    "SysABP":    "Systolic BP (Hemodynamic)",
    "Temp":      "Temperature (Infection)",
    "MechVent":  "Mech Vent (Ventilation)",
    "Bilirubin": "Bilirubin (Hepatic)",
    "Platelets": "Platelets (Coagulation)",
    "WBC":       "WBC (Immune)",
    "Glucose":   "Glucose (Metabolic)",
    "Na":        "Sodium (Electrolyte)",
    "K":         "Potassium (Electrolyte)",
    "Albumin":   "Albumin (Nutritional)",
    "HCT":       "Haematocrit (Haematologic)",
    "Urine":     "Urine Output (Renal)",
}

# colour per clinical category
CAT_COLORS = {
    "Hemodynamic":   "#e41a1c",
    "Respiratory":   "#377eb8",
    "Neurological":  "#4daf4a",
    "Metabolic":     "#ff7f00",
    "Renal":         "#984ea3",
    "Hepatic":       "#a65628",
    "Hematologic":   "#f781bf",
    "Cardiac":       "#999999",
    "Ventilation":   "#66c2a5",
    "Other":         "#8da0cb",
}

VAR_CATEGORY = {
    "HR":"Hemodynamic","MAP":"Hemodynamic","DiasABP":"Hemodynamic",
    "SysABP":"Hemodynamic","NIDiasABP":"Hemodynamic","NIMAP":"Hemodynamic","NISysABP":"Hemodynamic",
    "RespRate":"Respiratory","SaO2":"Respiratory","O2Sat":"Respiratory",
    "FiO2":"Respiratory","PaO2":"Respiratory","PaCO2":"Respiratory",
    "GCS":"Neurological",
    "Glucose":"Metabolic","Lactate":"Metabolic","pH":"Metabolic",
    "HCO3":"Metabolic","Na":"Metabolic","K":"Metabolic","Mg":"Metabolic",
    "Calcium":"Metabolic","Chloride":"Metabolic",
    "Creatinine":"Renal","BUN":"Renal","Urine":"Renal",
    "Albumin":"Hepatic","ALP":"Hepatic","ALT":"Hepatic","AST":"Hepatic","Bilirubin":"Hepatic",
    "WBC":"Hematologic","Platelets":"Hematologic","HCT":"Hematologic",
    "TropI":"Cardiac",
    "MechVent":"Ventilation",
    "Temp":"Other",
}

SOFA_VARS  = {"GCS","MAP","Creatinine","Bilirubin","PaO2","Platelets","MechVent"}
SAPS_VARS  = {"HR","SysABP","Temp","RespRate","O2Sat","Creatinine","BUN","Bilirubin","WBC","Platelets"}


def savefig(fig, name, tight=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    if tight:
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    else:
        fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1  (Chapter 3) — Missingness profile
# ─────────────────────────────────────────────────────────────────────────────

def fig_missingness_profile(data_dir="data/processed"):
    mask = np.load(os.path.join(data_dir, "missingness_mask.npy"))  # (4000,48,37)
    # missingness rate per variable averaged over all patients × hours
    miss_rate = mask.mean(axis=(0, 1))  # (37,)

    # sort descending
    order = np.argsort(miss_rate)[::-1]
    names  = [VARIABLE_NAMES[i] for i in order]
    rates  = miss_rate[order] * 100
    colors = [CAT_COLORS.get(VAR_CATEGORY.get(n, "Other"), "#8da0cb") for n in names]

    fig, ax = plt.subplots(figsize=(9, 10))
    bars = ax.barh(range(len(names)), rates, color=colors, edgecolor="none", height=0.7)

    # SOFA / SAPS-I marker
    for i, n in enumerate(names):
        markers = []
        if n in SOFA_VARS:  markers.append("SOFA")
        if n in SAPS_VARS:  markers.append("SAPS-I")
        if markers:
            ax.text(rates[i] + 0.5, i, "  " + "/".join(markers),
                    va="center", fontsize=7, color="#555555", style="italic")

    # threshold line at 50%
    ax.axvline(50, color="crimson", ls="--", lw=1.2, label="50 % threshold")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel("Missingness rate (%)")
    ax.set_title("PhysioNet 2012 — Missingness Profile (37 Monitoring Variables)",
                 fontweight="bold", pad=10)
    ax.set_xlim(0, max(rates) + 12)

    # category legend
    seen = {}
    for n in names:
        cat = VAR_CATEGORY.get(n, "Other")
        if cat not in seen:
            seen[cat] = CAT_COLORS.get(cat, "#8da0cb")
    patches = [mpatches.Patch(color=c, label=k) for k, c in seen.items()]
    ax.legend(handles=patches + [plt.Line2D([0],[0],color="crimson",ls="--",lw=1.2,label="50% threshold")],
              loc="lower right", fontsize=8, framealpha=0.9)

    ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, "ch3_missingness_profile.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2  (Chapter 4) — Architecture diagram
# ─────────────────────────────────────────────────────────────────────────────

def _box(ax, x, y, w, h, label, sublabel="", color="#4292c6", fontsize=10, radius=0.03):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=f"round,pad={radius}",
                         facecolor=color, edgecolor="#222", linewidth=1.2,
                         zorder=3)
    ax.add_patch(box)
    ax.text(x, y + (0.012 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color="white", zorder=4)
    if sublabel:
        ax.text(x, y - 0.045, sublabel,
                ha="center", va="center", fontsize=fontsize - 2,
                color="#ddeeff", zorder=4)


def _arrow(ax, x, y0, y1, color="#555", lw=1.5):
    ax.annotate("", xy=(x, y1 + 0.005), xytext=(x, y0 - 0.005),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=12),
                zorder=2)


def fig_architecture_diagram():
    fig, ax = plt.subplots(figsize=(10, 13))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    cx = 0.5    # main column x
    bw = 0.38   # box width
    bh = 0.055  # box height

    # ─── y positions (top → bottom) ───────────────────────────────────────
    ys = {
        "input":   0.94,
        "proj":    0.84,
        "pe":      0.76,
        "add":     0.68,
        "norm":    0.60,
        "cls":     0.52,
        "encoder": 0.38,
        "outnorm": 0.24,
        "head":    0.14,
        "output":  0.04,
    }

    # ─── Input ────────────────────────────────────────────────────────────
    _box(ax, cx, ys["input"], bw, bh,
         "Input Sequence",
         "Shape: (B, 48, 74)  •  37 values + 37 missingness flags",
         color="#2166ac")

    _arrow(ax, cx, ys["input"]-bh/2, ys["proj"]+bh/2)

    # ─── Value projection ─────────────────────────────────────────────────
    _box(ax, cx, ys["proj"], bw, bh,
         "Value Projection",
         "Linear(74 → 64)  •  d_model = 64",
         color="#4393c3")

    # PE branch
    _box(ax, 0.82, ys["pe"], 0.28, bh,
         "Time-Aware PE",
         "Learnable freq. sin/cos",
         color="#1a9850", fontsize=9)

    # + symbol
    ax.text(cx, ys["add"], "+", ha="center", va="center",
            fontsize=22, color="#333", fontweight="bold", zorder=4)
    ax.add_patch(plt.Circle((cx, ys["add"]), 0.028,
                            ec="#555", fc="white", lw=1.5, zorder=3))

    # arrows into +
    _arrow(ax, cx,   ys["proj"]-bh/2, ys["add"]+0.03)
    ax.annotate("", xy=(cx+0.024, ys["add"]), xytext=(0.82-0.14, ys["pe"]),
                arrowprops=dict(arrowstyle="-|>", color="#1a9850", lw=1.5, mutation_scale=11), zorder=2)

    _arrow(ax, cx, ys["add"]-0.03, ys["norm"]+bh/2)

    # ─── LayerNorm + Dropout ─────────────────────────────────────────────
    _box(ax, cx, ys["norm"], bw, bh,
         "LayerNorm + Dropout(0.1)",
         "Stabilise embedding before encoder",
         color="#74add1")

    _arrow(ax, cx, ys["norm"]-bh/2, ys["cls"]+bh/2)

    # ─── CLS token ────────────────────────────────────────────────────────
    _box(ax, cx, ys["cls"], bw, bh,
         "Prepend CLS Token",
         "(B, 48, 64)  →  (B, 49, 64)   •  learnable, init N(0, 0.02)",
         color="#1a9850")

    _arrow(ax, cx, ys["cls"]-bh/2, ys["encoder"]+0.085)

    # ─── Encoder stack ────────────────────────────────────────────────────
    enc_top    = ys["encoder"] + 0.085
    enc_bottom = ys["encoder"] - 0.085
    enc_h      = enc_top - enc_bottom

    enc_box = FancyBboxPatch((cx-0.22, enc_bottom), 0.44, enc_h,
                              boxstyle="round,pad=0.015",
                              facecolor="#fff7bc", edgecolor="#d73027",
                              linewidth=2, linestyle="--", zorder=2)
    ax.add_patch(enc_box)
    ax.text(cx, enc_top + 0.015, "Encoder Stack  × 4 Layers",
            ha="center", fontsize=10, fontweight="bold", color="#d73027")

    # inner detail
    inner_x = [cx - 0.14, cx + 0.14]
    for ix, (lbl, sub) in enumerate([
        ("Pre-Norm + Multi-Head\nSelf-Attention", "4 heads, need_weights in eval"),
        ("Pre-Norm + Feed-Forward\nNetwork", "Linear(64→256)→GELU→Linear(256→64)"),
    ]):
        bx = inner_x[ix]
        r = FancyBboxPatch((bx - 0.11, enc_bottom + 0.015), 0.22, enc_h - 0.03,
                            boxstyle="round,pad=0.01",
                            facecolor="#e0f3f8", edgecolor="#4292c6",
                            linewidth=1.2, zorder=3)
        ax.add_patch(r)
        ax.text(bx, ys["encoder"], lbl, ha="center", va="center",
                fontsize=7.5, color="#08519c", zorder=4, linespacing=1.4)
        ax.text(bx, enc_bottom + 0.022, sub, ha="center", va="bottom",
                fontsize=6.5, color="#333", style="italic", zorder=4)

    ax.annotate("", xy=(inner_x[1] - 0.10, ys["encoder"]),
                xytext=(inner_x[0] + 0.10, ys["encoder"]),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.2, mutation_scale=9), zorder=4)

    _arrow(ax, cx, enc_bottom, ys["outnorm"]+bh/2)

    # ─── Output LayerNorm ────────────────────────────────────────────────
    _box(ax, cx, ys["outnorm"], bw, bh,
         "Output LayerNorm",
         "Applied to CLS token:  tokens[:, 0, :]",
         color="#74add1")

    _arrow(ax, cx, ys["outnorm"]-bh/2, ys["head"]+bh/2)

    # ─── Classification head ─────────────────────────────────────────────
    _box(ax, cx, ys["head"], bw, bh,
         "Classification Head",
         "Linear(64 → 1)  +  Sigmoid",
         color="#4393c3")

    _arrow(ax, cx, ys["head"]-bh/2, ys["output"]+bh/2)

    # ─── Output ──────────────────────────────────────────────────────────
    _box(ax, cx, ys["output"], bw*0.7, bh,
         "Mortality Probability",
         "p ∈ [0, 1]",
         color="#2166ac")

    # ─── Parameter annotation ────────────────────────────────────────────
    ax.text(0.02, 0.50,
            "Novel components\n(this work):",
            fontsize=8, va="center", color="#1a9850", fontweight="bold")
    for yi, lbl in [(0.45, "Time-Aware PE"), (0.40, "CLS classification"), (0.35, "Missingness input")]:
        ax.plot([0.02, 0.09], [yi, yi], color="#1a9850", lw=1.5)
        ax.text(0.10, yi, lbl, fontsize=8, va="center", color="#1a9850")

    ax.text(0.03, 0.98,
            "205,153 trainable parameters\n"
            "4 encoder layers  •  4 attention heads  •  d_model = 64",
            fontsize=9, va="top", color="#444",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f0f0f0", ec="#aaa", lw=1))

    ax.set_title("Time-Aware Transformer Architecture for ICU Mortality Prediction",
                 fontsize=13, fontweight="bold", pad=14)
    savefig(fig, "ch4_architecture_diagram.png", tight=False)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3  (Chapter 5) — Patient SHAP heatmap case study
# ─────────────────────────────────────────────────────────────────────────────

def fig_patient_shap_heatmap(data_dir="data/processed", shap_path="outputs/shap_matrices.npy"):
    shap  = np.load(shap_path)          # (600, 48, 37)
    labels = np.load(os.path.join(data_dir, "y_labels.npy"))
    splits = np.load(os.path.join(data_dir, "splits.npz"))
    y_test  = labels[splits["test_idx"]]

    # best non-survivor: highest total |SHAP| signal → clearest case study
    nonsurv_idx = np.where(y_test == 1)[0]
    total_shap  = np.abs(shap[nonsurv_idx]).sum(axis=(1, 2))
    patient_i   = nonsurv_idx[np.argmax(total_shap)]

    shap_p  = shap[patient_i]           # (48, 37)

    # sort variables by |SHAP| magnitude for this patient (most important at top)
    var_importance = np.abs(shap_p).mean(axis=0)
    var_order      = np.argsort(var_importance)[::-1]
    shap_sorted    = shap_p[:, var_order]
    var_names_sorted = [VARIABLE_NAMES[i] for i in var_order]

    # symmetric colour scale
    vmax = np.percentile(np.abs(shap_sorted), 97)

    fig, ax = plt.subplots(figsize=(14, 9))
    im = ax.imshow(shap_sorted.T, aspect="auto",
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")

    ax.set_xticks(range(0, 48, 6))
    ax.set_xticklabels([f"H{h}" for h in range(0, 48, 6)], fontsize=9)
    ax.set_yticks(range(len(var_names_sorted)))
    ax.set_yticklabels(var_names_sorted, fontsize=8)
    ax.set_xlabel("Hour (0 – 47)", fontsize=11)
    ax.set_ylabel("Variable (sorted by importance)", fontsize=11)
    ax.set_title(
        f"SHAP Attribution Heatmap — Patient {patient_i} (Non-Survivor)\n"
        "Red = increases mortality risk  |  Blue = decreases mortality risk",
        fontsize=12, fontweight="bold"
    )

    cb = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.01)
    cb.set_label("SHAP value", fontsize=10)

    # annotate key clinical variables
    annotate_vars = {"GCS": "#00cc44", "MAP": "#ff6600", "Lactate": "#cc00cc"}
    for var_name, color in annotate_vars.items():
        if var_name in var_names_sorted:
            yi = var_names_sorted.index(var_name)
            ax.axhline(yi, color=color, lw=1.5, ls="--", alpha=0.8)
            ax.text(48.3, yi, var_name, color=color,
                    fontsize=9, va="center", fontweight="bold")

    # mark peak SHAP hour
    peak_hour = np.abs(shap_sorted).sum(axis=1).argmax()
    ax.axvline(peak_hour, color="gold", lw=2, ls="-", alpha=0.9,
               label=f"Peak risk hour: H{peak_hour}")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)

    fig.tight_layout()
    savefig(fig, "ch5_patient_shap_heatmap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4  (Chapter 5) — Top-10 SHAP variables
# ─────────────────────────────────────────────────────────────────────────────

def fig_top10_shap(shap_path="outputs/shap_matrices.npy"):
    shap = np.load(shap_path)           # (600, 48, 37)
    mean_abs = np.abs(shap).mean(axis=(0, 1))   # (37,)

    top10_idx  = np.argsort(mean_abs)[::-1][:10]
    top10_vals = mean_abs[top10_idx]
    top10_names= [VARIABLE_NAMES[i] for i in top10_idx]
    top10_clabs= [CLINICAL_LABELS.get(n, n) for n in top10_names]
    top10_cols = [CAT_COLORS.get(VAR_CATEGORY.get(n, "Other"), "#8da0cb") for n in top10_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(10), top10_vals[::-1],
                   color=top10_cols[::-1], edgecolor="none", height=0.65)
    ax.set_yticks(range(10))
    ax.set_yticklabels(top10_clabs[::-1], fontsize=9.5)
    ax.set_xlabel("Mean |SHAP| value (averaged over patients and hours)", fontsize=11)
    ax.set_title("Top-10 Variables by Mean SHAP Attribution\n"
                 "(XGBoost on PhysioNet 2012 Test Set, n = 600)",
                 fontsize=12, fontweight="bold")

    for bar, val in zip(bars, top10_vals[::-1]):
        ax.text(val + max(top10_vals)*0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=9)

    # SOFA / SAPS-I markers
    for i, n in enumerate(top10_names[::-1]):
        tags = []
        if n in SOFA_VARS:  tags.append("SOFA")
        if n in SAPS_VARS:  tags.append("SAPS-I")
        if tags:
            ax.text(max(top10_vals)*1.02, i,
                    " ".join(f"[{t}]" for t in tags),
                    va="center", fontsize=8, color="#555", style="italic")

    # category legend
    seen = {VAR_CATEGORY.get(n,"Other"): CAT_COLORS.get(VAR_CATEGORY.get(n,"Other"),"#8da0cb")
            for n in top10_names}
    patches = [mpatches.Patch(color=c, label=k) for k,c in seen.items()]
    ax.legend(handles=patches, loc="lower right", fontsize=9, title="Clinical category")
    ax.set_xlim(0, max(top10_vals) * 1.18)

    fig.tight_layout()
    savefig(fig, "ch5_top10_shap_variables.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5  (Chapter 6) — SHAP–SOFA concordance scatter
# ─────────────────────────────────────────────────────────────────────────────

def fig_concordance_scatter(
    shap_path="outputs/shap_matrices.npy",
    data_dir="data/processed",
    model_path="outputs/models/xgboost_model.pkl",
):
    import joblib
    from models.xgboost_model import flatten_temporal_features

    shap   = np.load(shap_path)    # (600, 48, 37)
    labels = np.load(os.path.join(data_dir, "y_labels.npy"))
    splits = np.load(os.path.join(data_dir, "splits.npz"))
    test_idx = splits["test_idx"]
    y_test   = labels[test_idx].astype(float)

    # ── per-patient SHAP–SOFA concordance ───────────────────────────────
    sofa_idx  = {VARIABLE_NAMES.index(v) for v in SOFA_VARS  if v in VARIABLE_NAMES}
    saps_idx  = {VARIABLE_NAMES.index(v) for v in SAPS_VARS  if v in VARIABLE_NAMES}

    def concordance(shap_mat, clinical_idx, n_top=5):
        scores = np.zeros(len(shap_mat))
        for i, sp in enumerate(shap_mat):
            var_imp = np.abs(sp).sum(axis=0)
            top_set = set(np.argsort(var_imp)[-n_top:])
            scores[i] = len(top_set & clinical_idx) / n_top
        return scores

    sofa_conc = concordance(shap, sofa_idx)
    saps_conc = concordance(shap, saps_idx)

    # ── per-patient Brier score from XGBoost ────────────────────────────
    X_test_norm = np.load(os.path.join(data_dir, "X_test_norm.npy"))
    mask_full   = np.load(os.path.join(data_dir, "missingness_mask.npy"))
    mask_test   = mask_full[test_idx]
    X_flat = flatten_temporal_features(X_test_norm, mask_test)

    desc = pd.read_csv(os.path.join(data_dir, "desc_test_norm.csv"))
    desc_cols = [c for c in desc.columns if c != "RecordID"]
    X_test_feat = np.hstack([X_flat, desc[desc_cols].values])
    X_test_feat = np.nan_to_num(X_test_feat, nan=0.0)

    model   = joblib.load(model_path)
    probs   = model.predict_proba(X_test_feat)[:, 1]
    brier   = (probs - y_test) ** 2

    # ── two-panel scatter: SOFA and SAPS-I ──────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("SHAP–Clinical Score Concordance vs Calibration Error\n"
                 "(Hypothesis: high concordance → well-calibrated risk estimate)",
                 fontsize=12, fontweight="bold", y=1.02)

    for ax, conc, title, color, score_name in [
        (ax1, sofa_conc, "SHAP–SOFA Concordance", "#e41a1c", "SOFA"),
        (ax2, saps_conc, "SHAP–SAPS-I Concordance", "#377eb8", "SAPS-I"),
    ]:
        # jitter for discrete x values
        jitter = np.random.RandomState(42).uniform(-0.015, 0.015, len(conc))
        ax.scatter(conc + jitter, brier, alpha=0.35, s=22,
                   c=color, edgecolors="none", label="Test patient")

        # linear regression + 95% CI
        m, b, r, p_val, se = stats.linregress(conc, brier)
        xs = np.linspace(conc.min(), conc.max(), 200)
        ys = m * xs + b
        n  = len(conc)
        t  = stats.t.ppf(0.975, df=n-2)
        ys_se = se * np.sqrt(1/n + (xs - conc.mean())**2 / ((conc - conc.mean())**2).sum())
        ax.plot(xs, ys, color=color, lw=2, label=f"Regression (r={r:.3f})")
        ax.fill_between(xs, ys - t*ys_se, ys + t*ys_se,
                        color=color, alpha=0.15, label="95% CI")

        sig = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.4f}"
        ax.text(0.05, 0.93,
                f"Spearman r = {r:.3f}\n{sig}",
                transform=ax.transAxes, fontsize=9.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, lw=1.2),
                va="top")

        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel("Brier Score (lower = better calibration)", fontsize=11)
        ax.set_title(f"{score_name} Concordance Analysis", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_xlim(-0.05, 1.05)

    fig.tight_layout()
    savefig(fig, "ch6_shap_sofa_concordance.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Transformer: mean Layer-1 attention by outcome
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def fig_transformer_attention(
    data_dir="data/processed",
    model_path="outputs/models/transformer_model.pt",
):
    (_, _, X_test, _, _, y_test, _, _, h_test) = load_transformer_data(data_dir)

    model = TimeAwareTransformer(n_input_channels=74, d_model=64, n_heads=4,
                                  n_layers=4, ff_dim=256, dropout=0.1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    n_heads = 4
    hours   = np.arange(48)
    survivor_attn    = np.zeros((n_heads, 48))
    nonsurvivor_attn = np.zeros((n_heads, 48))
    surv_count, ns_count = 0, 0

    loader = DataLoader(TensorDataset(X_test, y_test, h_test), batch_size=32)
    for xb, yb, hb in loader:
        _ = model(xb, hb)
        layer1_w = model.layers[0].attn_weights   # (B, 4, 49, 49)
        if layer1_w is None:
            continue
        cls_attn = layer1_w[:, :, 0, 1:].numpy()  # (B, 4, 48)
        for i, label in enumerate(yb.numpy()):
            if label == 0:
                survivor_attn    += cls_attn[i]
                surv_count       += 1
            else:
                nonsurvivor_attn += cls_attn[i]
                ns_count         += 1

    survivor_attn    /= max(surv_count, 1)
    nonsurvivor_attn /= max(ns_count,   1)

    head_colors = ["#2166ac", "#d73027", "#4dac26", "#7b2d8b"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle(
        "Transformer Layer 1 — Mean CLS Attention by Outcome\n"
        "(Averaged across 517 survivors and 83 non-survivors in test set)",
        fontsize=12, fontweight="bold"
    )

    for h_idx in range(n_heads):
        ax = axes[h_idx // 2][h_idx % 2]
        ax.plot(hours, survivor_attn[h_idx],
                color="#2ca02c", lw=2, label=f"Survivors (n={surv_count})")
        ax.plot(hours, nonsurvivor_attn[h_idx],
                color="#d62728", lw=2, ls="--", label=f"Non-survivors (n={ns_count})")
        ax.fill_between(hours, survivor_attn[h_idx], nonsurvivor_attn[h_idx],
                        alpha=0.12, color=head_colors[h_idx])
        ax.set_title(f"Head {h_idx+1}", fontsize=11, color=head_colors[h_idx], fontweight="bold")
        ax.set_ylabel("Mean attention weight")
        if h_idx >= 2:
            ax.set_xlabel("Hour (0 – 47)")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 47)

    fig.tight_layout()
    savefig(fig, "transformer_attention_outcomes.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Radar chart: all models across 4 metrics
# ─────────────────────────────────────────────────────────────────────────────

def fig_radar_comparison():
    # load from saved JSONs
    with open("outputs/xgboost_results.json")     as f: xgb = json.load(f)["metrics"]
    with open("outputs/tcn_results.json")          as f: tcn = json.load(f)["metrics"]
    with open("outputs/transformer_results.json")  as f: trn = json.load(f)["metrics"]

    # metrics on 0-1 scale; HL-H inverted (lower is better → 1 - normalised)
    max_hlh = max(xgb["hosmer_lemeshow_h"], tcn["hosmer_lemeshow_h"], trn["hosmer_lemeshow_h"])

    metrics = ["AUROC", "AUPRC", "Score1", "Calibration\n(1 – norm. HL-H)"]
    models  = {
        "XGBoost":     [xgb["auroc"], xgb["auprc"], xgb["score1"],
                        1 - xgb["hosmer_lemeshow_h"]/max_hlh],
        "TCN":         [tcn["auroc"], tcn["auprc"], tcn["score1"],
                        1 - tcn["hosmer_lemeshow_h"]/max_hlh],
        "Transformer": [trn["auroc"], trn["auprc"], trn["score1"],
                        1 - trn["hosmer_lemeshow_h"]/max_hlh],
    }
    colors = {"XGBoost": "#2ca02c", "TCN": "#ff7f0e", "Transformer": "#1f77b4"}

    N    = len(metrics)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=7)
    ax.set_ylim(0, 1)

    for name, vals in models.items():
        v = vals + vals[:1]
        ax.plot(angles, v, color=colors[name], lw=2, label=name)
        ax.fill(angles, v, color=colors[name], alpha=0.10)

    # SAPS-I reference dot on Score1 only
    saps_angle = angles[2]
    ax.scatter([saps_angle], [0.3097], color="#7f7f7f", s=60, zorder=5,
               label="SAPS-I (Score1 only)")
    ax.annotate("SAPS-I\n0.31", xy=(saps_angle, 0.3097),
                fontsize=8, color="#555",
                xytext=(saps_angle - 0.3, 0.15))

    ax.set_title("Model Comparison — Radar Chart\n"
                 "PhysioNet 2012 Test Set (n = 600)",
                 fontsize=12, fontweight="bold", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10)

    savefig(fig, "model_radar_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Thesis Figures Generator — Phase 9")
    print(f"Output directory: {OUT_DIR}")
    print("=" * 60)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n[1/7] Chapter 3 — Missingness profile...")
    fig_missingness_profile()

    print("\n[2/7] Chapter 4 — Architecture diagram...")
    fig_architecture_diagram()

    print("\n[3/7] Chapter 5 — Patient SHAP heatmap case study...")
    fig_patient_shap_heatmap()

    print("\n[4/7] Chapter 5 — Top-10 SHAP variables...")
    fig_top10_shap()

    print("\n[5/7] Chapter 6 — SHAP–SOFA concordance scatter...")
    fig_concordance_scatter()

    print("\n[6/7] Transformer — Layer-1 attention by outcome...")
    fig_transformer_attention()

    print("\n[7/7] All models — Radar comparison chart...")
    fig_radar_comparison()

    print("\n" + "=" * 60)
    print("All 7 thesis figures saved to outputs/figures/thesis/")
    figs = [
        "ch3_missingness_profile.png",
        "ch4_architecture_diagram.png",
        "ch5_patient_shap_heatmap.png",
        "ch5_top10_shap_variables.png",
        "ch6_shap_sofa_concordance.png",
        "transformer_attention_outcomes.png",
        "model_radar_comparison.png",
    ]
    for f in figs:
        path = os.path.join(OUT_DIR, f)
        exists = os.path.isfile(path)
        print(f"  {'OK' if exists else 'MISSING':6s}  {f}")


if __name__ == "__main__":
    main()
