"""
Fix correlation_heatmap.png — Full symmetric matrix with all annotations.

Issues in original:
  * Upper triangle masked out (half the information missing)
  * No numeric annotations on most cells

Fix:
  * Full symmetric matrix (no mask)
  * annot=True on all cells, fmt='.2f'
  * Footnote: GCS near-zero Pearson r != low SHAP importance (non-linearity)

Data source: X_tensor.npy (4000, 48, 37) reshaped to (192000, 37).
Correlations are Pearson on pairwise-complete observations (NaN = missing
measurements are dropped per pair, matching pandas default behaviour).

Usage (from d:/icu-xai/):
    python outputs/figures/thesis/fix_correlation_heatmap.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT    = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "outputs" / "figures" / "thesis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Variable names (canonical order matching X_tensor axis-2) ────────────
VARS = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Calcium',
    'Chloride', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
    'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent',
    'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'O2Sat', 'PaCO2',
    'PaO2', 'pH', 'Platelets', 'RespRate', 'SaO2', 'SysABP',
    'Temp', 'TropI', 'Urine', 'WBC',
]

# ── Load tensor and build flat DataFrame ─────────────────────────────────
print("Loading X_tensor.npy ...")
X = np.load(ROOT / "data" / "processed" / "X_tensor.npy")   # (4000, 48, 37)
print(f"  shape: {X.shape}  NaN fraction: {np.isnan(X).mean():.4f}")

# Flatten to (192000, 37) — each row is one (patient, hour) observation
X_flat = X.reshape(-1, 37)
df = pd.DataFrame(X_flat, columns=VARS)
print(f"  flat shape: {df.shape}  rows with any value: {df.notna().any(axis=1).sum():,}")

# ── Select variables and compute correlation ──────────────────────────────
vars_sel = ['HR', 'MAP', 'GCS', 'Lactate', 'Creatinine', 'BUN', 'Temp']
corr = df[vars_sel].corr().round(2)

print("\nCorrelation matrix:")
print(corr.to_string())

# ── Global style ─────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "axes.edgecolor"    : "#333333",
    "axes.linewidth"    : 0.8,
    "savefig.facecolor" : "white",
})

# ── Plot ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 11), dpi=300)  # ← LARGER to prevent overlap
fig.patch.set_facecolor("white")

# Create heatmap WITHOUT annotations first (we'll add them manually)
sns.heatmap(
    corr,
    annot      = False,          # ← We will add annotations manually for guarantee
    cmap       = "RdBu_r",
    center     = 0,
    vmin       = -1,
    vmax       = 1,
    square     = True,
    linewidths = 1.2,
    linecolor  = "white",
    cbar_kws   = {"label": "Pearson r", "shrink": 0.7},
    ax         = ax,
)

# ── MANUALLY ADD TEXT ANNOTATIONS TO ALL 49 CELLS ──────────────────────────
# This ensures every cell (upper AND lower triangle) gets a label
for i in range(len(vars_sel)):
    for j in range(len(vars_sel)):
        value = corr.iloc[i, j]
        text_color = "white" if abs(value) >= 0.5 else "black"  # white text on dark, black on light
        ax.text(
            j + 0.5, i + 0.5,
            f"{value:.2f}",
            ha="center", va="center",
            fontsize=12, fontweight="bold",  # ← reduced from 13 to fit better
            color=text_color,
        )

ax.set_title(
    "Correlation Matrix — Key Clinical Variables (FULL 7×7 SYMMETRIC)\n"
    "PhysioNet 2012 Set-A  |  n = 4,000 patients, 192,000 hourly observations",
    fontsize=14, fontweight="bold", pad=20,
)

# ← More padding for axis labels to prevent overlap
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

ax.text(
    0.5, -0.15,
    "✓ Full symmetric matrix with all 49 cell values annotated  |  "
    "✓ Creatinine–BUN = 0.69 (strong positive, kidney function)  |  "
    "✓ GCS near-zero correlation but #1 SHAP predictor (non-linear)",
    fontsize=10, color="#333333", style="italic",
    transform=ax.transAxes,
    ha="center",
    bbox=dict(boxstyle="round,pad=0.7", facecolor="#fffacd", edgecolor="#bbb", linewidth=0.8),
)

plt.tight_layout(rect=[0, 0.08, 1, 0.96])  # ← Give space for title and footer

plt.tight_layout(rect=[0, 0.04, 1, 1])

# ── Save ─────────────────────────────────────────────────────────────────
png_path = OUT_DIR / "correlation_heatmap_FIXED.png"
pdf_path = OUT_DIR / "correlation_heatmap_FIXED.pdf"
fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
fig.savefig(pdf_path, dpi=300, facecolor="white", bbox_inches="tight")
plt.close(fig)

print(f"\n[ok] {png_path}")
print(f"[ok] {pdf_path}")
