"""
Fix Figure 2.1 - Per-Variable Missingness Profile.

Rebuild the missingness heatmap from the raw 3-D tensor with:
  - all 37 variable names visible, no overlap
  - variables sorted by overall observation rate (vitals bottom, labs top)
  - clear green-to-red colormap (green = observed, red = missing)
  - explicit horizontal divider between vital-sign and laboratory groups
  - publication-quality 300-DPI output (PNG + PDF)

Notes
-----
* The variable order is the canonical PhysioNet 2012 order used by
  src/preprocess.py MONITORING_VARIABLES (37 variables, matches the
  tensor's variable axis exactly).
* The on-disk file missingness_mask.npy is all zeros (broken), so the
  observation mask is recomputed here from NaN positions in X_tensor.npy
  (which is saved pre-imputation, hence NaN = missing).

Usage (from d:/icu-xai/):
  python outputs/figures/thesis/fix_fig2_1_missingness.py
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ----------------------------------------------------------------------
# 0. Paths
# ----------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parents[3]      # d:/icu-xai
DATA_DIR = ROOT / "data" / "processed"
OUT_DIR  = ROOT / "outputs" / "figures" / "thesis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Load tensor and derive the observation mask from NaN positions
# ----------------------------------------------------------------------
X = np.load(DATA_DIR / "X_tensor.npy")               # (4000, 48, 37)
print(f"X shape: {X.shape}    X NaN fraction: {np.isnan(X).mean():.4f}")

# Observation mask: 1 = observed, 0 = missing
obs_mask = (~np.isnan(X)).astype(np.float32)         # (4000, 48, 37)

# Canonical variable order (must match src/preprocess.py MONITORING_VARIABLES)
VARS = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Calcium',
    'Chloride', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
    'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent',
    'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'O2Sat', 'PaCO2',
    'PaO2', 'pH', 'Platelets', 'RespRate', 'SaO2', 'SysABP',
    'Temp', 'TropI', 'Urine', 'WBC',
]
assert len(VARS) == X.shape[2] == 37, "Variable count mismatch"

# ----------------------------------------------------------------------
# 2. Per-(variable, hour) missingness percentage
# ----------------------------------------------------------------------
obs_rate    = obs_mask.mean(axis=0)         # (48, 37)
miss_pct    = (1.0 - obs_rate) * 100.0      # (48, 37)
miss_pct_vh = miss_pct.T                    # (37, 48): rows=vars, cols=hours

overall_miss = miss_pct_vh.mean(axis=1)     # (37,)

# ----------------------------------------------------------------------
# 3. Group + within-group sort: labs on top, vitals on bottom
#    Within each group: ascending observation rate (sparsest first/top)
# ----------------------------------------------------------------------
VITAL_SET = {
    'HR', 'MAP', 'RespRate', 'Temp', 'GCS', 'SaO2', 'O2Sat',
    'MechVent', 'NISysABP', 'NIMAP', 'NIDiasABP', 'SysABP', 'DiasABP',
}

lab_idx   = [i for i, v in enumerate(VARS) if v not in VITAL_SET]
vital_idx = [i for i, v in enumerate(VARS) if v in VITAL_SET]

# Sort each group by descending missingness so sparsest sits at the top
lab_idx   = sorted(lab_idx,   key=lambda i: -overall_miss[i])
vital_idx = sorted(vital_idx, key=lambda i: -overall_miss[i])

order_idx       = lab_idx + vital_idx
vars_sorted     = [VARS[i] for i in order_idx]
miss_sorted     = miss_pct_vh[order_idx]
first_vital_row = len(lab_idx)         # divider sits exactly between groups

# ----------------------------------------------------------------------
# 4. Plot
# ----------------------------------------------------------------------
sns.set_style("white")
mpl.rcParams.update({
    "font.family"   : "DejaVu Sans",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})

fig, ax = plt.subplots(figsize=(20, 12), dpi=300)
fig.patch.set_facecolor("white")

hm = sns.heatmap(
    miss_sorted,
    cmap        = "RdYlGn_r",
    vmin        = 0,
    vmax        = 100,
    ax          = ax,
    cbar_kws    = {"label": "Missingness %",
                   "shrink": 0.85,
                   "pad": 0.02},
    linewidths  = 0,
    xticklabels = False,
    yticklabels = vars_sorted,
)

# X-axis: hours 0..47, label every 4 hours
hour_ticks  = np.arange(0, 48, 4) + 0.5
hour_labels = [str(h) for h in range(0, 48, 4)]
ax.set_xticks(hour_ticks)
ax.set_xticklabels(hour_labels, fontsize=11, rotation=0)
ax.set_xlabel("Hour since ICU admission", fontsize=12, labelpad=8)

# Y-axis: all 37 names, horizontal
ax.set_yticks(np.arange(len(vars_sorted)) + 0.5)
ax.set_yticklabels(vars_sorted, fontsize=10, rotation=0)
ax.set_ylabel("Physiological variable", fontsize=12, labelpad=8)

ax.set_title(
    "Figure 2.1 - Per-Variable Missingness Profile Across "
    "48-Hour ICU Window (PhysioNet 2012 Set-A, n = 4,000)",
    fontsize=13, fontweight="bold", pad=15,
)

cbar = hm.collections[0].colorbar
cbar.set_label("Missingness %", fontsize=11)
cbar.ax.tick_params(labelsize=10)

# Horizontal divider between lab tests (top) and vital signs (bottom)
if 0 < first_vital_row < len(vars_sorted):
    ax.axhline(first_vital_row, color="black", linewidth=2.2,
               linestyle="-", alpha=0.9)

# Right-side group annotations (placed outside data area)
n_rows    = len(vars_sorted)
lab_mid   = first_vital_row / 2
vital_mid = first_vital_row + (n_rows - first_vital_row) / 2

ax.text(48.7, lab_mid,
        "Laboratory Tests\n(sparse sampling)",
        fontsize=11, fontweight="bold", color="#7A1B1B",
        ha="left", va="center", rotation=90,
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#FBE9E7",
                  edgecolor="#7A1B1B",
                  linewidth=0.8))
ax.text(48.7, vital_mid,
        "Vital Signs\n(dense sampling)",
        fontsize=11, fontweight="bold", color="#1B5E20",
        ha="left", va="center", rotation=90,
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="#E8F5E9",
                  edgecolor="#1B5E20",
                  linewidth=0.8))

plt.tight_layout()

# ----------------------------------------------------------------------
# 5. Save (PNG + PDF, 300 DPI)
# ----------------------------------------------------------------------
png_path = OUT_DIR / "fig2_1_missingness_FIXED.png"
pdf_path = OUT_DIR / "fig2_1_missingness_FIXED.pdf"
fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
fig.savefig(pdf_path, dpi=300, facecolor="white", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------
# 6. Report
# ----------------------------------------------------------------------
print("\n[ok] Figure saved:")
print(f"     {png_path}")
print(f"     {pdf_path}")

print("\nVariable order (top -> bottom of heatmap):")
for rank, (v, m) in enumerate(zip(vars_sorted, overall_miss[order_idx]), 1):
    group  = "VITAL" if v in VITAL_SET else "lab  "
    marker = "  <-- divider" if rank - 1 == first_vital_row else ""
    print(f"  {rank:2d}. [{group}] {v:<11s}  overall miss = {m:6.2f}%{marker}")

print("\nOverall missingness statistics:")
print(f"  mean    : {overall_miss.mean():6.2f}%")
print(f"  median  : {np.median(overall_miss):6.2f}%")
print(f"  min     : {overall_miss.min():6.2f}%  ({VARS[np.argmin(overall_miss)]})")
print(f"  max     : {overall_miss.max():6.2f}%  ({VARS[np.argmax(overall_miss)]})")
print(f"  vars > 90% miss : {(overall_miss > 90).sum():d} / 37")
print(f"  vars < 20% miss : {(overall_miss < 20).sum():d} / 37")
