"""
Fix distributions_by_outcome.png — Clear separation by outcome with KDE + zoom lactate

Issues in original:
  * Stacked bars hide the died distribution (minority class)
  * Lactate x-axis too wide — outliers stretch the plot
  * Overlapping histograms are hard to distinguish

Fix:
  * KDE curves overlayed on semi-transparent histograms
  * Separate transparency levels for clarity
  * Lactate zoomed to (0, 15) to show main mass
  * Grid lines for readability
  * Clear legend and styling

Data source: X_tensor.npy (4000, 48, 37) → compute mean per patient per variable
Outcome: y_labels.npy (binary: 0=survived, 1=died)

Usage (from d:/icu-xai/):
    python outputs/figures/thesis/fix_distributions.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.stats as stats

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

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading X_tensor.npy and y_labels.npy ...")
X = np.load(ROOT / "data" / "processed" / "X_tensor.npy")   # (4000, 48, 37)
y = np.load(ROOT / "data" / "processed" / "y_labels.npy")   # (4000,)
print(f"  X shape: {X.shape}  y shape: {y.shape}")
print(f"  Outcomes: {np.unique(y)} (0=survived, 1=died)")

# Compute mean across 48 hours for each variable
X_mean = np.nanmean(X, axis=1)  # (4000, 37)
df = pd.DataFrame(X_mean, columns=VARS)
df['In_hospital_death'] = y

print(f"  DataFrame shape: {df.shape}")
print(f"  Outcome counts: Survived={sum(y==0)}, Died={sum(y==1)}")

# ── Global style ──────────────────────────────────────────────────────────
mpl.rcParams.update({
    "font.family"       : "DejaVu Sans",
    "axes.edgecolor"    : "#333333",
    "axes.linewidth"    : 0.8,
    "savefig.facecolor" : "white",
})

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300)
fig.patch.set_facecolor("white")
plt.subplots_adjust(hspace=0.35, wspace=0.3)

survived = df[df['In_hospital_death'] == 0]
died     = df[df['In_hospital_death'] == 1]

# Define 4 key panels: variable, label, axis, x-axis limit (None = auto)
panels = [
    ('HR',      'Heart Rate (bpm)',        axes[0, 0], None),
    ('GCS',     'Glasgow Coma Scale',      axes[0, 1], (0, 15)),
    ('Lactate', 'Lactate (mmol/L)',        axes[1, 0], (0, 10)),
    ('MAP',     'Mean Arterial Pressure (mmHg)', axes[1, 1], None),
]

for col, xlabel, ax, xlim in panels:
    # Extract data
    surv_data = survived[col].dropna().values
    died_data = died[col].dropna().values
    
    print(f"\n{col}:")
    print(f"  Survived: n={len(surv_data)}, mean={np.mean(surv_data):.2f}, std={np.std(surv_data):.2f}")
    print(f"  Died:     n={len(died_data)}, mean={np.mean(died_data):.2f}, std={np.std(died_data):.2f}")
    
    # ── HISTOGRAMS (semi-transparent background) ────────────────────────
    bins_surv = np.histogram_bin_edges(surv_data, bins=25)
    bins_died = np.histogram_bin_edges(died_data, bins=25)
    bins = np.union1d(bins_surv, bins_died)
    
    ax.hist(surv_data, bins=bins, alpha=0.35, color='#2E9E5B', 
            label=f'Survived (n={len(surv_data)})', density=True, edgecolor='none')
    ax.hist(died_data, bins=bins, alpha=0.40, color='#D85A30', 
            label=f'Died (n={len(died_data)})', density=True, edgecolor='none')
    
    # ── KDE CURVES (sharp, opaque) ────────────────────────────────────
    try:
        kde_surv = stats.gaussian_kde(surv_data)
        kde_died = stats.gaussian_kde(died_data)
        
        # Determine x range for KDE
        if xlim:
            x_range = np.linspace(xlim[0], xlim[1], 200)
        else:
            x_min, x_max = min(surv_data.min(), died_data.min()), max(surv_data.max(), died_data.max())
            x_range = np.linspace(x_min, x_max, 200)
        
        ax.plot(x_range, kde_surv(x_range), color='#0D5E2F', linewidth=2.5, 
                label='Survived (KDE)', linestyle='-')
        ax.plot(x_range, kde_died(x_range), color='#A03D1F', linewidth=2.5, 
                label='Died (KDE)', linestyle='-')
    except Exception as e:
        print(f"  Warning: KDE failed for {col}: {e}")
    
    # ── Styling ────────────────────────────────────────────────────────
    ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.set_title(f'{xlabel} by Outcome', fontsize=12, fontweight='bold', pad=10)
    ax.legend(fontsize=9.5, loc='upper right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    
    # ── X-axis limit (if specified) ────────────────────────────────────
    if xlim:
        ax.set_xlim(xlim)
        ax.text(0.98, 0.97, f'x-axis clipped at {xlim[1]} to show main mass',
                ha='right', va='top', fontsize=8.5, color='#666', style='italic',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#f9f9f9', edgecolor='#ddd', linewidth=0.5))

# ── Figure title and layout ────────────────────────────────────────────────
fig.suptitle('Predictor Distributions by Mortality Outcome\nPhysioNet 2012 Set-A (n = 4,000 patients)',
             fontsize=14, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.98])

# ── Save ───────────────────────────────────────────────────────────────────
png_path = OUT_DIR / "distributions_by_outcome_FIXED.png"
pdf_path = OUT_DIR / "distributions_by_outcome_FIXED.pdf"
fig.savefig(png_path, dpi=300, facecolor="white", bbox_inches="tight")
fig.savefig(pdf_path, dpi=300, facecolor="white", bbox_inches="tight")
plt.close(fig)

print(f"\n[ok] {png_path}")
print(f"[ok] {pdf_path}")
