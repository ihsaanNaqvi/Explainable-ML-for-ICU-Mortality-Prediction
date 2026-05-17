"""
Clinical validation of SHAP explanations using established severity scores.
Novel MS research contribution: validates that XGBoost learns clinically meaningful patterns.
Compares SHAP feature importance with SOFA and SAPS-I scoring systems.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Import preprocessing utilities
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from load_data import load_outcomes
from preprocess import finalize_features


# Clinical severity scores variable mappings
SOFA_VARIABLES = {
    'GCS': 'neurological',
    'MAP': 'cardiovascular',
    'Creatinine': 'renal',
    'Bilirubin': 'hepatic',
    'PaO2': 'respiratory',
    'Platelets': 'coagulation',
    'MechVent': 'respiratory',
}

SAPS_I_VARIABLES = {
    # Age is intentionally excluded: it is a static general descriptor, not a
    # monitoring variable, so it has no index in VARIABLE_NAMES and cannot be
    # compared against the SHAP matrices which cover only the 37 monitored signals.
    # Concordance is therefore computed over the remaining 10 SAPS-I variables.
    'HR': 'heart_rate',
    'SysABP': 'systolic_bp',
    'Temp': 'temperature',
    'RespRate': 'respiratory_rate',
    'O2Sat': 'oxygen_saturation',
    'Creatinine': 'renal',
    'BUN': 'renal',
    'Bilirubin': 'hepatic',
    'WBC': 'immune',
    'Platelets': 'coagulation',
}

# All variable names matching the MONITORING_VARIABLES in preprocess.py
VARIABLE_NAMES = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Calcium',
    'Chloride', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
    'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent',
    'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'O2Sat', 'PaCO2',
    'PaO2', 'pH', 'Platelets', 'RespRate', 'SaO2', 'SysABP',
    'Temp', 'TropI', 'Urine', 'WBC'
]


def load_outcomes_data(data_dir: str) -> pd.DataFrame:
    """
    Load outcome data including SOFA and SAPS-I scores.
    
    Args:
        data_dir: Path to data directory
        
    Returns:
        DataFrame with RecordID, SOFA, SAPS-I, and other outcomes
    """
    print("Loading outcome data...")
    
    outcomes_df = load_outcomes(data_dir, outcomes_file="Outcomes-a.txt")
    
    print(f"  Loaded outcomes for {len(outcomes_df)} patients")
    print(f"  Columns: {list(outcomes_df.columns)}")
    
    return outcomes_df


def extract_top_variables_per_patient(
    shap_matrices: np.ndarray,
    n_top: int = 5,
) -> np.ndarray:
    """
    Extract top contributing variables for each patient.
    
    Args:
        shap_matrices: (n_samples, 48, 37) time-step SHAP values
        n_top: Number of top variables to extract
        
    Returns:
        (n_samples, n_top) array of variable indices
    """
    n_samples, n_hours, n_variables = shap_matrices.shape
    
    top_var_indices = np.zeros((n_samples, n_top), dtype=int)
    
    for sample_idx in range(n_samples):
        # Sum SHAP across time
        var_shap = np.sum(np.abs(shap_matrices[sample_idx, :, :]), axis=0)
        
        # Get top n variables
        top_idx = np.argsort(var_shap)[-n_top:][::-1]
        top_var_indices[sample_idx, :] = top_idx
    
    return top_var_indices


def compute_concordance_with_sofa(
    top_var_indices: np.ndarray,
    n_top: int = 5,
) -> np.ndarray:
    """
    Compute concordance between SHAP top variables and SOFA variables.
    
    Concordance = (overlap count) / n_top
    
    Args:
        top_var_indices: (n_samples, n_top) variable indices
        n_top: Number of top variables
        
    Returns:
        (n_samples,) concordance scores [0, 1]
    """
    n_samples = top_var_indices.shape[0]
    concordance = np.zeros(n_samples)
    
    # Get SOFA variable indices
    sofa_var_indices = set()
    for var_name in SOFA_VARIABLES.keys():
        if var_name in VARIABLE_NAMES:
            sofa_var_indices.add(VARIABLE_NAMES.index(var_name))
    
    for sample_idx in range(n_samples):
        top_vars = set(top_var_indices[sample_idx, :])
        overlap = len(top_vars & sofa_var_indices)
        concordance[sample_idx] = overlap / n_top
    
    return concordance


def compute_concordance_with_saps_i(
    top_var_indices: np.ndarray,
    n_top: int = 5,
) -> np.ndarray:
    """
    Compute concordance between SHAP top variables and SAPS-I variables.
    
    Args:
        top_var_indices: (n_samples, n_top) variable indices
        n_top: Number of top variables
        
    Returns:
        (n_samples,) concordance scores [0, 1]
    """
    n_samples = top_var_indices.shape[0]
    concordance = np.zeros(n_samples)
    
    # Get SAPS-I variable indices
    saps_var_indices = set()
    for var_name in SAPS_I_VARIABLES.keys():
        if var_name in VARIABLE_NAMES:
            saps_var_indices.add(VARIABLE_NAMES.index(var_name))
    
    for sample_idx in range(n_samples):
        top_vars = set(top_var_indices[sample_idx, :])
        overlap = len(top_vars & saps_var_indices)
        concordance[sample_idx] = overlap / n_top
    
    return concordance


def compute_calibration_metric(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_ece_bins: int = 10,
) -> Tuple[np.ndarray, float, float]:
    """
    Compute per-patient Brier score and Expected Calibration Error (ECE).

    Brier score per patient = (pred_prob - actual)^2, a proper scoring rule.
    ECE = weighted mean |bin_accuracy - bin_confidence| across n_ece_bins bins.

    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test labels
        n_ece_bins: Number of equal-width bins for ECE

    Returns:
        (brier_per_patient, mean_brier, ece)
    """
    X_test_clean = np.nan_to_num(X_test, nan=0.0)
    y_test_clean = y_test[~np.isnan(y_test)]
    X_test_clean = X_test_clean[~np.isnan(y_test)]

    y_pred_proba = model.predict_proba(X_test_clean)[:, 1]

    # Brier score per patient (proper calibration scoring rule)
    brier_per_patient = (y_pred_proba - y_test_clean) ** 2

    # Expected Calibration Error over equal-width bins
    bin_edges = np.linspace(0.0, 1.0, n_ece_bins + 1)
    ece = 0.0
    n = len(y_test_clean)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (y_pred_proba >= lo) & (y_pred_proba < hi)
        if in_bin.sum() == 0:
            continue
        bin_conf = y_pred_proba[in_bin].mean()
        bin_acc = y_test_clean[in_bin].mean()
        ece += (in_bin.sum() / n) * abs(bin_acc - bin_conf)

    return brier_per_patient, float(np.mean(brier_per_patient)), float(ece)


def test_concordance_calibration_hypothesis(
    concordance_sofa: np.ndarray,
    concordance_saps: np.ndarray,
    calibration: np.ndarray,
) -> Dict:
    """
    Test hypothesis: Patients with high concordance have better calibration.
    
    Uses Spearman correlation (non-parametric).
    
    Args:
        concordance_sofa: SOFA concordance scores
        concordance_saps: SAPS-I concordance scores
        calibration: Calibration errors
        
    Returns:
        Dictionary with correlation stats
    """
    results = {}
    
    # SOFA concordance vs calibration
    r_sofa, p_sofa = spearmanr(-concordance_sofa, calibration)  # Negate: higher concordance = better
    results['sofa'] = {
        'spearman_r': float(r_sofa),
        'p_value': float(p_sofa),
        'interpretation': 'Negative correlation means high concordance → better calibration'
    }
    
    # SAPS-I concordance vs calibration
    r_saps, p_saps = spearmanr(-concordance_saps, calibration)
    results['saps_i'] = {
        'spearman_r': float(r_saps),
        'p_value': float(p_saps),
        'interpretation': 'Negative correlation means high concordance → better calibration'
    }
    
    return results


def plot_concordance_distribution(
    concordance_sofa: np.ndarray,
    concordance_saps: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Plot distribution of concordance scores.
    
    Args:
        concordance_sofa: SOFA concordance scores
        concordance_saps: SAPS-I concordance scores
        output_dir: Directory to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # SOFA concordance
    axes[0].hist(concordance_sofa, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(np.mean(concordance_sofa), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(concordance_sofa):.3f}')
    axes[0].set_xlabel('Concordance Score', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Number of Patients', fontsize=11, fontweight='bold')
    axes[0].set_title('SHAP-SOFA Concordance Distribution', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # SAPS-I concordance
    axes[1].hist(concordance_saps, bins=20, color='coral', edgecolor='black', alpha=0.7)
    axes[1].axvline(np.mean(concordance_saps), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(concordance_saps):.3f}')
    axes[1].set_xlabel('Concordance Score', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Number of Patients', fontsize=11, fontweight='bold')
    axes[1].set_title('SHAP-SAPS-I Concordance Distribution', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'concordance_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved concordance distribution plot")


def plot_concordance_vs_calibration(
    concordance_sofa: np.ndarray,
    concordance_saps: np.ndarray,
    calibration: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Plot scatter: concordance vs calibration error.
    
    Args:
        concordance_sofa: SOFA concordance scores
        concordance_saps: SAPS-I concordance scores
        calibration: Calibration errors
        output_dir: Directory to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # SOFA
    axes[0].scatter(concordance_sofa, calibration, alpha=0.5, s=50, color='steelblue', edgecolor='navy')
    z_sofa = np.polyfit(concordance_sofa, calibration, 1)
    p_sofa = np.poly1d(z_sofa)
    x_sofa = np.linspace(concordance_sofa.min(), concordance_sofa.max(), 100)
    axes[0].plot(x_sofa, p_sofa(x_sofa), "r--", linewidth=2, label='Trend')
    axes[0].set_xlabel('SHAP-SOFA Concordance', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Calibration Error', fontsize=11, fontweight='bold')
    axes[0].set_title('SOFA: Concordance vs Calibration', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # SAPS-I
    axes[1].scatter(concordance_saps, calibration, alpha=0.5, s=50, color='coral', edgecolor='darkred')
    z_saps = np.polyfit(concordance_saps, calibration, 1)
    p_saps = np.poly1d(z_saps)
    x_saps = np.linspace(concordance_saps.min(), concordance_saps.max(), 100)
    axes[1].plot(x_saps, p_saps(x_saps), "r--", linewidth=2, label='Trend')
    axes[1].set_xlabel('SHAP-SAPS-I Concordance', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Calibration Error', fontsize=11, fontweight='bold')
    axes[1].set_title('SAPS-I: Concordance vs Calibration', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'concordance_vs_calibration.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved concordance vs calibration plot")


def plot_reliability_diagram(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    n_bins: int = 10,
) -> None:
    """
    Plot reliability diagram (calibration curve) with histogram of predicted probs.

    A perfectly calibrated model lies on the diagonal y=x.

    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test labels
        output_dir: Directory to save figure
        n_bins: Number of equal-width confidence bins
    """
    X_clean = np.nan_to_num(X_test, nan=0.0)
    y_clean = y_test[~np.isnan(y_test)]
    X_clean = X_clean[~np.isnan(y_test)]

    y_pred_proba = model.predict_proba(X_clean)[:, 1]

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_mids = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_acc = np.full(n_bins, np.nan)
    bin_conf = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        in_bin = (y_pred_proba >= lo) & (y_pred_proba < hi)
        bin_counts[i] = in_bin.sum()
        if in_bin.sum() > 0:
            bin_acc[i] = y_clean[in_bin].mean()
            bin_conf[i] = y_pred_proba[in_bin].mean()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9),
                                    gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    valid = ~np.isnan(bin_acc)
    ax1.plot(bin_conf[valid], bin_acc[valid], 'o-', color='steelblue',
             linewidth=2, markersize=8, label='XGBoost')
    ax1.set_xlabel('Mean predicted probability', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Fraction of positives', fontsize=11, fontweight='bold')
    ax1.set_title('Reliability Diagram (Calibration Curve)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.bar(bin_mids, bin_counts, width=1.0 / n_bins, color='steelblue', alpha=0.7, edgecolor='navy')
    ax2.set_xlabel('Predicted probability', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.set_xlim(0, 1)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'reliability_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("  ✓ Saved reliability diagram")


def main(
    data_dir: str = "data",
    output_models_dir: str = None,
) -> Dict:
    """
    Main clinical validation pipeline.
    
    Args:
        data_dir: Path to data directory
        output_models_dir: Directory with trained models
        
    Returns:
        Dictionary with validation results
    """
    root = Path(__file__).resolve().parents[1]
    data_dir = Path(data_dir) if Path(data_dir).is_absolute() else root / data_dir
    
    if output_models_dir is None:
        output_models_dir = root / "outputs" / "models"
    else:
        output_models_dir = Path(output_models_dir)
    
    output_figures_dir = root / "outputs" / "figures" / "validation"
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    
    output_results_dir = root / "outputs"
    output_results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Clinical Validation - SHAP vs Established Severity Scores")
    print("=" * 80)
    
    # Step 1: Load SHAP matrices
    print("\n[1/5] Loading SHAP matrices...")
    shap_matrices = np.load(output_results_dir / "shap_matrices.npy")
    print(f"  SHAP matrices shape: {shap_matrices.shape}")
    
    # Step 2: Load model and test data
    print("\n[2/5] Loading model and test data...")
    model = joblib.load(output_models_dir / "xgboost_model.pkl")
    result = finalize_features(str(data_dir), str(output_models_dir))
    
    X_test_norm = result['X_test_norm']
    X_test_desc = result['X_test_desc']
    y_test = result['y_test']
    
    from models.xgboost_model import prepare_model_features

    processed_dir = data_dir / "processed"
    missingness_mask = np.load(processed_dir / "missingness_mask.npy")

    # Slice mask by split indices so each split gets its own patients' rows
    train_idx = result['train_indices']
    val_idx = result['val_indices']
    test_idx = result['test_indices']

    X_train_norm = result['X_train_norm']
    X_val_norm = result['X_val_norm']
    X_train_desc = result['X_train_desc']
    X_val_desc = result['X_val_desc']

    _, _, X_test_features = prepare_model_features(
        X_train_norm, X_val_norm, X_test_norm,
        X_train_desc, X_val_desc, X_test_desc,
        mask_train=missingness_mask[train_idx],
        mask_val=missingness_mask[val_idx],
        mask_test=missingness_mask[test_idx],
    )

    print(f"  ✓ Loaded test features: {X_test_features.shape}")
    
    # Step 3: Extract top variables and compute concordance
    print("\n[3/5] Computing concordance scores...")
    
    top_var_indices = extract_top_variables_per_patient(shap_matrices, n_top=5)
    print(f"  Extracted top 5 variables per patient")
    
    concordance_sofa = compute_concordance_with_sofa(top_var_indices, n_top=5)
    concordance_saps = compute_concordance_with_saps_i(top_var_indices, n_top=5)
    
    print(f"  Mean SOFA concordance: {np.mean(concordance_sofa):.4f}")
    print(f"  Mean SAPS-I concordance: {np.mean(concordance_saps):.4f}")
    
    # Step 4: Compute calibration metric (Brier score + ECE)
    print("\n[4/5] Computing calibration metrics (Brier score + ECE)...")
    calibration, mean_brier, ece = compute_calibration_metric(model, X_test_features, y_test)
    print(f"  Mean Brier score: {mean_brier:.4f}")
    print(f"  Expected Calibration Error (ECE): {ece:.4f}")
    
    # Step 5: Test hypothesis
    print("\n[5/5] Testing concordance-calibration hypothesis...")
    hypothesis_results = test_concordance_calibration_hypothesis(
        concordance_sofa, concordance_saps, calibration
    )
    
    print(f"\n  SOFA Concordance vs Calibration:")
    print(f"    Spearman r: {hypothesis_results['sofa']['spearman_r']:.4f}")
    print(f"    p-value: {hypothesis_results['sofa']['p_value']:.6f}")
    
    print(f"\n  SAPS-I Concordance vs Calibration:")
    print(f"    Spearman r: {hypothesis_results['saps_i']['spearman_r']:.4f}")
    print(f"    p-value: {hypothesis_results['saps_i']['p_value']:.6f}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_concordance_distribution(concordance_sofa, concordance_saps, output_figures_dir)
    plot_concordance_vs_calibration(concordance_sofa, concordance_saps, calibration, output_figures_dir)
    plot_reliability_diagram(model, X_test_features, y_test, output_figures_dir)
    
    # Save results
    print("\nSaving results...")
    results = {
        'concordance_scores': {
            'sofa': {
                'mean': float(np.mean(concordance_sofa)),
                'std': float(np.std(concordance_sofa)),
                'min': float(np.min(concordance_sofa)),
                'max': float(np.max(concordance_sofa)),
            },
            'saps_i': {
                'mean': float(np.mean(concordance_saps)),
                'std': float(np.std(concordance_saps)),
                'min': float(np.min(concordance_saps)),
                'max': float(np.max(concordance_saps)),
            },
        },
        'calibration': {
            'metric': 'brier_score',
            'mean_brier': float(mean_brier),
            'std_brier': float(np.std(calibration)),
            'ece': float(ece),
        },
        'hypothesis_testing': hypothesis_results,
        'interpretation': {
            'high_concordance_threshold': 0.6,
            'hypothesis': 'Patients with high SHAP-SOFA/SAPS-I concordance have better-calibrated risk estimates',
            'conclusion': 'p-value < 0.05 supports hypothesis (SHAP learns clinically meaningful patterns)',
        }
    }
    
    with open(output_results_dir / "clinical_validation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved results to {output_results_dir / 'clinical_validation_results.json'}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Clinical Validation Complete")
    print("=" * 80)
    print(f"\nMean Concordance Scores:")
    print(f"  SOFA: {np.mean(concordance_sofa):.4f} ± {np.std(concordance_sofa):.4f}")
    print(f"  SAPS-I: {np.mean(concordance_saps):.4f} ± {np.std(concordance_saps):.4f}")
    print(f"\nCalibration:")
    print(f"  Mean Brier score: {mean_brier:.4f} ± {np.std(calibration):.4f}")
    print(f"  ECE: {ece:.4f}")
    print(f"\nSpearman Correlation (Concordance vs Brier score):")
    print(f"  SOFA: r = {hypothesis_results['sofa']['spearman_r']:.4f}, p = {hypothesis_results['sofa']['p_value']:.6f}")
    print(f"  SAPS-I: r = {hypothesis_results['saps_i']['spearman_r']:.4f}, p = {hypothesis_results['saps_i']['p_value']:.6f}")
    print(f"\nFigures saved to: {output_figures_dir}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    main(str(root / "data"))
