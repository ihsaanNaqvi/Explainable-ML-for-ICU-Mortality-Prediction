"""
SHAP-based explainability for XGBoost mortality prediction.
Core contribution: Time-step level SHAP attribution matrices.
Maps feature-level SHAP values back to (time × variable) space.
"""

import json
from pathlib import Path
from typing import Tuple, Dict, List

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Import preprocessing utilities
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))
from preprocess import finalize_features


# Constants matching xgboost_model.py feature structure
N_TEMPORAL_STATS_PER_VAR = 5  # mean, std, min, max, last
N_VARIABLES = 37  # monitoring variables
N_TEMPORAL_FEATURES = N_TEMPORAL_STATS_PER_VAR * N_VARIABLES  # 185
N_MISSINGNESS_FEATURES = N_VARIABLES  # 37
N_DESCRIPTOR_FEATURES = 6  # Age, Gender, Height, Weight, ICUType, + 1
TOTAL_FEATURES = N_TEMPORAL_FEATURES + N_MISSINGNESS_FEATURES + N_DESCRIPTOR_FEATURES  # 228


def compute_shap_values(
    model,
    X_test: np.ndarray,
    use_sampling: bool = False,
    n_samples: int = 100,
) -> np.ndarray:
    """
    Compute SHAP values using TreeExplainer.
    
    Args:
        model: Trained XGBoost model
        X_test: Test features (n_samples, n_features)
        use_sampling: Use sampling for faster computation
        n_samples: Number of samples for background data
        
    Returns:
        SHAP values array (n_samples, n_features)
    """
    print("Computing SHAP values...")
    
    # Handle NaN values
    X_test_clean = np.nan_to_num(X_test, nan=0.0)
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    shap_values = explainer.shap_values(X_test_clean)
    
    # TreeExplainer returns shape (n_samples, n_features) for binary classification
    # If it returns a list [class0, class1], take class 1 (positive)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    print(f"  SHAP values shape: {shap_values.shape}")
    
    return shap_values


def reshape_shap_to_variable_level(
    shap_values: np.ndarray,
) -> np.ndarray:
    """
    Aggregate SHAP values from flattened features to variable level.
    
    Maps 227 features → 37 variables by:
    - Summing 5 temporal statistics per variable (features 0-184)
    - Adding missingness SHAP (features 185-221)
    - Ignoring descriptor SHAP (features 222-226)
    
    Args:
        shap_values: (n_samples, 227) SHAP values from TreeExplainer
        
    Returns:
        (n_samples, 37) aggregated SHAP per variable
    """
    n_samples = shap_values.shape[0]
    var_shap = np.zeros((n_samples, N_VARIABLES))
    
    # Sum temporal statistics (5 per variable)
    for var_idx in range(N_VARIABLES):
        stat_start = var_idx * N_TEMPORAL_STATS_PER_VAR
        stat_end = stat_start + N_TEMPORAL_STATS_PER_VAR
        var_shap[:, var_idx] += np.sum(np.abs(shap_values[:, stat_start:stat_end]), axis=1)
    
    # Add missingness contributions
    miss_start = N_TEMPORAL_FEATURES
    miss_end = miss_start + N_VARIABLES
    var_shap += np.abs(shap_values[:, miss_start:miss_end])
    
    return var_shap


def distribute_shap_across_timesteps(
    X_tensor: np.ndarray,
    var_shap: np.ndarray,
) -> np.ndarray:
    """
    Distribute variable-level SHAP across time steps based on value magnitude.

    For each variable in each patient, the absolute value at each time step
    determines the fraction of SHAP importance attributed to that time step.

    LIMITATION (paper section — methods): This is a post-hoc redistribution
    heuristic, not a mathematically grounded time-step SHAP decomposition.
    The XGBoost model operates on aggregate temporal statistics (mean, std,
    min, max, last), so true per-hour Shapley values cannot be recovered
    without retraining on hourly features. The proportional-magnitude
    weighting assumes that hours with larger observed values contributed more
    to the statistic that drove the prediction. This holds approximately for
    'last value' and 'max/min' statistics but is a weak approximation for
    'mean' (uniform contribution) and 'std' (deviation-dependent). Readers
    should treat the resulting (48 × 37) heatmaps as clinically informative
    summaries rather than exact causal attributions.

    Args:
        X_tensor: (n_samples, 48, 37) normalized features
        var_shap: (n_samples, 37) variable-level SHAP

    Returns:
        (n_samples, 48, 37) time-step level SHAP attribution
    """
    n_samples, n_hours, n_variables = X_tensor.shape
    timestep_shap = np.zeros((n_samples, n_hours, n_variables))
    
    for sample_idx in range(n_samples):
        for var_idx in range(n_variables):
            series = np.abs(X_tensor[sample_idx, :, var_idx])
            series_sum = np.sum(series)
            
            if series_sum > 0:
                # Distribute SHAP proportional to value magnitude
                weights = series / series_sum
                timestep_shap[sample_idx, :, var_idx] = var_shap[sample_idx, var_idx] * weights
            else:
                # Uniform distribution if all values are zero
                timestep_shap[sample_idx, :, var_idx] = var_shap[sample_idx, var_idx] / n_hours
    
    return timestep_shap


def identify_top_features_per_patient(
    timestep_shap: np.ndarray,
    X_tensor: np.ndarray,
    n_top_vars: int = 3,
    n_top_hours: int = 3,
) -> Dict:
    """
    For each patient, identify top variables and critical hours.
    
    Args:
        timestep_shap: (n_samples, 48, 37) time-step SHAP
        X_tensor: (n_samples, 48, 37) feature values
        n_top_vars: Number of top variables
        n_top_hours: Number of critical hours
        
    Returns:
        Dictionary with per-patient top features
    """
    n_samples = timestep_shap.shape[0]
    
    top_features = {}
    
    for sample_idx in range(n_samples):
        # Top variables: sum across time
        var_importance = np.sum(np.abs(timestep_shap[sample_idx, :, :]), axis=0)
        top_var_idx = np.argsort(var_importance)[-n_top_vars:][::-1]
        top_var_shap = var_importance[top_var_idx]
        
        # Top hours: sum across variables
        hour_importance = np.sum(np.abs(timestep_shap[sample_idx, :, :]), axis=1)
        top_hour_idx = np.argsort(hour_importance)[-n_top_hours:][::-1]
        top_hour_shap = hour_importance[top_hour_idx]
        
        top_features[sample_idx] = {
            'top_variables': {
                'indices': top_var_idx.tolist(),
                'shap_values': top_var_shap.tolist(),
            },
            'critical_hours': {
                'indices': top_hour_idx.tolist(),
                'shap_values': top_hour_shap.tolist(),
            },
        }
    
    return top_features


def plot_global_feature_importance(
    var_shap: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Plot mean absolute SHAP values per variable (global importance).
    
    Args:
        var_shap: (n_samples, 37) variable-level SHAP
        output_dir: Directory to save figure
    """
    mean_abs_shap = np.mean(np.abs(var_shap), axis=0)
    
    # Variable names
    variable_names = [
        'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Calcium',
        'Chloride', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
        'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent',
        'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'O2Sat', 'PaCO2',
        'PaO2', 'pH', 'Platelets', 'RespRate', 'SaO2', 'SysABP',
        'Temp', 'TropI', 'Urine', 'WBC'
    ]
    
    # Sort by importance
    sorted_idx = np.argsort(mean_abs_shap)
    sorted_shap = mean_abs_shap[sorted_idx]
    sorted_names = [variable_names[i] for i in sorted_idx]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(range(len(sorted_shap)), sorted_shap, color='steelblue')
    ax.set_yticks(range(len(sorted_shap)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_xlabel('Mean |SHAP value|', fontsize=11, fontweight='bold')
    ax.set_title('Global Feature Importance (Mean Absolute SHAP)', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'global_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved global feature importance plot")


def plot_patient_shap_heatmaps(
    timestep_shap: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    n_patients: int = 5,
) -> None:
    """
    Plot SHAP heatmaps for sample patients (48 hours × 37 variables).
    
    Args:
        timestep_shap: (n_samples, 48, 37) time-step SHAP
        y_test: Test labels
        output_dir: Directory to save figures
        n_patients: Number of patients to visualize
    """
    variable_names = [
        'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Calcium',
        'Chloride', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
        'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent',
        'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'O2Sat', 'PaCO2',
        'PaO2', 'pH', 'Platelets', 'RespRate', 'SaO2', 'SysABP',
        'Temp', 'TropI', 'Urine', 'WBC'
    ]
    
    # Select diverse patients
    np.random.seed(42)
    patient_indices = np.random.choice(len(y_test), min(n_patients, len(y_test)), replace=False)
    
    for patient_id in patient_indices:
        shap_matrix = timestep_shap[patient_id, :, :]
        outcome = "DEATH" if y_test[patient_id] == 1 else "SURVIVAL"
        outcome_color = 'red' if y_test[patient_id] == 1 else 'green'
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create heatmap
        im = ax.imshow(shap_matrix.T, cmap='RdYlBu_r', aspect='auto', origin='lower')
        
        # Labels
        ax.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Variable', fontsize=11, fontweight='bold')
        ax.set_xticks(np.arange(0, 48, 6))
        ax.set_yticks(np.arange(0, 37))
        ax.set_yticklabels(variable_names, fontsize=8)
        
        # Title
        title = f'Patient {patient_id}: Time-Step SHAP Attribution (Outcome: {outcome})'
        ax.set_title(title, fontsize=12, fontweight='bold', color=outcome_color)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('|SHAP value|', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'patient_{patient_id}_shap_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  ✓ Saved {len(patient_indices)} patient-level heatmaps")


def plot_critical_hours_distribution(
    timestep_shap: np.ndarray,
    output_dir: Path,
) -> None:
    """
    Plot distribution of critical hours across all patients.
    
    Args:
        timestep_shap: (n_samples, 48, 37) time-step SHAP
        output_dir: Directory to save figure
    """
    # Sum SHAP across variables for each hour
    hour_importance = np.sum(np.abs(timestep_shap), axis=(0, 2))  # (48,)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(48), hour_importance, color='steelblue', edgecolor='navy', alpha=0.7)
    ax.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total |SHAP value|', fontsize=11, fontweight='bold')
    ax.set_title('Critical Hour Distribution: Which Hours Matter Most?', fontsize=12, fontweight='bold')
    ax.set_xticks(np.arange(0, 48, 6))
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'critical_hours_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved critical hours distribution plot")


def main(
    data_dir: str = "data",
    output_models_dir: str = None,
) -> Dict:
    """
    Main SHAP analysis pipeline.
    
    Args:
        data_dir: Path to data directory
        output_models_dir: Directory with trained models
        
    Returns:
        Dictionary with SHAP analysis results
    """
    root = Path(__file__).resolve().parents[1]
    data_dir = Path(data_dir) if Path(data_dir).is_absolute() else root / data_dir
    
    if output_models_dir is None:
        output_models_dir = root / "outputs" / "models"
    else:
        output_models_dir = Path(output_models_dir)
    
    output_figures_dir = root / "outputs" / "figures" / "shap"
    output_figures_dir.mkdir(parents=True, exist_ok=True)
    
    output_results_dir = root / "outputs"
    output_results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("SHAP Explainability Analysis - Time-Step Attribution")
    print("=" * 80)
    
    # Step 1: Load model and data
    print("\n[1/5] Loading trained model and test data...")
    
    model = joblib.load(output_models_dir / "xgboost_model.pkl")
    print(f"  ✓ Loaded XGBoost model")
    
    result = finalize_features(str(data_dir), str(output_models_dir))
    X_test_norm = result['X_test_norm']
    X_test_desc = result['X_test_desc']
    y_test = result['y_test']

    # Prepare test features (same as in xgboost_model.py)
    from models.xgboost_model import prepare_model_features

    processed_dir = data_dir / "processed"
    missingness_mask = np.load(processed_dir / "missingness_mask.npy")

    # Slice mask by split indices so each split gets its own patients' rows
    train_idx = result['train_indices']
    val_idx = result['val_indices']
    test_idx = result['test_indices']

    # Get all split data to prepare features correctly
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
    
    print(f"  ✓ Loaded test data with {len(y_test)} patients")
    print(f"  ✓ Test features shape: {X_test_features.shape}")
    
    # Step 2: Compute SHAP values
    print("\n[2/5] Computing SHAP values...")
    shap_values = compute_shap_values(model, X_test_features)
    
    # Step 3: Reshape to variable level
    print("\n[3/5] Aggregating SHAP to variable level...")
    var_shap = reshape_shap_to_variable_level(shap_values)
    print(f"  Variable-level SHAP shape: {var_shap.shape}")
    
    # Step 4: Distribute to time steps
    print("\n[4/5] Distributing SHAP across time steps...")
    timestep_shap = distribute_shap_across_timesteps(X_test_norm, var_shap)
    print(f"  Time-step SHAP shape: {timestep_shap.shape}")
    
    # Step 5: Identify top features per patient
    print("\n[5/5] Identifying critical features and hours...")
    top_features = identify_top_features_per_patient(timestep_shap, X_test_norm)
    print(f"  Analyzed {len(top_features)} patients")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_global_feature_importance(var_shap, output_figures_dir)
    plot_patient_shap_heatmaps(timestep_shap, y_test, output_figures_dir, n_patients=5)
    plot_critical_hours_distribution(timestep_shap, output_figures_dir)
    
    # Save SHAP matrices
    print("\nSaving SHAP matrices...")
    np.save(output_results_dir / "shap_matrices.npy", timestep_shap)
    print(f"  ✓ Saved SHAP matrices to {output_results_dir / 'shap_matrices.npy'}")
    
    # Save top features summary as JSON
    top_features_summary = {
        str(k): v for k, v in top_features.items()
    }
    with open(output_results_dir / "top_features_per_patient.json", 'w') as f:
        json.dump(top_features_summary, f, indent=2)
    print(f"  ✓ Saved top features summary")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SHAP Analysis Complete")
    print("=" * 80)
    hour_importance = np.sum(np.abs(timestep_shap), axis=(0, 2))
    top_hour = np.argmax(hour_importance)
    var_importance = np.mean(np.abs(var_shap), axis=0)
    top_var = np.argmax(var_importance)
    
    variable_names = [
        'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Calcium',
        'Chloride', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
        'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent',
        'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'O2Sat', 'PaCO2',
        'PaO2', 'pH', 'Platelets', 'RespRate', 'SaO2', 'SysABP',
        'Temp', 'TropI', 'Urine', 'WBC'
    ]
    
    print(f"Figures saved to: {output_figures_dir}")
    print(f"SHAP matrices saved to: {output_results_dir / 'shap_matrices.npy'}")
    print(f"\nKey Insights:")
    print(f"  Most critical hour: Hour {top_hour}")
    print(f"  Most important variable: {variable_names[top_var]}")
    print(f"  Test set size: {len(y_test)} patients")
    print("=" * 80)
    
    return {
        'timestep_shap': timestep_shap,
        'var_shap': var_shap,
        'shap_values': shap_values,
        'top_features': top_features,
        'y_test': y_test,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    main(str(root / "data"))
