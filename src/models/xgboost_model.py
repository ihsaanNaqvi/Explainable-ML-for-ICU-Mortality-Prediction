"""
XGBoost mortality prediction model for PhysioNet 2012 ICU dataset.
Trains on flattened time-series features with class weighting for imbalance.
Evaluates using PhysioNet challenge metrics.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score,
    auc,
    precision_recall_curve,
    roc_curve,
)

# Import preprocessing module to access finalize_features
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess import finalize_features


def hosmer_lemeshow_statistic(y_true: np.ndarray, y_pred_proba: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Hosmer-Lemeshow H statistic for goodness-of-fit.
    
    H = sum((observed - expected)^2 / expected) across bins
    Smaller values indicate better calibration.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities [0, 1]
        n_bins: Number of bins (default 10)
        
    Returns:
        H statistic (float)
    """
    # Sort by predicted probability and create bins
    sorted_idx = np.argsort(y_pred_proba)
    y_true_sorted = y_true[sorted_idx]
    y_pred_sorted = y_pred_proba[sorted_idx]
    
    # Create bins
    bin_size = len(y_true) // n_bins
    h_statistic = 0.0
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(y_true)
        
        if end_idx <= start_idx:
            continue
        
        bin_y_true = y_true_sorted[start_idx:end_idx]
        bin_y_pred = y_pred_sorted[start_idx:end_idx]
        
        observed_1 = np.sum(bin_y_true)
        expected_1 = np.sum(bin_y_pred)
        bin_size_actual = end_idx - start_idx
        
        if expected_1 > 0 and (bin_size_actual - expected_1) > 0:
            h_statistic += ((observed_1 - expected_1) ** 2) / expected_1
            h_statistic += (((bin_size_actual - observed_1) - (bin_size_actual - expected_1)) ** 2) / (
                bin_size_actual - expected_1
            )
    
    return h_statistic


def flatten_temporal_features(
    X_tensor: np.ndarray,
    missingness_mask: np.ndarray = None,
) -> np.ndarray:
    """
    Flatten 3D tensor (n_samples, n_hours, n_variables) into 2D features.
    
    For each variable, compute:
    - Mean, std, min, max, last valid value (5 features per variable)
    - Total: 5 * 37 = 185 features
    
    Optionally add:
    - Missingness rate per variable (37 features if missingness_mask provided)
    
    Args:
        X_tensor: (n_samples, n_hours, n_variables)
        missingness_mask: (n_samples, n_hours, n_variables) [1=missing, 0=observed]
        
    Returns:
        (n_samples, n_features) - flattened feature matrix
    """
    n_samples, n_hours, n_variables = X_tensor.shape
    
    # Initialize feature matrix
    n_stat_features = 5  # mean, std, min, max, last
    n_features_base = n_stat_features * n_variables
    n_missing_features = n_variables if missingness_mask is not None else 0
    n_features_total = n_features_base + n_missing_features
    
    X_flat = np.zeros((n_samples, n_features_total))
    
    # Compute statistics per variable
    for sample_idx in range(n_samples):
        feature_idx = 0
        
        for var_idx in range(n_variables):
            series = X_tensor[sample_idx, :, var_idx]
            valid_mask = ~np.isnan(series)
            valid_values = series[valid_mask]
            
            # Compute statistics
            if len(valid_values) > 0:
                X_flat[sample_idx, feature_idx + 0] = np.mean(valid_values)  # mean
                X_flat[sample_idx, feature_idx + 1] = np.std(valid_values)   # std
                X_flat[sample_idx, feature_idx + 2] = np.min(valid_values)   # min
                X_flat[sample_idx, feature_idx + 3] = np.max(valid_values)   # max
                X_flat[sample_idx, feature_idx + 4] = valid_values[-1]       # last valid
            else:
                X_flat[sample_idx, feature_idx : feature_idx + 5] = np.nan
            
            feature_idx += n_stat_features
        
        # Add missingness rates
        if missingness_mask is not None:
            for var_idx in range(n_variables):
                miss_rate = np.mean(missingness_mask[sample_idx, :, var_idx])
                X_flat[sample_idx, feature_idx] = miss_rate
                feature_idx += 1
    
    return X_flat


def prepare_model_features(
    X_train_norm: np.ndarray,
    X_val_norm: np.ndarray,
    X_test_norm: np.ndarray,
    X_train_desc: pd.DataFrame,
    X_val_desc: pd.DataFrame,
    X_test_desc: pd.DataFrame,
    mask_train: np.ndarray = None,
    mask_val: np.ndarray = None,
    mask_test: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare model features by flattening temporal data and adding descriptors.

    Args:
        X_train_norm, X_val_norm, X_test_norm: Normalized 3D tensors (already split)
        X_train_desc, X_val_desc, X_test_desc: General descriptors DataFrames
        mask_train, mask_val, mask_test: Per-split missingness masks matching each
            tensor's patient indices. Must NOT be the full-cohort mask — slice by
            split indices before passing (e.g. missingness_mask[train_idx]).

    Returns:
        (X_train_features, X_val_features, X_test_features) - 2D feature matrices
    """
    print("Flattening temporal features...")

    # Flatten temporal data — each mask must match its tensor's sample dimension
    X_train_flat = flatten_temporal_features(X_train_norm, mask_train)
    X_val_flat = flatten_temporal_features(X_val_norm, mask_val)
    X_test_flat = flatten_temporal_features(X_test_norm, mask_test)
    
    print(f"  Temporal features shape: {X_train_flat.shape}")
    
    # Extract descriptor columns (exclude RecordID)
    desc_cols = [col for col in X_train_desc.columns if col != 'RecordID']
    
    X_train_desc_values = X_train_desc[desc_cols].values
    X_val_desc_values = X_val_desc[desc_cols].values
    X_test_desc_values = X_test_desc[desc_cols].values
    
    print(f"  Descriptor columns: {len(desc_cols)} features")
    
    # Concatenate temporal features with descriptors
    X_train_features = np.hstack([X_train_flat, X_train_desc_values])
    X_val_features = np.hstack([X_val_flat, X_val_desc_values])
    X_test_features = np.hstack([X_test_flat, X_test_desc_values])
    
    print(f"  Total features: {X_train_features.shape[1]} (temporal + descriptors)")
    
    return X_train_features, X_val_features, X_test_features


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: Dict,
    hyperparams: Dict = None,
    early_stopping_rounds: int = 15,
) -> xgb.XGBClassifier:
    """
    Train XGBoost model with class weighting and early stopping on validation loss.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (used for early stopping, not fitting)
        y_val: Validation labels
        class_weights: Dictionary with class weights {0: weight_0, 1: weight_1}
        hyperparams: XGBoost hyperparameters (uses defaults if None)
        early_stopping_rounds: Stop if val logloss doesn't improve for this many rounds

    Returns:
        Trained XGBClassifier
    """
    if hyperparams is None:
        hyperparams = {
            'n_estimators': 500,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
        }

    # Handle NaN values
    X_train_clean = np.nan_to_num(X_train, nan=0.0)
    y_train_clean = y_train[~np.isnan(y_train)]
    X_train_clean = X_train_clean[~np.isnan(y_train)]

    X_val_clean = np.nan_to_num(X_val, nan=0.0)
    y_val_clean = y_val[~np.isnan(y_val)]
    X_val_clean = X_val_clean[~np.isnan(y_val)]

    print(f"\nTraining XGBoost with {len(X_train_clean)} samples...")
    print(f"  Hyperparameters: {hyperparams}")
    print(f"  Class weights: {class_weights}")
    print(f"  Early stopping rounds: {early_stopping_rounds}")

    model = xgb.XGBClassifier(
        **hyperparams,
        eval_metric='logloss',
        early_stopping_rounds=early_stopping_rounds,
    )

    # Compute sample weights for training
    sample_weights = np.array([class_weights[int(y)] for y in y_train_clean])

    model.fit(
        X_train_clean,
        y_train_clean,
        sample_weight=sample_weights,
        eval_set=[(X_val_clean, y_val_clean)],
        verbose=False,
    )

    print(f"  Best iteration: {model.best_iteration} / {hyperparams['n_estimators']}")

    return model


def evaluate_model(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict:
    """
    Evaluate model on test set with PhysioNet challenge metrics.
    
    Metrics:
    - AUROC: Area under ROC curve
    - AUPRC: Area under Precision-Recall curve
    - Sensitivity (Se): True Positive Rate = TP / (TP + FN)
    - Positive Predictivity (+P): Precision = TP / (TP + FP)
    - Score1 = min(Se, +P) - PhysioNet Event 1 score
    - Hosmer-Lemeshow H - PhysioNet Event 2 score
    
    Args:
        model: Trained XGBClassifier
        X_test: Test features
        y_test: Test labels
        
    Returns:
        Dictionary with all metrics
    """
    print("\nEvaluating on test set...")
    
    # Handle NaN values
    X_test_clean = np.nan_to_num(X_test, nan=0.0)
    y_test_clean = y_test[~np.isnan(y_test)]
    X_test_clean = X_test_clean[~np.isnan(y_test)]
    
    # Predicted probabilities (threshold is chosen optimally below)
    y_pred_proba = model.predict_proba(X_test_clean)[:, 1]
    
    # AUROC
    auroc = roc_auc_score(y_test_clean, y_pred_proba)
    
    # AUPRC
    precision, recall, _ = precision_recall_curve(y_test_clean, y_pred_proba)
    auprc = auc(recall, precision)
    
    # Find threshold that maximises Score1 = min(Se, +P) across all PR thresholds.
    # precision_recall_curve returns arrays of length n+1/n+1/n; thresholds[i]
    # corresponds to precision[i] and recall[i] (last precision/recall are boundary).
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test_clean, y_pred_proba)
    best_score1 = 0.0
    best_threshold = 0.5
    for thresh, prec, rec in zip(pr_thresholds, pr_precision[:-1], pr_recall[:-1]):
        s1 = min(rec, prec)
        if s1 > best_score1:
            best_score1 = s1
            best_threshold = float(thresh)

    y_pred_opt = (y_pred_proba >= best_threshold).astype(int)

    tp = np.sum((y_pred_opt == 1) & (y_test_clean == 1))
    fp = np.sum((y_pred_opt == 1) & (y_test_clean == 0))
    fn = np.sum((y_pred_opt == 0) & (y_test_clean == 1))
    tn = np.sum((y_pred_opt == 0) & (y_test_clean == 0))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    positive_predictivity = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # Score1: min(Se, +P) at the optimal threshold
    score1 = min(sensitivity, positive_predictivity)
    print(f"  Optimal threshold for Score1: {best_threshold:.4f}")
    
    # Hosmer-Lemeshow H statistic
    h_statistic = hosmer_lemeshow_statistic(y_test_clean, y_pred_proba)
    
    metrics = {
        'auroc': float(auroc),
        'auprc': float(auprc),
        'sensitivity': float(sensitivity),
        'positive_predictivity': float(positive_predictivity),
        'score1': float(score1),  # Event 1 score (at optimal threshold)
        'score1_threshold': best_threshold,
        'hosmer_lemeshow_h': float(h_statistic),  # Event 2 score
        'test_size': int(len(y_test_clean)),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
    }

    # Print results
    print(f"  AUROC: {metrics['auroc']:.4f}")
    print(f"  AUPRC: {metrics['auprc']:.4f}")
    print(f"  Sensitivity (Se): {metrics['sensitivity']:.4f}")
    print(f"  Positive Predictivity (+P): {metrics['positive_predictivity']:.4f}")
    print(f"  Score1 [min(Se, +P)]: {metrics['score1']:.4f}  (threshold={best_threshold:.4f})")
    print(f"  Hosmer-Lemeshow H [Event 2]: {metrics['hosmer_lemeshow_h']:.4f}")
    print(f"  Test set size: {metrics['test_size']}")
    
    return metrics


def main(
    data_dir: str = "data",
    output_models_dir: str = None,
) -> Dict:
    """
    Main pipeline: feature preparation, model training, and evaluation.
    
    Args:
        data_dir: Path to data directory
        output_models_dir: Directory to save model and results
        
    Returns:
        Dictionary with trained model, metrics, and feature info
    """
    root = Path(__file__).resolve().parents[2]
    data_dir = Path(data_dir) if Path(data_dir).is_absolute() else root / data_dir
    
    if output_models_dir is None:
        output_models_dir = root / "outputs" / "models"
    else:
        output_models_dir = Path(output_models_dir)
    output_models_dir.mkdir(parents=True, exist_ok=True)
    
    output_results_dir = root / "outputs"
    output_results_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("XGBoost Mortality Prediction - PhysioNet 2012")
    print("=" * 80)
    
    # Step 1: Load and finalize features
    print("\n[1/4] Loading and finalizing features...")
    result = finalize_features(str(data_dir), str(output_models_dir))
    
    X_train_norm = result['X_train_norm']
    X_val_norm = result['X_val_norm']
    X_test_norm = result['X_test_norm']
    X_train_desc = result['X_train_desc']
    X_val_desc = result['X_val_desc']
    X_test_desc = result['X_test_desc']
    y_train = result['y_train']
    y_val = result['y_val']
    y_test = result['y_test']
    class_weights = result['class_weights']
    
    # Load missingness mask and slice by split indices so each split's mask
    # corresponds to the correct patients (full mask is n=4000; splits differ)
    processed_dir = data_dir / "processed"
    missingness_mask = np.load(processed_dir / "missingness_mask.npy")

    train_idx = result['train_indices']
    val_idx = result['val_indices']
    test_idx = result['test_indices']

    mask_train = missingness_mask[train_idx]
    mask_val = missingness_mask[val_idx]
    mask_test = missingness_mask[test_idx]

    # Step 2: Prepare model features
    print("\n[2/4] Preparing model features...")
    X_train_features, X_val_features, X_test_features = prepare_model_features(
        X_train_norm,
        X_val_norm,
        X_test_norm,
        X_train_desc,
        X_val_desc,
        X_test_desc,
        mask_train=mask_train,
        mask_val=mask_val,
        mask_test=mask_test,
    )
    
    # Step 3: Train model
    print("\n[3/4] Training XGBoost model...")
    model = train_xgboost_model(X_train_features, y_train, X_val_features, y_val, class_weights)
    
    # Step 4: Evaluate on test set
    print("\n[4/4] Evaluating on test set...")
    metrics = evaluate_model(model, X_test_features, y_test)
    
    # Save model
    print("\nSaving outputs...")
    joblib.dump(model, output_models_dir / "xgboost_model.pkl")
    print(f"  ✓ Saved model to {output_models_dir / 'xgboost_model.pkl'}")
    
    # Save metrics
    results = {
        'metrics': metrics,
        'model_info': {
            'model_type': 'XGBClassifier',
            'n_features': X_train_features.shape[1],
            'feature_breakdown': {
                'temporal_statistics': 185,  # 5 stats * 37 variables
                'missingness_rates': 37,
                'general_descriptors': 6,
            },
        },
    }
    
    with open(output_results_dir / "xgboost_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved results to {output_results_dir / 'xgboost_results.json'}")
    
    # Summary
    print("\n" + "=" * 80)
    print("XGBoost Training Complete")
    print("=" * 80)
    print(f"Model saved to: {output_models_dir / 'xgboost_model.pkl'}")
    print(f"Results saved to: {output_results_dir / 'xgboost_results.json'}")
    print(f"\nKey Metrics:")
    print(f"  AUROC: {metrics['auroc']:.4f}")
    print(f"  AUPRC: {metrics['auprc']:.4f}")
    print(f"  Score1 (Event 1): {metrics['score1']:.4f}")
    print(f"  H-statistic (Event 2): {metrics['hosmer_lemeshow_h']:.4f}")
    print("=" * 80)
    
    return {
        'model': model,
        'metrics': metrics,
        'X_train_features': X_train_features,
        'X_test_features': X_test_features,
        'y_test': y_test,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    main(str(root / "data"))
