"""
Preprocessing for PhysioNet 2012 ICU dataset.
Builds 3D tensor: (4000 patients, 48 hours, 37 variables)
Includes missingness tracking, forward-filling, and feature normalization.
"""

import pickle
from typing import Tuple, Dict

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from load_data import load_all_patients, load_outcomes


# General patient descriptors (static, extracted once per patient)
GENERAL_DESCRIPTORS = ['Age', 'Gender', 'Height', 'Weight', 'ICUType']

# All monitoring variables in PhysioNet 2012
MONITORING_VARIABLES = [
    'Albumin', 'ALP', 'ALT', 'AST', 'Bilirubin', 'BUN', 'Calcium',
    'Chloride', 'Creatinine', 'DiasABP', 'FiO2', 'GCS', 'Glucose',
    'HCO3', 'HCT', 'HR', 'K', 'Lactate', 'Mg', 'MAP', 'MechVent',
    'Na', 'NIDiasABP', 'NIMAP', 'NISysABP', 'O2Sat', 'PaCO2',
    'PaO2', 'pH', 'Platelets', 'RespRate', 'SaO2', 'SysABP',
    'Temp', 'TropI', 'Urine', 'WBC'
]


def bin_to_hourly_grid(
    raw_df: pd.DataFrame,
    window_hours: int = 48,
) -> pd.DataFrame:
    """
    Bin irregular time-series into hourly buckets.
    
    Args:
        raw_df: Raw long-form data with TimeMinutes column
        window_hours: Number of hours to include (0-47)
        
    Returns:
        DataFrame with [RecordID, Hour, Parameter, Value]
    """
    df = raw_df.copy()
    
    # Filter to observation window
    df = df[df['TimeMinutes'] <= window_hours * 60].copy()
    
    # Bin into hourly buckets
    df['Hour'] = (df['TimeMinutes'] // 60).astype(int)
    
    # Aggregate multiple measurements per hour per patient per parameter
    binned = df.groupby(['RecordID', 'Hour', 'Parameter'])['Value'].mean().reset_index()
    
    return binned


def create_3d_tensor(
    binned_df: pd.DataFrame,
    variables: list,
    window_hours: int = 48,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create 3D tensor and tracking matrices for patients × hours × variables.
    
    Args:
        binned_df: Binned data with [RecordID, Hour, Parameter, Value]
        variables: List of variables to include
        window_hours: Number of hours (0-47)
        
    Returns:
        (X_tensor, missingness_mask, record_ids)
        - X_tensor: (n_patients, window_hours, n_variables)
        - missingness_mask: (n_patients, window_hours, n_variables) [1=missing, 0=observed]
        - record_ids: array of sorted RecordIDs
    """
    record_ids = sorted(binned_df['RecordID'].unique())
    n_patients = len(record_ids)
    n_hours = window_hours
    n_vars = len(variables)
    
    X_tensor = np.full((n_patients, n_hours, n_vars), np.nan, dtype=np.float32)
    missingness_mask = np.zeros((n_patients, n_hours, n_vars), dtype=np.uint8)
    
    # Map variable names to indices
    var_to_idx = {var: i for i, var in enumerate(variables)}
    record_to_patient_idx = {rid: i for i, rid in enumerate(record_ids)}
    
    # Populate tensor
    for _, row in binned_df.iterrows():
        record_id = int(row['RecordID'])
        hour = int(row['Hour'])
        param = row['Parameter']
        value = row['Value']
        
        if param not in var_to_idx or hour >= n_hours:
            continue
        
        patient_idx = record_to_patient_idx[record_id]
        var_idx = var_to_idx[param]
        
        if pd.notna(value):
            X_tensor[patient_idx, hour, var_idx] = value
            missingness_mask[patient_idx, hour, var_idx] = 0
        else:
            missingness_mask[patient_idx, hour, var_idx] = 1
    
    return X_tensor, missingness_mask, np.array(record_ids)


def forward_fill_tensor(
    X_tensor: np.ndarray,
) -> np.ndarray:
    """
    Forward-fill missing values within each patient's trajectory (across hours).
    Operates per variable: fills forward in time for each patient-variable pair.
    
    Args:
        X_tensor: (n_patients, n_hours, n_variables)
        
    Returns:
        Forward-filled tensor
    """
    X_filled = X_tensor.copy()
    
    for patient_idx in range(X_tensor.shape[0]):
        for var_idx in range(X_tensor.shape[2]):
            series = X_tensor[patient_idx, :, var_idx]
            
            # Forward-fill: propagate last valid value forward
            valid_idx = np.where(~np.isnan(series))[0]
            if len(valid_idx) == 0:
                continue
            
            # Fill from first valid value forward
            first_valid = valid_idx[0]
            last_valid = first_valid
            
            for hour_idx in range(first_valid, len(series)):
                if not np.isnan(series[hour_idx]):
                    last_valid = hour_idx
                else:
                    X_filled[patient_idx, hour_idx, var_idx] = series[last_valid]
    
    return X_filled


def extract_general_descriptors(
    raw_df: pd.DataFrame,
    record_ids: np.ndarray,
) -> pd.DataFrame:
    """
    Extract static patient descriptors (one row per patient).
    
    Args:
        raw_df: Raw long-form data
        record_ids: Sorted array of RecordIDs
        
    Returns:
        DataFrame with [RecordID, Age, Gender, Height, Weight, ICUType]
    """
    descriptors_data = []
    
    for record_id in record_ids:
        patient_data = raw_df[raw_df['RecordID'] == record_id]
        row = {'RecordID': int(record_id)}
        
        for desc in GENERAL_DESCRIPTORS:
            values = patient_data[patient_data['Parameter'] == desc]['Value']
            # Get first (should be same for static descriptors)
            row[desc] = values.iloc[0] if len(values) > 0 else np.nan
        
        descriptors_data.append(row)
    
    return pd.DataFrame(descriptors_data)


def preprocess_end_to_end(
    data_dir: str = "data",
    window_hours: int = 48,
    save_dir: str = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    """
    Complete preprocessing pipeline.
    
    Args:
        data_dir: Path to data directory
        window_hours: Hours of observation (0-47 = 48 hours total)
        save_dir: Directory to save outputs (defaults to data/processed)
        
    Returns:
        (X_tensor, missingness_mask, record_ids, general_desc_df, y_labels)
    """
    data_dir = Path(data_dir)
    if save_dir is None:
        save_dir = data_dir / "processed"
    else:
        save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PhysioNet 2012 Preprocessing Pipeline")
    print("=" * 80)
    
    # Load raw data
    print("\n[1/5] Loading raw ICU records...")
    raw_df = load_all_patients(str(data_dir), dataset="set-a")
    outcomes_df = load_outcomes(str(data_dir), outcomes_file="Outcomes-a.txt")
    print(f"  Loaded {raw_df['RecordID'].nunique()} patients")
    print(f"  Total raw records: {len(raw_df)}")
    
    # Bin to hourly grid
    print("\n[2/5] Binning to hourly intervals...")
    binned_df = bin_to_hourly_grid(raw_df, window_hours=window_hours)
    print(f"  Binned records: {len(binned_df)}")
    
    # Create 3D tensor
    print("\n[3/5] Creating 3D tensor (patients × hours × variables)...")
    X_tensor, missingness_mask, record_ids = create_3d_tensor(
        binned_df,
        MONITORING_VARIABLES,
        window_hours=window_hours,
    )
    print(f"  X_tensor shape: {X_tensor.shape}")
    print(f"  missingness_mask shape: {missingness_mask.shape}")
    
    # Calculate missing rate before forward-fill
    missing_rate_before = np.isnan(X_tensor).sum() / X_tensor.size
    print(f"  Missing rate (before forward-fill): {missing_rate_before:.4f}")
    
    # Forward-fill
    print("\n[4/5] Forward-filling missing values...")
    X_tensor = forward_fill_tensor(X_tensor)
    missing_rate_after = np.isnan(X_tensor).sum() / X_tensor.size
    print(f"  Missing rate (after forward-fill): {missing_rate_after:.4f}")
    
    # Extract general descriptors
    print("\n[5/5] Extracting general descriptors...")
    general_desc_df = extract_general_descriptors(raw_df, record_ids)
    print(f"  General descriptors shape: {general_desc_df.shape}")
    
    # Prepare outcomes
    y_labels = []
    for rid in record_ids:
        outcome = outcomes_df[outcomes_df['RecordID'] == rid]['In-hospital_death'].values
        y_labels.append(outcome[0] if len(outcome) > 0 else np.nan)
    y_labels = np.array(y_labels, dtype=np.float32)
    
    # Save outputs
    print("\nSaving outputs...")
    np.save(save_dir / "X_tensor.npy", X_tensor)
    print(f"  ✓ Saved X_tensor.npy {X_tensor.shape}")
    
    np.save(save_dir / "missingness_mask.npy", missingness_mask)
    print(f"  ✓ Saved missingness_mask.npy {missingness_mask.shape}")
    
    np.save(save_dir / "y_labels.npy", y_labels)
    print(f"  ✓ Saved y_labels.npy {y_labels.shape}")
    
    general_desc_df.to_csv(save_dir / "general_descriptors.csv", index=False)
    print(f"  ✓ Saved general_descriptors.csv {general_desc_df.shape}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Preprocessing Complete")
    print("=" * 80)
    print(f"X_tensor shape: {X_tensor.shape} (patients × hours × variables)")
    print(f"Mortality rate: {np.nanmean(y_labels):.4f}")
    print(f"Missing rate (after forward-fill): {missing_rate_after:.4f}")
    print(f"Saved to: {save_dir}")
    print("=" * 80)
    
    return X_tensor, missingness_mask, record_ids, general_desc_df, y_labels


def main():
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    preprocess_end_to_end(str(data_dir), window_hours=48)


def finalize_features(
    data_dir: str = "data",
    output_models_dir: str = None,
    val_split: float = 0.15,
    test_split: float = 0.15,
    random_state: int = 42,
) -> Dict:
    """
    Normalize features and prepare train/val/test splits with class weights.
    
    Args:
        data_dir: Path to data directory containing preprocessed outputs
        output_models_dir: Directory to save scaler objects
        val_split: Fraction for validation set
        test_split: Fraction for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary with:
        - X_train_norm, X_val_norm, X_test_norm (normalized tensors)
        - X_train_desc, X_val_desc, X_test_desc (normalized descriptors)
        - y_train, y_val, y_test (labels)
        - class_weights (dict: {0: weight_0, 1: weight_1})
        - train_indices, val_indices, test_indices
    """
    data_dir = Path(data_dir)
    processed_dir = data_dir / "processed"
    
    if output_models_dir is None:
        output_models_dir = Path(__file__).resolve().parents[1] / "outputs" / "models"
    else:
        output_models_dir = Path(output_models_dir)
    output_models_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Feature Finalization & Split Preparation")
    print("=" * 80)
    
    # Load preprocessed data
    print("\n[1/5] Loading preprocessed data...")
    X_tensor = np.load(processed_dir / "X_tensor.npy")
    y_labels = np.load(processed_dir / "y_labels.npy")
    general_desc_df = pd.read_csv(processed_dir / "general_descriptors.csv")
    
    print(f"  X_tensor shape: {X_tensor.shape}")
    print(f"  y_labels shape: {y_labels.shape}")
    print(f"  General descriptors shape: {general_desc_df.shape}")
    
    n_samples = X_tensor.shape[0]
    
    # Create train/val/test split (stratified)
    print("\n[2/5] Creating stratified train/val/test split (70/15/15)...")
    
    # First split: train (70%) vs temp (30%)
    train_idx, temp_idx = train_test_split(
        np.arange(n_samples),
        test_size=(val_split + test_split),
        stratify=y_labels,
        random_state=random_state,
    )
    
    # Second split: val (15%) vs test (15%)
    val_ratio = val_split / (val_split + test_split)
    y_temp = y_labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=(1 - val_ratio),
        stratify=y_temp,
        random_state=random_state,
    )
    
    train_size = len(train_idx)
    val_size = len(val_idx)
    test_size = len(test_idx)
    
    print(f"  Train size: {train_size} ({train_size/n_samples*100:.1f}%)")
    print(f"  Val size: {val_size} ({val_size/n_samples*100:.1f}%)")
    print(f"  Test size: {test_size} ({test_size/n_samples*100:.1f}%)")
    
    # Extract splits
    X_train = X_tensor[train_idx]
    X_val = X_tensor[val_idx]
    X_test = X_tensor[test_idx]
    
    y_train = y_labels[train_idx]
    y_val = y_labels[val_idx]
    y_test = y_labels[test_idx]
    
    desc_train = general_desc_df.iloc[train_idx]
    desc_val = general_desc_df.iloc[val_idx]
    desc_test = general_desc_df.iloc[test_idx]
    
    # Normalize time-series (3D tensor)
    print("\n[3/5] Normalizing time-series variables...")
    scaler_timeseries = StandardScaler()
    
    # Reshape tensor for scaler: (samples * hours, variables)
    X_train_reshaped = X_train.reshape(-1, X_train.shape[2])
    scaler_timeseries.fit(X_train_reshaped)
    
    # Apply to all splits
    X_train_norm = X_train.copy()
    X_val_norm = X_val.copy()
    X_test_norm = X_test.copy()
    
    for hour in range(X_train.shape[1]):
        X_train_norm[:, hour, :] = scaler_timeseries.transform(X_train[:, hour, :])
        X_val_norm[:, hour, :] = scaler_timeseries.transform(X_val[:, hour, :])
        X_test_norm[:, hour, :] = scaler_timeseries.transform(X_test[:, hour, :])
    
    print(f"  X_train_norm shape: {X_train_norm.shape}")
    print(f"  X_val_norm shape: {X_val_norm.shape}")
    print(f"  X_test_norm shape: {X_test_norm.shape}")
    
    # Normalize general descriptors
    print("\n[4/5] Normalizing general descriptors...")
    scaler_descriptors = StandardScaler()
    
    desc_cols = [col for col in desc_train.columns if col != 'RecordID']
    scaler_descriptors.fit(desc_train[desc_cols].values)
    
    desc_train_norm = desc_train.copy()
    desc_val_norm = desc_val.copy()
    desc_test_norm = desc_test.copy()
    
    desc_train_norm[desc_cols] = scaler_descriptors.transform(desc_train[desc_cols].values)
    desc_val_norm[desc_cols] = scaler_descriptors.transform(desc_val[desc_cols].values)
    desc_test_norm[desc_cols] = scaler_descriptors.transform(desc_test[desc_cols].values)
    
    print(f"  Descriptor columns: {desc_cols}")
    
    # Compute class weights
    print("\n[5/5] Computing class weights...")
    unique, counts = np.unique(y_train[~np.isnan(y_train)], return_counts=True)
    class_weights = {}
    
    # Weight inversely proportional to class frequency
    total = counts.sum()
    for cls, count in zip(unique, counts):
        class_weights[int(cls)] = total / (len(unique) * count)
    
    # Normalize so average weight is 1
    avg_weight = np.mean(list(class_weights.values()))
    class_weights = {k: v / avg_weight for k, v in class_weights.items()}
    
    weight_ratio = class_weights[1] / class_weights[0] if 0 in class_weights else 1.0
    print(f"  Class weights: {class_weights}")
    print(f"  Class weight ratio (minority/majority): {weight_ratio:.4f}")
    
    # Save split indices
    print("\nSaving outputs...")
    np.savez(
        processed_dir / "splits.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    print(f"  ✓ Saved splits.npz")
    
    # Save scalers
    joblib.dump(scaler_timeseries, output_models_dir / "scaler_timeseries.pkl")
    joblib.dump(scaler_descriptors, output_models_dir / "scaler_descriptors.pkl")
    print(f"  ✓ Saved scalers to {output_models_dir}")
    
    # Save normalized data
    np.save(processed_dir / "X_train_norm.npy", X_train_norm)
    np.save(processed_dir / "X_val_norm.npy", X_val_norm)
    np.save(processed_dir / "X_test_norm.npy", X_test_norm)
    print(f"  ✓ Saved normalized tensors")
    
    desc_train_norm.to_csv(processed_dir / "desc_train_norm.csv", index=False)
    desc_val_norm.to_csv(processed_dir / "desc_val_norm.csv", index=False)
    desc_test_norm.to_csv(processed_dir / "desc_test_norm.csv", index=False)
    print(f"  ✓ Saved normalized descriptors")
    
    print("\n" + "=" * 80)
    print("Feature Finalization Complete")
    print("=" * 80)
    print(f"Train size: {train_size} | Val size: {val_size} | Test size: {test_size}")
    print(f"Class weight ratio (death/survivor): {weight_ratio:.4f}")
    print("=" * 80)
    
    return {
        'X_train_norm': X_train_norm,
        'X_val_norm': X_val_norm,
        'X_test_norm': X_test_norm,
        'X_train_desc': desc_train_norm,
        'X_val_desc': desc_val_norm,
        'X_test_desc': desc_test_norm,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'class_weights': class_weights,
        'train_indices': train_idx,
        'val_indices': val_idx,
        'test_indices': test_idx,
        'scaler_timeseries': scaler_timeseries,
        'scaler_descriptors': scaler_descriptors,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    
    # Step 1: Preprocess raw data
    print("\n" + "█" * 80)
    print("STEP 1: DATA PREPROCESSING")
    print("█" * 80)
    preprocess_end_to_end(str(data_dir), window_hours=48)
    
    # Step 2: Finalize features and create splits
    print("\n" + "█" * 80)
    print("STEP 2: FEATURE FINALIZATION & SPLITTING")
    print("█" * 80)
    finalize_features(str(data_dir))
