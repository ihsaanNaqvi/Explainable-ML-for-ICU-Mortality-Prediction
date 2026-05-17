"""
Complete preprocessing pipeline:
1. Load raw ICU time-series and outcomes
2. Bin to hourly intervals
3. Forward-fill missing values
4. Create missing indicators
5. Flatten to one row per patient
6. Save model-ready output
"""

import sys
from pathlib import Path

import pandas as pd

from load_data import load_all_patients, load_outcomes, save_processed_data
from preprocess import ICUPreprocessor


def main():
    """Execute the complete preprocessing pipeline."""
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    
    print("=" * 80)
    print("ICU XAI Preprocessing Pipeline")
    print("=" * 80)
    
    # Load raw data
    print("\n[1/3] Loading raw data...")
    raw_df = load_all_patients(str(data_dir), dataset="set-a")
    outcomes_df = load_outcomes(str(data_dir), outcomes_file="Outcomes-a.txt")
    print(f"  Loaded {raw_df['RecordID'].nunique()} patients")
    print(f"  Raw data shape: {raw_df.shape}")
    
    # Preprocess
    print("\n[2/3] Preprocessing to model-ready format...")
    preprocessor = ICUPreprocessor(test_size=0.2, val_size=0.1)
    processed_df = preprocessor.preprocess_end_to_end(
        raw_df,
        outcomes_df,
        window_hours=48,
    )
    print(f"  Final shape: {processed_df.shape}")
    print(f"  Columns: {processed_df.shape[1]}")
    
    # Show sample
    print("\n[3/3] Sample of processed data (first 5 patients):")
    outcome_cols = ['RecordID', 'SAPS-I', 'SOFA', 'In-hospital_death']
    available_outcome_cols = [col for col in outcome_cols if col in processed_df.columns]
    print(processed_df[available_outcome_cols].head())
    
    # Save
    print("\nSaving processed data...")
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = processed_dir / "set_a_model_ready.csv"
    save_processed_data(processed_df, str(output_file))
    print(f"✓ Saved to {output_file}")
    
    # Data summary
    print("\n" + "=" * 80)
    print("Data Summary")
    print("=" * 80)
    print(f"Total patients: {processed_df['RecordID'].nunique()}")
    print(f"Total features (after flattening): {processed_df.shape[1] - 4}")
    print(f"Mortality rate: {processed_df['In-hospital_death'].mean():.4f}")
    print(f"Missing outcomes: {processed_df['In-hospital_death'].isna().sum()}")
    print("=" * 80)


if __name__ == "__main__":
    main()
