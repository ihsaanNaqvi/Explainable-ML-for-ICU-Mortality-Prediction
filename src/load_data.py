"""
Data loading functions for ICU XAI research
Handles loading PhysioNet set-a records and outcomes labels.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


def _parse_time_to_minutes(time_str: str) -> int:
    """Convert HH:MM time string to minutes since ICU admission."""
    if pd.isna(time_str):
        return np.nan
    hours, minutes = str(time_str).split(":")
    return int(hours) * 60 + int(minutes)


def _extract_record_id(df: pd.DataFrame) -> int:
    """Extract RecordID from a raw patient file."""
    record_id_rows = df.loc[df["Parameter"] == "RecordID", "Value"]
    if record_id_rows.empty:
        raise ValueError("Record file is missing a RecordID row")
    return int(record_id_rows.iloc[0])


def load_patient_file(txt_path: Path) -> pd.DataFrame:
    """Load a single patient record file from the PhysioNet set-a folder."""
    df = pd.read_csv(txt_path)
    if not {"Time", "Parameter", "Value"}.issubset(df.columns):
        raise ValueError(f"Unexpected file format in {txt_path}")

    record_id = _extract_record_id(df)
    df = df.loc[df["Parameter"] != "RecordID", ["Time", "Parameter", "Value"]].copy()
    df["RecordID"] = record_id
    df["Value"] = df["Value"].replace(-1, np.nan)
    df["TimeMinutes"] = df["Time"].astype(str).apply(_parse_time_to_minutes)
    return df


def load_all_patients(data_dir: str, dataset: str = "set-a") -> pd.DataFrame:
    """Load all patient files from the set-a folder."""
    raw_path = Path(data_dir) / "raw" / dataset
    if not raw_path.exists() or not raw_path.is_dir():
        raise FileNotFoundError(f"Raw data folder not found: {raw_path}")

    patient_files = sorted(raw_path.glob("*.txt"))
    if len(patient_files) == 0:
        raise FileNotFoundError(f"No .txt files found in {raw_path}")

    records = [load_patient_file(txt_file) for txt_file in patient_files]
    return pd.concat(records, ignore_index=True)


def load_outcomes(data_dir: str, outcomes_file: str = "Outcomes-a.txt") -> pd.DataFrame:
    """Load outcome labels file and normalize missing values."""
    outcomes_path = Path(data_dir) / "raw" / outcomes_file
    if not outcomes_path.exists():
        raise FileNotFoundError(f"Outcomes file not found: {outcomes_path}")

    outcomes = pd.read_csv(outcomes_path)
    if "In-hospital_death" not in outcomes.columns and "In_hospital_death" in outcomes.columns:
        outcomes = outcomes.rename(columns={"In_hospital_death": "In-hospital_death"})

    # Preserve Survival semantics: -1 means patient survived, not missing.
    mask_cols = [col for col in outcomes.columns if col != "Survival"]
    outcomes[mask_cols] = outcomes[mask_cols].replace(-1, np.nan)
    return outcomes


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """Save a DataFrame to disk under the processed directory."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def summarize_and_save(data_dir: str = "data") -> None:
    """Load raw ICU data, merge outcomes, print summaries, and save processed outputs."""
    root = Path(data_dir)
    print(f"Loading raw patient records from {root / 'raw' / 'set-a'}")
    raw_df = load_all_patients(data_dir, dataset="set-a")
    print(f"Loading outcomes from {root / 'raw' / 'Outcomes-a.txt'}")
    outcomes_df = load_outcomes(data_dir, outcomes_file="Outcomes-a.txt")

    combined = raw_df.merge(outcomes_df, on="RecordID", how="left")

    processed_dir = root / "processed"
    save_processed_data(raw_df, processed_dir / "set_a_raw_longform.csv")
    save_processed_data(outcomes_df, processed_dir / "set_a_outcomes.csv")
    save_processed_data(combined, processed_dir / "set_a_combined.csv")

    total_patients = combined["RecordID"].nunique()
    mortality_rate = combined.drop_duplicates(subset=["RecordID"])["In-hospital_death"].mean()
    sample_id = combined["RecordID"].unique()[0]
    sample_record = combined[combined["RecordID"] == sample_id].head(20)

    print(f"Total patients: {total_patients}")
    print(f"Mortality rate: {mortality_rate:.4f}")
    print(f"Sample patient RecordID: {sample_id}")
    print(sample_record.to_string(index=False))


def main() -> None:
    summarize_and_save(data_dir=Path(__file__).resolve().parents[1] / "data")


if __name__ == "__main__":
    main()
