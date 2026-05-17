"""
Exploratory Data Analysis (EDA) for ICU XAI dataset.
Generates visualizations for missingness, class imbalance, distributions, and correlations.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from load_data import load_all_patients, load_outcomes


def plot_missingness_heatmap(raw_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create heatmap of % missing values for all variables.
    
    Args:
        raw_df: Raw long-form data
        output_dir: Directory to save figures
    """
    missing_pct = (
        raw_df.groupby('Parameter')['Value']
        .apply(lambda x: (x.isna().sum() / len(x)) * 100)
        .sort_values(ascending=False)
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    missing_pct.plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('% Missing', fontsize=12)
    ax.set_ylabel('Variable', fontsize=12)
    ax.set_title('Missingness Heatmap: % Missing Values per Variable', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'missingness_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved missingness heatmap to {output_dir / 'missingness_heatmap.png'}")
    plt.close()


def plot_class_imbalance(outcomes_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Bar chart of survivors vs deaths (class imbalance).
    
    Args:
        outcomes_df: Outcomes data with In-hospital_death label
        output_dir: Directory to save figures
    """
    death_counts = outcomes_df['In-hospital_death'].value_counts().sort_index()
    labels = ['Survived', 'Died']
    colors = ['#2ecc71', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, death_counts.values, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add count labels on bars
    for bar, count in zip(bars, death_counts.values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}\n({count/death_counts.sum()*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Patient Count', fontsize=12)
    ax.set_title('Class Imbalance: Mortality Outcome Distribution', fontsize=14, fontweight='bold')
    ax.set_ylim(0, death_counts.max() * 1.15)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'class_imbalance.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved class imbalance chart to {output_dir / 'class_imbalance.png'}")
    plt.close()


def plot_distributions_by_outcome(combined_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Distributions of key variables (Age, GCS, Lactate, MAP) by mortality outcome.
    
    Args:
        combined_df: Combined raw + outcomes data (long form)
        output_dir: Directory to save figures
    """
    key_vars = ['Age', 'GCS', 'Lactate', 'NIMAP']
    var_display = {'NIMAP': 'Mean Arterial Pressure', 'Age': 'Age', 'GCS': 'Glasgow Coma Scale', 'Lactate': 'Lactate'}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, var in enumerate(key_vars):
        ax = axes[idx]
        
        # Filter long-form data for this variable
        var_data = combined_df[combined_df['Parameter'] == var][['RecordID', 'In-hospital_death', 'Value']].copy()
        var_data = var_data.dropna(subset=['Value'])
        
        # Get one value per patient (first occurrence)
        var_data = var_data.drop_duplicates(subset=['RecordID'], keep='first')
        
        alive = var_data[var_data['In-hospital_death'] == 0]['Value'].dropna()
        dead = var_data[var_data['In-hospital_death'] == 1]['Value'].dropna()
        
        display_name = var_display.get(var, var)
        
        ax.hist(alive, bins=30, alpha=0.6, label='Survived', color='#2ecc71', edgecolor='black')
        ax.hist(dead, bins=30, alpha=0.6, label='Died', color='#e74c3c', edgecolor='black')
        
        ax.set_xlabel(display_name, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{display_name} Distribution by Outcome', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'distributions_by_outcome.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved distributions chart to {output_dir / 'distributions_by_outcome.png'}")
    plt.close()


def plot_missingness_per_patient(raw_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Histogram of % missing values per patient.
    
    Args:
        raw_df: Raw long-form data
        output_dir: Directory to save figures
    """
    missingness_per_patient = (
        raw_df.groupby('RecordID')['Value']
        .apply(lambda x: (x.isna().sum() / len(x)) * 100)
    )
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(missingness_per_patient, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    
    ax.axvline(missingness_per_patient.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {missingness_per_patient.mean():.1f}%')
    ax.axvline(missingness_per_patient.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {missingness_per_patient.median():.1f}%')
    
    ax.set_xlabel('% Missing Values per Patient', fontsize=12)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_title('Distribution of Missingness per Patient', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'missingness_per_patient.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved per-patient missingness histogram to {output_dir / 'missingness_per_patient.png'}")
    plt.close()


def plot_correlation_heatmap(combined_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Correlation heatmap of key clinical variables.
    
    Args:
        combined_df: Combined raw + outcomes data (long form)
        output_dir: Directory to save figures
    """
    key_vars = ['HR', 'NIMAP', 'GCS', 'Lactate', 'Creatinine', 'BUN', 'Temp']
    var_display = {
        'HR': 'Heart Rate',
        'NIMAP': 'Mean Arterial Pressure',
        'GCS': 'Glasgow Coma Scale',
        'Lactate': 'Lactate',
        'Creatinine': 'Creatinine',
        'BUN': 'Blood Urea Nitrogen',
        'Temp': 'Temperature',
    }
    
    # Extract one value per patient per variable from long-form data
    patient_data = []
    for record_id in combined_df['RecordID'].unique():
        patient_records = combined_df[combined_df['RecordID'] == record_id]
        row = {'RecordID': record_id}
        
        for var in key_vars:
            var_vals = patient_records[patient_records['Parameter'] == var]['Value']
            # Get first non-null value for this variable
            var_vals_clean = var_vals.dropna()
            row[var] = var_vals_clean.iloc[0] if len(var_vals_clean) > 0 else np.nan
        
        patient_data.append(row)
    
    patient_df = pd.DataFrame(patient_data)
    
    # Compute correlation
    corr_matrix = patient_df[key_vars].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        vmin=-1, vmax=1,
        square=True,
        cbar_kws={'label': 'Correlation'},
        ax=ax,
        xticklabels=[var_display.get(var, var) for var in key_vars],
        yticklabels=[var_display.get(var, var) for var in key_vars],
    )
    
    ax.set_title('Correlation Heatmap: Key Clinical Variables', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved correlation heatmap to {output_dir / 'correlation_heatmap.png'}")
    plt.close()


def main():
    """Execute all EDA analyses."""
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    output_dir = root / "outputs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("ICU XAI Exploratory Data Analysis (EDA)")
    print("=" * 80)
    
    # Load data
    print("\nLoading raw ICU data...")
    raw_df = load_all_patients(str(data_dir), dataset="set-a")
    outcomes_df = load_outcomes(str(data_dir), outcomes_file="Outcomes-a.txt")
    
    # Merge for analysis
    combined_df = raw_df.merge(outcomes_df, on="RecordID", how="left")
    
    print(f"Loaded {combined_df['RecordID'].nunique()} unique patients")
    print(f"Total records: {len(combined_df)}")
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    plot_missingness_heatmap(raw_df, output_dir)
    plot_class_imbalance(outcomes_df, output_dir)
    plot_distributions_by_outcome(combined_df, output_dir)
    plot_missingness_per_patient(raw_df, output_dir)
    plot_correlation_heatmap(combined_df, output_dir)
    
    print("\n" + "=" * 80)
    print(f"All figures saved to {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
