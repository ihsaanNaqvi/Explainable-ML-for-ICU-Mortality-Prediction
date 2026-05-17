# APPENDICES

## Appendix A — Variable list

The thirty-seven physiological and laboratory variables retained in this work, with their PhysioNet 2012 short names, the unit in which they are recorded, and the broadly accepted adult reference range.

| # | Short name | Description | Unit | Reference range |
|:-:|:-----------|:------------|:-----|:---------------:|
| 1 | ALP | Alkaline phosphatase | IU/L | 40–129 |
| 2 | ALT | Alanine aminotransferase | IU/L | 7–55 |
| 3 | AST | Aspartate aminotransferase | IU/L | 8–48 |
| 4 | Albumin | Serum albumin | g/dL | 3.5–5.0 |
| 5 | BUN | Blood urea nitrogen | mg/dL | 7–20 |
| 6 | Bilirubin | Total bilirubin | mg/dL | 0.1–1.2 |
| 7 | Cholesterol | Total cholesterol | mg/dL | < 200 |
| 8 | Creatinine | Serum creatinine | mg/dL | 0.7–1.3 |
| 9 | DiasABP | Diastolic arterial blood pressure | mmHg | 60–80 |
| 10 | FiO2 | Fraction of inspired oxygen | 0–1 | 0.21–1.0 |
| 11 | GCS | Glasgow Coma Scale | 3–15 | 15 |
| 12 | Glucose | Serum glucose | mg/dL | 70–110 |
| 13 | HCO3 | Serum bicarbonate | mmol/L | 22–29 |
| 14 | HCT | Haematocrit | % | 36–48 |
| 15 | HR | Heart rate | bpm | 60–100 |
| 16 | K | Serum potassium | mmol/L | 3.5–5.0 |
| 17 | Lactate | Serum lactate | mmol/L | 0.5–2.2 |
| 18 | MAP | Mean arterial pressure | mmHg | 70–105 |
| 19 | MechVent | Mechanical ventilation indicator | 0/1 | — |
| 20 | Mg | Serum magnesium | mg/dL | 1.7–2.2 |
| 21 | NIDiasABP | Non-invasive diastolic BP | mmHg | 60–80 |
| 22 | NIMAP | Non-invasive MAP | mmHg | 70–105 |
| 23 | NISysABP | Non-invasive systolic BP | mmHg | 90–140 |
| 24 | Na | Serum sodium | mmol/L | 135–145 |
| 25 | PaCO2 | Partial pressure of CO₂ | mmHg | 35–45 |
| 26 | PaO2 | Partial pressure of O₂ | mmHg | 75–100 |
| 27 | Platelets | Platelet count | ×10³/μL | 150–400 |
| 28 | RespRate | Respiratory rate | breaths/min | 12–20 |
| 29 | SaO2 | Arterial O₂ saturation | % | 95–100 |
| 30 | SysABP | Systolic arterial blood pressure | mmHg | 90–140 |
| 31 | Temp | Body temperature | °C | 36.5–37.5 |
| 32 | TroponinI | Troponin-I | μg/L | < 0.04 |
| 33 | TroponinT | Troponin-T | μg/L | < 0.01 |
| 34 | Urine | Hourly urine output | mL | > 0.5 mL/kg/h |
| 35 | WBC | White-cell count | ×10³/μL | 4.0–11.0 |
| 36 | Weight | Body weight | kg | — |
| 37 | pH | Arterial pH | — | 7.35–7.45 |

General descriptors (six dimensions): Age (years), Gender (0 = female, 1 = male), Height (cm), ICUType (1 = Coronary Care, 2 = Cardiac Surgery Recovery, 3 = Medical ICU, 4 = Surgical ICU), Weight at admission (kg), and recordID time-stamp. Of these, recordID is removed before modelling; the remaining five enter the feature vector.

## Appendix B — Full ablation table

| Variant | AUROC | AUPRC | Sensitivity | +Predictivity | Score1 | HL-H | TP | FP | TN | FN |
|:--------|:-----:|:-----:|:-----------:|:-------------:|:------:|:----:|:--:|:--:|:--:|:--:|
| full_model | 0.8535 | 0.4772 | 0.5060 | 0.5000 | 0.5000 | 257.30 | 42 | 42 | 475 | 41 |
| no_time_pe | 0.8594 | 0.5076 | 0.5301 | 0.5301 | 0.5301 | 249.82 | 44 | 39 | 478 | 39 |
| no_missingness | 0.8240 | 0.4576 | 0.4337 | 0.4337 | 0.4337 | 114.68 | 36 | 47 | 470 | 47 |
| no_focal_loss | 0.8437 | 0.4712 | 0.5181 | 0.5181 | 0.5181 | 26.31 | 43 | 40 | 477 | 40 |

| Δ vs full_model | ΔAUROC | ΔAUPRC | ΔScore1 | ΔHL-H |
|:----------------|:------:|:------:|:-------:|:-----:|
| no_time_pe | +0.0058 | +0.0303 | +0.0301 | −7.49 |
| no_missingness | −0.0295 | −0.0196 | −0.0663 | −142.62 |
| no_focal_loss | −0.0098 | −0.0061 | +0.0181 | −230.99 |

All variants trained with seed = 42, learning rate = 3 × 10⁻⁴, weight decay = 5 × 10⁻⁴, batch size = 64, gradient norm clipped at 1.0, patience = 25 epochs, early stopping on validation AUPRC. The full_model row is the headline Transformer configuration; each subsequent row removes exactly one design choice.

## Appendix C — Hyperparameters of the three model families

**XGBoost**
- n_estimators: 500
- learning_rate: 0.05
- max_depth: 6
- subsample: 0.8
- colsample_bytree: 0.8
- scale_pos_weight: ratio of negatives to positives in the training fold
- early stopping on validation AUPRC, patience 25 rounds

**Temporal Convolutional Network**
- Input channels: 74 (37 values + 37 masks)
- Number of filters per block: 64
- Kernel size: 3
- Dilations: {1, 2, 4, 8}
- Dropout: 0.2
- Receptive field: 31 hours
- Parameters: 58 753
- Loss: focal (α = 0.8614, γ = 2.0)
- Optimiser: Adam (lr 3 × 10⁻⁴, weight decay 5 × 10⁻⁴)

**Time-Aware Transformer**
- Input channels: 74
- d_model: 64
- Number of heads: 4
- Number of encoder layers: 4
- Feed-forward dimension: 256
- Dropout: 0.1
- Positional encoding: Time-Aware (learnable sinusoidal frequencies)
- CLS-token classification head: 64 → 32 → 1
- Parameters: 205 153
- Loss: focal (α = 0.8614, γ = 2.0)
- Optimiser: Adam (lr 3 × 10⁻⁴, weight decay 5 × 10⁻⁴)
- Best epoch: 16, total epochs: 41

## Appendix D — Figure pack

Seven publication-quality figures at 300 DPI are produced by `src/thesis_figures.py` and saved to `outputs/figures/thesis/`:

1. `ch3_missingness_profile.png` — heatmap of per-variable missingness across the 48-hour window.
2. `ch4_architecture_diagram.png` — schematic of the Time-Aware Transformer.
3. `ch5_patient_shap_heatmap.png` — paired (time × variable) SHAP attribution maps for the two case-study patients of Section 3.4.
4. `ch5_top10_shap_variables.png` — bar chart of the ten variables with the largest mean absolute SHAP attribution.
5. `ch6_shap_sofa_concordance.png` — scatter of SHAP–SAPS-I concordance against absolute calibration error per patient.
6. `model_radar_comparison.png` — radar chart of AUROC / AUPRC / Sensitivity / +Predictivity / Score1 across the four models.
7. `transformer_attention_outcomes.png` — mean attention-weight matrices for the four heads of the first encoder layer, computed across the 600 test patients.

## Appendix E — Reproducibility checklist

- **Code repository:** `d:/icu-xai/` (local checkout, public mirror to follow on acceptance).
- **Dataset:** PhysioNet/Computing in Cardiology Challenge 2012, Set-A. Available at https://physionet.org/content/challenge-2012/.
- **Random seeds:** all experiments use seed = 42 for NumPy, PyTorch (CPU and CUDA), and DataLoader shuffling.
- **Splits:** train/validation/test = 2800/600/600 patients, stratified on the mortality label.
- **Result files:** `outputs/xgboost_results.json`, `outputs/transformer_results.json`, `outputs/tcn_results.json`, `outputs/ablation_results.json`, `outputs/clinical_validation_results.json`, `outputs/final_comparison_metrics.json`, `outputs/final_results_table.csv`, `outputs/shap_matrices.npy`, `outputs/top_features_per_patient.json`.
- **Hardware:** NVIDIA RTX 3060 (12 GB).
- **Software:** Python 3.11, PyTorch 2.1, XGBoost 2.0, SHAP 0.44, scikit-learn 1.3, NumPy 1.26, Matplotlib 3.8.
- **Run order:** the canonical reproduction script is `python src/run_all.py`, which executes pre-processing → XGBoost → TCN → Transformer → SHAP → clinical validation → ablation → final comparison → figures in that order. Total wall-clock time on the reference hardware is approximately 35 minutes.
