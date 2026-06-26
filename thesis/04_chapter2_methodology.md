# CHAPTER 2. METHODOLOGY AND DATA

This chapter describes the dataset, pre-processing pipeline, three model families, the time-step SHAP attribution procedure, the clinical validation protocol, and the ablation design. The presentation is deliberately operational: every number reported in Chapter 3 was produced by the procedure described here, with random seeds, train/validation/test partitions, and hyperparameters disclosed in full.

## 2.1 Dataset

The PhysioNet/Computing in Cardiology Challenge 2012 [Silva et al., 2012] released three sets of 4000 adult ICU records each. Set-A is the only publicly labelled subset and is the focus of this work. Each record contains time-stamped physiological and laboratory observations from the first 48 hours after ICU admission together with a six-element general descriptor vector (age, gender, height, ICU type, weight, time of admission). The outcome label is in-hospital mortality, a binary variable indicating whether the patient died before discharge. The overall mortality rate after cleaning is 13.5 per cent — 554 deaths in 4000 admissions — which makes the prediction problem strongly imbalanced.

The thirty-seven physiological variables retained in this work span the full range of bedside and laboratory measurements relevant to early-ICU triage: vital signs (heart rate, mean arterial pressure, systolic and diastolic blood pressure, respiratory rate, oxygen saturation, temperature), blood gas analyses (pH, PaCO₂, PaO₂, base excess, fraction of inspired oxygen), haematology (haematocrit, white-cell count, platelets), chemistry (sodium, potassium, chloride, bicarbonate, blood urea nitrogen, creatinine, glucose, albumin, total bilirubin, alkaline phosphatase, ALT, AST, magnesium, lactate, troponin-I, troponin-T), urine output, the Glasgow Coma Scale, and the mechanical ventilation indicator. The full list with units and reference ranges is in Appendix A.

## 2.2 Pre-processing

The raw records arrive as variable-length time-stamped tuples. The first goal of pre-processing is to project each patient onto a fixed-shape tensor without falsifying the temporal information.

### 2.2.1 Hourly aggregation

For each patient *p*, each variable *v*, and each hour *h ∈ {0, ..., 47}* relative to ICU admission, the median of all observations of *v* taken between hour *h* and hour *h+1* is recorded. If no observation is taken, the cell receives a sentinel NaN. This produces a (4000, 48, 37) tensor *X*. The choice of median over mean is conservative — it discards transient spikes that may reflect sensor artefact rather than physiology — and corresponds to standard practice in the MIMIC-III benchmark of Harutyunyan and colleagues [Harutyunyan et al., 2019].

### 2.2.2 Missingness mask

A binary tensor *M ∈ {0,1}^(4000×48×37)* is constructed in parallel, with *M[p, h, v] = 1* iff at least one observation of variable *v* was recorded for patient *p* in hour *h*. The mask is concatenated with the value tensor along the variable axis, producing a (4000, 48, 74) tensor that is the actual input to the deep models. This explicit encoding of *when measurements were taken* is the single most informative architectural choice in the pipeline, as the ablation study of Section 3.6 will demonstrate.

![Figure 2.1 — Per-variable missingness profile across the 48-hour window of the PhysioNet 2012 Set-A cohort. Variables are partitioned into laboratory tests (top block, 24 variables sorted from sparsest to densest) and vital signs (bottom block, 13 variables, same ordering). The heatmap makes visible the bimodal pattern in which a small core of variables (HR 3.9 %, Temp 4.8 %, GCS 5.3 %, Urine 6.8 %) is sampled almost every hour, while laboratory measurements are sampled at unit-driven intervals of four to twelve hours, with admission-hour missingness substantially higher than steady-state missingness for every variable.](../outputs/figures/thesis/fig2_1_missingness_FIXED.png){ width=88% }

### 2.2.3 Imputation and normalisation

After mask extraction, missing cells in *X* are filled with the *per-variable cohort mean* computed on the training split only. This avoids leakage of test-set statistics into the trained model. After imputation, each variable is standardised to zero mean and unit variance using training-split statistics. The general-descriptor vector (six dimensions) is normalised analogously.

### 2.2.4 Outlier handling and physiological constraints

A small number of values in the raw records lie outside physiologically plausible ranges (negative heart rates, pH > 8, etc.) and are treated as recording errors. They are replaced with NaN before hourly aggregation and therefore contribute zero to the missingness mask, ensuring that obvious data-entry errors do not influence either the value tensor or the missingness signal.

### 2.2.5 Train/validation/test split

The 4000 patients are partitioned 70/15/15 — 2800 training, 600 validation, 600 test — stratified on the mortality label. The split is fixed across all models for direct comparability. The random seed is 42 throughout.

## 2.3 Model family I — XGBoost on engineered features

The first comparator is a gradient-boosted decision-tree model trained on a hand-engineered flat feature vector. The vector has 227 dimensions:

- **Temporal statistics (185 dimensions).** For each of the 37 physiological variables, five summaries are extracted: minimum, maximum, mean, standard deviation, and the slope of the ordinary-least-squares line fitted to the observed values against hour-of-day.
- **Missingness rates (37 dimensions).** For each variable, the fraction of the 48 hourly bins in which at least one observation was recorded.
- **General descriptors (six dimensions, of which five are retained after collinearity filtering).** Age, gender, height, weight, and ICU type.

The model is `xgboost.XGBClassifier` with 500 estimators, learning rate 0.05, max depth 6, subsample 0.8, colsample_bytree 0.8, and `scale_pos_weight` set to the ratio of negatives to positives in the training fold to handle the class imbalance. Training is performed with early stopping on the validation AUPRC, patience 25 rounds. Once trained, the model is used both for its predictive output and as the substrate for the time-step SHAP procedure of Section 2.6.

## 2.4 Model family II — Temporal Convolutional Network

The TCN baseline follows the canonical residual block design of Bai, Kolter, and Koltun [Bai et al., 2018]. The input is the (48, 74) tensor (37 normalised values + 37 missingness flags) and the output is the scalar logit. Four dilated causal convolutional blocks with dilations {1, 2, 4, 8} are stacked, each with 64 filters, kernel size 3, and dropout 0.2. The receptive field at the output layer covers 31 hours of input, which is sufficient for the 48-hour window. Total parameter count is 58 753. Training uses Adam with learning rate 3 × 10⁻⁴, weight decay 5 × 10⁻⁴, batch size 64, gradient norm clipped at 1.0, focal loss with α = 0.861 and γ = 2.0 (see Section 2.5.3), early stopping on validation AUPRC with patience 25, maximum 100 epochs.

## 2.5 Model family III — Time-Aware Transformer

The proposed model is a Transformer encoder adapted to the irregularly sampled, missingness-rich setting of bedside monitoring.

### 2.5.1 Input projection

The (48, 74) tensor is projected into model space *d_model = 64* by a learned linear layer. This is the minimum dimensionality at which four attention heads can each receive a 16-dimensional sub-space without collapsing.

### 2.5.2 Time-Aware Positional Encoding

The standard Transformer of Vaswani and colleagues [Vaswani et al., 2017] uses a fixed sinusoidal positional encoding of the form

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

For ICU physiology, the relevant time scales are unknown a priori — does the model need to discriminate between hour 14 and hour 16 with the same resolution it needs to discriminate between hour 1 and hour 40? Probably not. We therefore replace the fixed geometric frequency series 10000^(2i / d_model) with a learnable parameter ω_i ∈ ℝ^(d_model/2), initialised at the geometric series and trained jointly with the rest of the network. The construction preserves the bounded-norm, position-permutation-equivariant properties of the original encoding while allowing gradient descent to discover the appropriate time-scale spectrum.

### 2.5.3 Encoder stack

Four pre-norm Transformer encoder layers follow [Xiong et al., 2020], each with four attention heads, feed-forward dimension 256, and dropout 0.1. Pre-norm (LayerNorm before the residual sum, rather than after) was chosen because it has been shown to train more stably without warm-up on small-dimension Transformers.

### 2.5.4 CLS-token classification head

In the spirit of BERT [Devlin et al., 2019], a learnable [CLS] vector is prepended to the encoded sequence. After the encoder stack, the contextualised CLS vector is projected through a two-layer MLP (64 → 32 → 1) to produce the mortality logit. Total parameter count is 205 153.

![Figure 2.2 — Schematic of the Time-Aware Transformer. The (48 × 74) input is projected to d = 64 via a shared linear layer, summed with the learnable sinusoidal TAPE, prepended with a [CLS] token (sequence length becomes 49), and passed through four pre-norm encoder layers (Multi-Head Self-Attention with h = 4 heads, Feed-Forward 64 → 256 → 64, residual Add & LayerNorm after each sub-layer). The contextualised CLS vector feeds a two-layer MLP (64 → 32 → 1) that emits the mortality logit.](../outputs/figures/thesis/fig2_2_transformer_FIXED.png){ width=85% }

### 2.5.5 Focal loss

The class imbalance — 13.5 per cent positives — makes ordinary binary cross-entropy training unstable and biased toward the majority class. We use the focal loss of Lin and colleagues [Lin et al., 2017]:

```
FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)
```

with γ = 2 and α set to the empirical negative-class fraction in the training split (0.8614). The (1 - p_t)^γ factor down-weights easy examples and concentrates gradient on hard ones; the α reweighting offsets the class-frequency bias. The ablation study of Section 3.6 quantifies the contribution of this choice.

### 2.5.6 Optimisation

Adam with learning rate 3 × 10⁻⁴, weight decay 5 × 10⁻⁴, batch size 64, gradient norm clipped at 1.0, maximum 100 epochs, early stopping on validation AUPRC with patience 25. The best-checkpoint epoch is 16 and the run terminates at epoch 41. All randomness — weight initialisation, batch shuffling, dropout — is seeded at 42.

## 2.6 Time-step SHAP attribution

This section describes the principal methodological contribution of the thesis. The construction operates on the XGBoost model of Section 2.3 because TreeSHAP returns *exact* Shapley values in polynomial time, eliminating the sampling variance that complicates the interpretation of KernelSHAP or DeepSHAP outputs.

### 2.6.1 Variable-level SHAP via TreeSHAP

For each test patient *p*, TreeSHAP returns a vector *φ_p ∈ ℝ^227* with the property

```
f(x_p) = φ_0 + Σ_i φ_p,i
```

where *f(x_p)* is the model logit, *φ_0* the expected logit over the training set, and *φ_p,i* the Shapley value of feature *i*. Local accuracy is exact, not approximate.

Of the 227 features, 185 are temporal statistics over 37 physiological variables (five statistics per variable: min, max, mean, std, slope), 37 are missingness rates, and five are general descriptors. The variable-level Shapley contribution to the prediction for patient *p* on variable *v* is the sum of the five temporal-statistic Shapley values for variable *v* plus the missingness-rate Shapley value:

```
Φ_p,v = Σ_{s ∈ {min, max, mean, std, slope}} φ_p,(v,s) + φ_p,miss(v)
```

By the additivity axiom, the *Φ_p,v* sum to the model logit deviation across variables (plus the contribution of general descriptors).

### 2.6.2 Redistribution onto the time axis

The variable-level Shapley value *Φ_p,v* is then redistributed across the 48 hourly bins of the raw trajectory *X[p, ·, v]*. The redistribution rule is

```
Φ_p,v,h = Φ_p,v · |X[p, h, v]| / Σ_{h'} |X[p, h', v]|
```

with the convention 0/0 = 0 when the trajectory is identically zero (which on standardised data corresponds to "always at the cohort mean"). The rule satisfies three desirable properties.

1. **Conservation.** For each (p, v), the redistributed attributions sum to *Φ_p,v*. The total attribution mass is preserved.
2. **Temporal localisation.** Hours in which the variable departs furthest from the cohort mean receive proportionally more attribution. This reflects the intuition that abnormal values are the values that should drive risk estimates.
3. **Graceful degradation.** If a variable's trajectory is uniformly close to the cohort mean, the attribution is spread uniformly across hours, expressing the inability to localise.

Applied to the 600 test patients, 48 hours, and 37 variables, the procedure produces a SHAP attribution tensor *Φ ∈ ℝ^{600×48×37}* in the same coordinate system as the raw input *X*. This tensor is the primary output of the explanation pipeline and is the substrate of all downstream clinical-validation experiments.

### 2.6.3 Top-k features per patient

For clinical reporting, the procedure also outputs, for each test patient, the three variables with the largest |Φ_p,v| and the three hours with the largest column sum Σ_v |Φ_p,h,v|. These are saved to `outputs/top_features_per_patient.json` and serve as the basis for the patient-level case studies in Section 3.4.

## 2.7 Clinical validation against SOFA and SAPS-I

The validation procedure tests whether the SHAP attributions agree, on a per-patient basis, with the variables that established severity scores (SOFA, SAPS-I) emphasise for the same patient.

### 2.7.1 Reference variable sets

For each patient, the SOFA-relevant variable set is {creatinine, bilirubin, platelets, MAP, GCS, FiO₂/PaO₂ ratio}. The SAPS-I-relevant set comprises the fourteen variables listed in [Le Gall et al., 1984] that are present in PhysioNet 2012 (age, heart rate, systolic blood pressure, body temperature, mechanical ventilation, urine output, BUN, haematocrit, white-cell count, glucose, potassium, sodium, bicarbonate, GCS).

### 2.7.2 Concordance score

For each test patient *p* the model produces a ranked list of variables by total |Φ_p,v|. The top-five entries of this list are compared with the reference variable set *R* by the Jaccard-like concordance

```
C(p) = |Top5(p) ∩ R| / 5
```

which ranges from 0 (none of the model's top-five variables appear in the reference set) to 1 (all of them do).

### 2.7.3 Hypothesis test

The hypothesis is that patients with higher SHAP–clinical concordance are also patients for whom the model is better calibrated, on the rationale that a model whose attributions agree with established clinical wisdom is more likely to be making predictions for the right reasons. Calibration is measured at the patient level by the absolute error |p̂_p − y_p| where *p̂_p* is the model's predicted mortality probability and *y_p* the outcome. The Spearman rank correlation between *C(p)* and the absolute calibration error is computed across the 600 test patients, separately for SOFA and SAPS-I reference sets. A negative correlation would indicate that higher concordance is associated with lower calibration error and would support the hypothesis.

## 2.8 Ablation study

To isolate the contribution of individual design choices, four variants of the Transformer are trained under identical hyperparameters and the same seed (42):

1. **full_model** — the model of Section 2.5 in its entirety.
2. **no_time_pe** — Time-Aware Positional Encoding replaced by no positional encoding at all (purely permutation-invariant attention).
3. **no_missingness** — the missingness mask channels removed, input reduced to (48, 37).
4. **no_focal_loss** — focal loss replaced by ordinary binary cross-entropy.

All four are evaluated on the same 600-patient test split and the four PhysioNet metrics (AUROC, AUPRC, Score1, HL-H).

## 2.9 Evaluation metrics

For comparability with the original Challenge and with subsequent literature, three classes of metric are reported:

- **Discrimination.** AUROC and AUPRC. AUPRC is preferred for imbalanced classification because the baseline AUROC of a random predictor is 0.5 regardless of class balance, whereas the baseline AUPRC equals the positive-class frequency (here ≈ 0.135), which makes AUPRC more sensitive to genuine modelling gains.

- **Event 1.** *Score1 = min(Sensitivity, Positive Predictivity)* at the operating threshold that maximises this minimum. This metric, defined by the Challenge organisers, deliberately penalises models that achieve high sensitivity at the cost of positive predictivity or vice versa.

- **Event 2.** The Hosmer–Lemeshow H statistic over ten equal-mass deciles of predicted probability. Lower is better; values below ~ 20 are typically taken as evidence of adequate calibration on a cohort of this size.

## 2.10 Implementation and reproducibility

All experiments are implemented in Python 3.11 with PyTorch 2.1, XGBoost 2.0, scikit-learn 1.3, SHAP 0.44, NumPy 1.26, and Matplotlib 3.8 for figures. Random seeds are fixed at 42 for NumPy, PyTorch (CPU and CUDA RNG), and the dataloader shuffling order. Training was performed on an NVIDIA RTX 3060 (12 GB) workstation. All code is structured under `src/` with the following layout: data loaders and pre-processing under `src/data/`, model definitions under `src/models/`, SHAP under `src/shap_analysis.py`, clinical validation under `src/clinical_validation.py`, ablation under `src/ablation_study.py`, and final comparison under `src/final_comparison.py`. All result JSONs and the figure pack are written to `outputs/`.
