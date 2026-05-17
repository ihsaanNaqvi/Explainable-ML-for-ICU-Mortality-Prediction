# INTRODUCTION

## Background and motivation

Intensive care units consume between thirteen and twenty per cent of total inpatient hospital costs in high-income health systems while caring for under ten per cent of admitted patients [Halpern & Pastores, 2010]. The disproportion is driven by the density of monitoring and the irreversibility of many treatment decisions: a delayed escalation to mechanical ventilation, a missed early indication of septic shock, or an inappropriate transfer from a step-down unit each carry mortality consequences that surgical or pharmacological intervention cannot later undo. Risk stratification — the act of estimating, on the basis of measurements taken in the first hours of admission, how likely a given patient is to die before hospital discharge — has therefore been a central problem of critical care medicine since the introduction of APACHE I in 1981 [Knaus et al., 1981]. Successive generations of severity scores (APACHE II/III/IV, SAPS I/II/3, SOFA, MPM) have refined the regression weights, expanded the variable lists, and recalibrated to changing case mixes, yet all of them share two assumptions that modern data make untenable. First, they collapse the first twenty-four hours into a single worst-value summary, discarding the trajectory information that bedside clinicians act upon. Second, they treat missing observations as missing-at-random, even though it is well established that *what is not measured* in an ICU is a function of clinical judgement and therefore informative [Sharafoddini et al., 2019].

The widespread deployment of high-frequency clinical data warehouses since approximately 2010 — MIMIC-III/IV at Beth Israel Deaconess, eICU-CRD across the Philips network, AmsterdamUMCdb in the Netherlands, and the PhysioNet/Computing in Cardiology Challenge corpora — has made it feasible to train far more flexible models. Deep learning architectures originally developed for natural language processing (recurrent networks, temporal convolutional networks, Transformers) translate naturally to multivariate clinical time series, and have repeatedly been shown to outperform classical severity scores under area-under-the-curve metrics by five to ten percentage points [Shickel et al., 2018; Rajkomar et al., 2018; Sheikhalishahi et al., 2020]. Yet adoption at the bedside has been slow. The reason is not metric performance but trust: a Transformer that emits a probability of 0.74 without explaining *why* offers the attending physician nothing they can argue with on a ward round, nothing they can defend in a morbidity-and-mortality conference, and nothing they can correct if the model has latched onto a confounding artefact of the unit's documentation habits.

Explainable artificial intelligence (XAI) is the response to this trust gap. The two methods that have come to dominate the post-hoc explanation literature are LIME [Ribeiro et al., 2016] and SHAP [Lundberg & Lee, 2017]. SHAP, which extends Shapley values from cooperative game theory to feature attribution, has particularly desirable theoretical properties — local accuracy, missingness, consistency — that make its outputs admissible as the basis for clinical reasoning. In its standard form, however, SHAP returns one attribution per input feature, which is appropriate for tabular data but loses essential information when the input is a multivariate trajectory: knowing that "heart rate matters" is far less useful to a clinician than knowing that "the heart rate excursion between hour 14 and hour 19 mattered." Reconstructing the temporal axis from a model trained on flattened features is the methodological problem that this thesis addresses.

## Research problem

Two related problems are studied:

1. **Modelling problem.** Construct a predictor of in-hospital mortality from the first forty-eight hours of ICU monitoring on PhysioNet 2012 Set-A that matches or surpasses both classical severity scores and recent deep learning baselines on the official Event 1 (discrimination at the optimal operating point) and Event 2 (calibration) metrics, while respecting the imbalanced (≈13.5 % positive) nature of the cohort.

2. **Explanation problem.** Produce, for each individual prediction, an attribution map of shape *(time × variable)* that identifies the specific hourly bins of the specific physiological variables driving the model's risk estimate, and demonstrate that those attributions are clinically meaningful — meaning, in particular, that they agree on average with the variables that established severity scores (SOFA, SAPS-I) draw upon for the same patient.

## Research questions

The thesis is organised around four questions:

- **RQ1.** Can a model that explicitly encodes missingness as a parallel channel of binary masks outperform an identically configured model that imputes missing values silently?
- **RQ2.** Does a Time-Aware Transformer architecture, in which positional encodings are learnable rather than fixed, achieve discrimination comparable to a heavily feature-engineered gradient-boosted tree model on this dataset?
- **RQ3.** How can a SHAP explanation computed on a flat feature vector be redistributed back onto the (time × variable) coordinate system in a way that preserves the total attribution and reflects the temporal incidence of evidence?
- **RQ4.** Do the resulting per-patient attribution maps agree with the variables that established clinical severity scores (SOFA, SAPS-I) emphasise for the same patient?

## Aim and objectives

The aim of the thesis is to develop, evaluate, and explain a deep learning model for early in-hospital mortality prediction on PhysioNet 2012 Set-A. The aim decomposes into the following objectives:

1. Pre-process the 4000-patient cohort into a uniformly shaped (4000, 48, 37) tensor while preserving information about when each measurement was taken.
2. Train and evaluate three model families — XGBoost, TCN, Time-Aware Transformer — under an identical experimental protocol and on the official Challenge metrics.
3. Develop a time-step SHAP attribution procedure that maps variable-level Shapley values onto the (patient, hour, variable) coordinate system.
4. Validate the clinical plausibility of the resulting attributions by comparing them against SOFA and SAPS-I component variables on a per-patient basis.
5. Conduct an ablation study to identify which architectural and loss-function choices contribute most to performance.

## Scientific novelty

Three contributions distinguish this work from the existing literature:

- **Time-step SHAP redistribution.** The procedure described in Chapter 2 computes Shapley values on a flattened feature vector and then redistributes each value across the corresponding temporal trajectory in proportion to the absolute observed values, producing an attribution tensor of the same shape as the raw input. The construction preserves the local accuracy property of SHAP at the variable level while restoring the temporal axis. The redistribution is exact when the observed trajectory contains a single non-zero value and degrades gracefully when measurements are uniformly distributed.

- **Quantitative clinical validation of post-hoc explanations.** Rather than presenting attribution maps as a qualitative illustration, the thesis defines a per-patient concordance score against SOFA and SAPS-I component variables and tests, via Spearman correlation between concordance and calibration error, whether patients for whom the model "agrees with the clinical scores" are also patients for whom the model is better calibrated. The hypothesis is supported (ρ = 0.139, p = 6.3 × 10⁻⁴ for SAPS-I), providing one of the few statistically tested clinical-plausibility results in the SHAP-for-ICU literature.

- **Ablation isolating informative missingness.** The ablation table in Chapter 3 isolates four variants of the Transformer and shows that removing the missingness-mask channels degrades Score1 by 6.6 points and the Hosmer–Lemeshow statistic by more than 140 units. This quantifies the contribution of informative missingness on PhysioNet 2012 with a level of precision absent from the published literature on this corpus.

## Significance

For the engineering side, the thesis provides a reproducible pipeline (227-feature XGBoost, dilated TCN, 205 k-parameter Transformer) calibrated to a public benchmark, with all hyperparameters and training seeds disclosed. For the clinical side, it offers a per-patient explanation tool whose output lives in a coordinate system clinicians already inhabit — hours since admission × physiological variable — rather than a coordinate system imposed by the model.

## Structure of the thesis

The remainder of the thesis is organised as follows. Chapter 1 surveys the literature on severity scoring, deep learning for ICU prediction, and explainable artificial intelligence. Chapter 2 details the dataset, pre-processing, model architectures, the time-step SHAP procedure, and the clinical validation protocol. Chapter 3 presents the comparative results, the ablation analysis, the attribution case studies, and discusses what the experiments do and do not show. The Conclusion summarises the contributions and points to extensions on MIMIC-IV. Appendices contain the variable list, the full ablation table, and the figure pack at publication resolution.
