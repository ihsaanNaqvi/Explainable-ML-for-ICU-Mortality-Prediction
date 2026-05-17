# CONCLUSION

This thesis set out to construct, evaluate, and explain a machine-learning predictor of in-hospital mortality from the first forty-eight hours of intensive-care monitoring on the PhysioNet/Computing in Cardiology Challenge 2012 Set-A cohort. The work was framed by four research questions on missingness encoding (RQ1), architecture (RQ2), time-step SHAP attribution (RQ3), and clinical-score concordance (RQ4). The conclusion summarises the answers, the contributions to knowledge, the practical implications, and the open questions that the experiments did not resolve.

## Summary of findings

**Modelling.** Three model families were compared under identical pre-processing, splits, and evaluation metrics. The gradient-boosted tree model on 227 hand-engineered features achieved AUROC = 0.8798, Score1 = 0.5422, and Hosmer–Lemeshow H = 37.67, the strongest overall performance on the cohort. The Time-Aware Transformer achieved AUROC = 0.8542 and Score1 = 0.5301, marginally below XGBoost on discrimination but with poorer calibration. The Temporal Convolutional Network trailed both. All three models substantially exceeded the SAPS-I baseline of Score1 = 0.3097.

**RQ1 — Missingness encoding.** The ablation study isolated the contribution of explicit missingness-mask channels and found that their removal degrades the Transformer's Score1 by 6.6 percentage points, the largest single-component effect in the ablation table. Encoding *when* measurements were taken is therefore not an incidental modelling detail but the dominant architectural choice on this corpus. The result aligns with, and quantifies on PhysioNet 2012, the GRU-D finding of Che and colleagues [Che et al., 2018].

**RQ2 — Architecture.** The Time-Aware Transformer matched but did not exceed the XGBoost baseline. The honest conclusion is that on a 48-hour, 37-variable cohort of 4000 patients, hand-engineered features feeding a tuned gradient-boosted tree are still competitive with deep sequence models. The deep models begin to pull ahead in regimes where feature engineering becomes brittle (longer windows, larger variable sets, multi-task pre-training), and the present cohort is below that scale threshold.

**RQ3 — Time-step SHAP attribution.** The redistribution procedure of Section 2.6, which preserves the local accuracy of variable-level Shapley values while restoring the temporal axis through a proportional-to-|value| rule, produced attribution tensors of shape (600 × 48 × 37) for the test cohort. The top-ten attributed variables on the global cohort — GCS, mechanical ventilation, urine output, BUN, creatinine, age, heart rate, MAP, lactate, glucose — coincide with the variables that classical severity scores draw upon, providing prima facie evidence that the model's attributions are clinically plausible.

**RQ4 — Clinical-score concordance.** The mean SHAP–SAPS-I concordance across the 600 test patients is 0.498 (chance baseline ≈ 0.135), and the Spearman correlation between concordance and absolute calibration error is statistically significant (ρ = 0.139, p = 6.3 × 10⁻⁴). The relationship is in the opposite direction to the original hypothesis — patients on whom the model leans on classical-score variables are not the best-calibrated patients — but its statistical detectability provides quantitative support for the claim that SHAP attributions carry clinically meaningful signal on this dataset.

## Contributions to knowledge

The thesis advances three contributions that are, to the author's knowledge, novel in the published literature on PhysioNet 2012:

1. **Time-step SHAP redistribution procedure.** A deterministic, conservation-preserving rule for mapping flat-feature Shapley values back onto the (time × variable) coordinate system of the raw input, retaining the computational efficiency of TreeSHAP while restoring the temporal axis that flat-feature SHAP otherwise discards.

2. **Quantitative clinical validation of SHAP attributions on PhysioNet 2012.** The per-patient concordance score and the Spearman test against calibration error replace the prevailing qualitative-figure mode of presenting SHAP results in the ICU literature with a falsifiable hypothesis on a public benchmark.

3. **Ablation isolating informative missingness on PhysioNet 2012.** A controlled four-variant ablation that pins down the contribution of missingness encoding to within ± a tenth of a percentage point of Score1, providing a level of precision absent from the existing published work on this cohort.

## Practical and clinical implications

For clinicians and machine-learning engineers working on bedside risk-stratification tools, three implications follow.

The first concerns *what the model sees*. The single largest gain in this thesis came from making missingness an explicit input channel rather than imputing and forgetting. Any production pipeline that silently fills missing values with cohort means or last-observation-carried-forward is discarding a stratum of evidence that the data warehouse already contains and that established clinical intuition already weighs.

The second concerns *what the model explains*. The time-step SHAP procedure produces attribution maps in the same coordinate system the clinician already inhabits — hours since admission × physiological variable — which is a precondition for the maps being usable on a ward round. A model that emits a probability is a model that gets argued with; a model that emits a probability *and* a (time × variable) heatmap is a model that gets reasoned with.

The third concerns *what the model is asked to do*. Focal loss optimises discrimination at the cost of calibration; binary cross-entropy with class weighting does the reverse. If the deployment target is a triage rank-list, focal loss is preferable. If the target is a calibrated risk estimate that downstream resource-allocation logic depends on, BCE — or post-hoc temperature scaling on a focal-loss-trained model — is the right choice. The ablation table in this thesis is the empirical evidence on which that choice should be made.

## Open questions and future work

The experiments leave four directions for extension.

**External validation.** Re-running the pipeline on MIMIC-IV [Johnson et al., 2023] and eICU-CRD [Pollard et al., 2018] would test whether the Time-Aware Transformer and the time-step SHAP procedure generalise beyond the PhysioNet 2012 cohort. MIMIC-IV in particular, with its substantially larger patient count and richer variable list, would test whether the architecture begins to pull ahead of XGBoost in the regime where feature engineering becomes brittle.

**Counterfactual extension.** The thesis establishes that SHAP attributions correlate with established clinical severity scores. It does not establish that those attributions are causal. The natural extension is to combine the time-step SHAP procedure with a counterfactual estimator [Pearl, 2009] that asks how the predicted risk would change if a specific variable trajectory had taken a different value during a specific hour. This is the move from explainable to actionable AI.

**Calibration-aware training.** Section 3.2 documented the focal-loss-induced calibration penalty. Training under the focal-with-temperature loss of Mukhoti and colleagues [Mukhoti et al., 2020] would test whether the discrimination/calibration trade-off can be partially dissolved rather than merely managed by post-hoc rescaling.

**Prospective clinical study.** All evaluation in this thesis is retrospective on a historical cohort. A meaningful test of clinical utility requires a prospective study in which the model is run live in a unit, attending clinicians are asked to use the explanation maps during ward rounds, and downstream decisions (escalation, transfer, code status) are recorded as outcomes. This is the eventual destination of the line of work but lies well beyond the scope of an MS thesis.

## Final remarks

The history of intensive-care risk stratification, from APACHE I in 1981 to the present, is a history of trading interpretability for accuracy. Logistic regression on twelve variables yielded readily defensible numbers but discriminated weakly; deep sequence models on hundreds of variables and thousands of time steps discriminate well but defend their numbers poorly. The thesis argues that this trade-off is no longer forced. The combination of (a) a Transformer architecture that explicitly encodes missingness, (b) a SHAP procedure that returns attributions on the time × variable axis the clinician already uses, and (c) a quantitative validation against established severity scores, recovers most of the accuracy of the deep approach while restoring most of the interpretability of the classical approach. The remaining gap — calibration, prospective validation, causal interpretation — is the next decade of work.

The contribution that this MS thesis hopes to defend is not a state-of-the-art number on a benchmark, although the numbers are competitive. It is the demonstration, on a public cohort and with disclosed seeds, that an explanation procedure for ICU mortality prediction can be both temporally resolved and clinically validated at the same time. That demonstration, if it survives external replication, is one small step toward the kind of model an attending physician can argue with at the bedside.
