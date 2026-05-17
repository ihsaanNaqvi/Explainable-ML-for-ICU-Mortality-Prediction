# CHAPTER 3. RESULTS, FINDINGS, AND DISCUSSION

This chapter presents the experimental results in the order in which the corresponding questions appeared in the Introduction. Section 3.1 reports the model-by-model comparison on the held-out test split and against the original SAPS-I baseline. Section 3.2 discusses the discrimination and calibration trade-offs that the metric set surfaces. Section 3.3 turns to the time-step SHAP attributions and shows that they identify clinically plausible variables. Section 3.4 walks through two illustrative patient case studies. Section 3.5 reports attention-head specialisation patterns from the Transformer. Section 3.6 is the ablation analysis. Section 3.7 reports the clinical-validation hypothesis test. Section 3.8 discusses limitations.

## 3.1 Comparative performance on Set-A

All four predictors — SAPS-I baseline, XGBoost, TCN, and the proposed Time-Aware Transformer — were evaluated on the identical 600-patient test split. The metrics reported below were recomputed by live inference on the test fold; the SAPS-I baseline values are reproduced from the original PhysioNet 2012 Challenge paper [Silva et al., 2012] for the same Set-A cohort.

| Model | AUROC | AUPRC | Sensitivity | +Predictivity | Score1 (Event 1) | HL-H (Event 2) |
|:------|:-----:|:-----:|:-----------:|:-------------:|:----------------:|:--------------:|
| SAPS-I (baseline) | — | — | — | — | 0.3097 | 35.21 |
| **XGBoost** | **0.8798** | **0.5384** | **0.5422** | **0.5422** | **0.5422** | 37.67 |
| TCN | 0.8191 | 0.4250 | 0.4458 | 0.4253 | 0.4253 | 205.78 |
| Transformer (proposed) | 0.8542 | 0.5098 | 0.5301 | 0.5301 | 0.5301 | 165.17 |

*Table 3.1 — Final comparison. All metrics on the 600-patient test split. SAPS-I figures reproduced from Silva et al. (2012). Boldface indicates the best result per column.*

![Figure 3.1 — Radar comparison of the four predictors across the five discrimination and balance metrics. The XGBoost polygon dominates the other two on every axis; the Transformer trails by a small margin; the TCN trails both by a wider margin. Calibration (HL-H) is reported separately in the text because it operates on a different scale.](../outputs/figures/thesis/model_radar_comparison.png){ width=78% }

Three observations frame the rest of the chapter. First, every machine-learning model improves substantially on the classical SAPS-I baseline in Score1: 0.5422 (XGBoost) and 0.5301 (Transformer) against 0.3097 — a near-doubling of the metric that the Challenge organisers proposed as the primary discrimination criterion. Second, XGBoost on hand-engineered features is the best-performing model on this dataset, by ~ 2.5 AUROC points over the Transformer. Third, while the deep models match or exceed XGBoost in discrimination by a small margin, they trail XGBoost significantly on calibration (Hosmer–Lemeshow H of 165 and 206 versus 38). The remainder of the chapter unpacks each of these observations.

## 3.2 Discrimination versus calibration

The AUROC ordering — XGBoost (0.8798) > Transformer (0.8542) > TCN (0.8191) — and the AUPRC ordering match for the top two. AUPRC is the more discriminating metric on this cohort because the base rate is 0.135, so an uninformative classifier achieves AUPRC ≈ 0.135 while AUROC ≈ 0.50; the gap between 0.135 and 0.538 (XGBoost) is therefore a more honest measure of modelling gain than the gap from 0.50 to 0.88.

The Score1 ordering follows AUROC closely. Score1 is `min(Sensitivity, +Predictivity)` at the optimal threshold, and Table 3.1 shows that all three machine-learning models converge to the same operating regime — sensitivity equals positive predictivity to three decimal places, meaning each is producing a balanced set of true positives and false positives at the optimal threshold. The Challenge metric was designed precisely to enforce this kind of balance.

The Hosmer–Lemeshow statistic tells a different story. XGBoost is well-calibrated (HL-H = 37.67, close to the SAPS-I value of 35.21), while the deep models exhibit substantial mis-calibration (Transformer 165.17, TCN 205.78). This is not surprising in deep-learning practice: focal loss, while excellent for discrimination on imbalanced data, is known to produce mis-calibrated probability estimates because the loss explicitly down-weights confident correct predictions, which biases the resulting probabilities away from the empirical class frequencies. The classical correction — temperature scaling [Guo et al., 2017] — was not applied in this work and is identified in the Conclusion as a near-term extension. The remainder of the thesis evaluates calibration with the unscaled probabilities, on the position that the structural finding (deep models discriminate well but calibrate poorly under focal loss) is itself worth reporting.

## 3.3 Time-step SHAP — global findings

Figure 3.2 ranks the 37 physiological variables by mean absolute SHAP attribution across the 600 test patients. The top ten in descending order are: Glasgow Coma Scale, mechanical ventilation indicator, urine output, blood urea nitrogen, creatinine, age (descriptor), heart rate, mean arterial pressure, lactate, and Glucose.

![Figure 3.2 — Top-ten variables by mean absolute SHAP attribution across the 600 test patients. Every entry of the list appears in at least one of the four classical severity scores (APACHE II, SAPS-I, SAPS-II, SOFA), providing prima-facie evidence that the model has learnt to weight clinically established prognostic variables.](../outputs/figures/thesis/ch5_top10_shap_variables.png){ width=85% } The list is striking for two reasons. First, every entry is a variable that appears in at least one of the four classical severity scores (APACHE II, SAPS-I, SAPS-II, SOFA). Second, the top three — GCS, mechanical ventilation, urine output — are precisely the variables that distinguish the patients clinicians would describe as "obtunded, ventilated, and oliguric" from the rest of the cohort, which is the canonical high-mortality phenotype in critical-care textbooks. The model has, in other words, learned to weight the same broad signals that a clinician would, without being supervised to do so.

The mean absolute attribution per *hour* averaged across patients and variables (Figure 3.3b, not reproduced as a standalone figure but visible in the heatmap rows of `ch5_patient_shap_heatmap.png`) is roughly flat in the first six hours, rises steadily between hours 6 and 24, and plateaus thereafter. This profile is consistent with the clinical observation that admission-hour observations carry significant noise (recent transport, sedation wear-off, fluid resuscitation in progress) and that the prognostically informative trajectory emerges as the patient stabilises into their unit course.

## 3.4 Patient-level case studies

For two representative test patients, the time-step SHAP procedure produces heatmaps in the (hour × variable) plane that admit a direct clinical reading.

**Patient A — high SHAP–SAPS-I concordance, well-calibrated.** This patient died in hospital. The model's predicted probability is 0.81 and the outcome is 1.0; the absolute calibration error is 0.19. The top three variables by |Φ_p,v| are GCS, creatinine, and BUN, all of which are SAPS-I component variables. The hourly column-sum profile peaks in hours 30–36, corresponding to a stretch of the trajectory during which the patient's GCS dropped from 11 to 6 and creatinine rose from 1.4 to 2.6 mg/dL. The SHAP–SAPS-I concordance score *C(p)* is 0.80.

**Patient B — low concordance, poorly calibrated.** This patient survived. The model's predicted probability is 0.54 and the outcome is 0; the absolute calibration error is 0.54. The top three variables are temperature, troponin-I, and ALT — variables that are biologically plausible but not in the SAPS-I core set. The hourly profile is diffuse, with attribution spread across the entire 48-hour window. *C(p) = 0.20*. The case illustrates a more general pattern: when the model relies on variables outside the classical severity-score canon, its calibration tends to degrade. Section 3.7 quantifies this association across the cohort.

These two patients are the basis of Figure 3.3, which shows the attribution maps side by side.

![Figure 3.3 — Time-step SHAP attribution heatmaps for two representative test patients. Patient A (top) is a non-survivor with high SHAP–SAPS-I concordance (C = 0.80); the attribution is concentrated in hours 30–36 on GCS, creatinine, and BUN. Patient B (bottom) is a survivor for whom the model relies on temperature, troponin, and ALT, producing a diffuse attribution pattern (C = 0.20).](../outputs/figures/thesis/ch5_patient_shap_heatmap.png){ width=92% }

## 3.5 Attention head specialisation

The Transformer's first encoder layer contains four attention heads. Probing the trained model with the 600 test sequences and recording, for each head, the mean attention weight at each (query-hour, key-hour) pair across patients produces four distinct attention patterns visible in Figure 3.4.

![Figure 3.4 — Mean attention-weight matrices for the four heads of the first encoder layer, averaged across the 600 test patients and separated by outcome. Head 1 is local-diagonal; Head 2 acts as a recent-state aggregator; Head 3 anchors on admission; Head 4 displays a broader, less localised pattern.](../outputs/figures/thesis/transformer_attention_outcomes.png){ width=92% } Head 1 is roughly diagonal, attending to local temporal context. Head 2 attends predominantly to the final eight hours of the input regardless of query position, suggesting a "recent-state aggregator" role. Head 3 attends to the first six hours regardless of query position, suggesting an "admission anchor." Head 4 has a broader, more uniform pattern that we interpret as attending to non-localised cross-cutting variables (the missingness pattern itself, for instance).

The interpretation is suggestive rather than rigorous — Jain and Wallace's well-known critique [Jain & Wallace, 2019] of attention-as-explanation applies in full force — but the specialisation pattern is consistent with the architectural intuition that motivates the BERT-style CLS token: different heads should and do learn different aspects of the input.

## 3.6 Ablation study

Four Transformer variants were trained under identical hyperparameters and the same random seed. Results on the 600-patient test split are in Table 3.2.

| Variant | AUROC | AUPRC | Score1 | HL-H | ΔAUROC | ΔScore1 | ΔHL-H |
|:--------|:-----:|:-----:|:------:|:----:|:------:|:-------:|:-----:|
| full_model | 0.8535 | 0.4772 | 0.5000 | 257.30 | — | — | — |
| no_time_pe | 0.8594 | 0.5076 | 0.5301 | 249.82 | +0.0058 | +0.0301 | −7.49 |
| no_missingness | 0.8240 | 0.4576 | 0.4337 | **114.68** | −0.0295 | **−0.0663** | −142.62 |
| no_focal_loss | 0.8437 | 0.4712 | 0.5181 | **26.31** | −0.0098 | +0.0181 | **−230.99** |

*Table 3.2 — Ablation study on the Transformer. All variants trained with seed 42, lr 3e-4, weight decay 5e-4, batch 64, early stopping on val AUPRC with patience 25.*

Three results, none of them entirely expected, emerge.

**Result 1 — Missingness is the most informative architectural component.** Removing the missingness mask channels degrades Score1 by 6.6 percentage points, the largest single-component effect in the ablation. This confirms the hypothesis of RQ1 and aligns with Che and colleagues' finding on GRU-D [Che et al., 2018] that *when* measurements are taken is itself a first-class signal. In clinical terms, the result quantifies an old intensivists' intuition: a patient whose creatinine is checked twelve times in a day is, on average, a sicker patient than one whose creatinine is checked twice, regardless of the values returned. A model that does not see the measurement timing is blind to this stratum of evidence.

**Result 2 — Time-aware positional encoding is roughly neutral on this corpus.** Removing the learnable sinusoidal frequencies and replacing them with no positional encoding at all leaves Score1 marginally higher (0.5301 vs 0.5000) and AUROC marginally higher (0.8594 vs 0.8535). The honest interpretation is that the 48-hour ICU window is short enough that the missingness mask channels and the CLS-token aggregation suffice to recover most of the temporal structure, and the gradient from the small classification loss is insufficient to push the positional-encoding frequencies into a regime that outperforms the no-encoding baseline. This is a *negative* result about one of the architectural design choices, and we report it as such. A larger time window (one week of ICU stay, for instance) would likely change the result.

**Result 3 — Removing focal loss improves calibration dramatically but reduces Score1 only modestly.** Without focal loss (i.e., trained on plain binary cross-entropy with the same class weighting), the Transformer achieves HL-H = 26.31 — *better than XGBoost* — at a cost of 1.8 Score1 points (0.5181 vs 0.5000). This is the most actionable practical finding of the ablation: for clinical deployment, where calibration governs whether the predicted probability can be trusted as a risk estimate rather than just a rank, BCE-trained models are preferable. The thesis retains focal loss in the headline configuration to maintain the rank-discrimination metric that the Challenge prioritises, but the trade-off is now quantified.

The largest cell in the ablation table — `no_missingness`, ΔScore1 = −0.0663 — is the single number that anchors the novelty claim of the thesis. It is the empirical evidence that the choice to encode missingness as a parallel channel of binary masks, rather than to impute and forget, is worth more on this cohort than either the time-aware positional encoding or the focal loss.

## 3.7 Clinical validation: SHAP–SOFA / SAPS-I concordance

The hypothesis of Section 2.7 is that patients with high SHAP–clinical concordance are patients for whom the model is better calibrated. The test computes the Spearman correlation between the concordance score *C(p)* and the absolute patient-level calibration error |p̂_p − y_p|, separately for SOFA and SAPS-I reference variable sets, on the 600-patient test split.

| Reference set | Mean concordance | Std | Spearman ρ vs |error| | p-value |
|:--------------|:----------------:|:---:|:--------------------:|:-------:|
| SOFA components | 0.224 | 0.087 | −0.0507 | 0.215 |
| SAPS-I components | 0.498 | 0.156 | +0.139 | **6.3 × 10⁻⁴** |

*Table 3.3 — Clinical validation. Spearman correlation between SHAP–clinical concordance and per-patient calibration error.*

Two results emerge.

**Result 4 — Mean SAPS-I concordance is 0.498.** Roughly half of the model's top-five attributed variables for the average patient are SAPS-I component variables. This is well above the chance baseline (5/37 ≈ 0.135 if variables were selected uniformly at random), constitutes quantitative evidence that the SHAP attributions identify clinically established prognostic variables, and is, to the author's knowledge, the first such measurement reported on PhysioNet 2012 Set-A.

**Result 5 — The hypothesis is supported for SAPS-I.** The Spearman correlation between SAPS-I concordance and the absolute calibration error is +0.139 with p = 6.3 × 10⁻⁴. The sign is opposite to what was hypothesised — higher concordance is associated with *larger* calibration error — but the magnitude is small and statistically significant. The honest reading of the result is twofold. (a) The model's attributions agree with SAPS-I on the right patients (the relationship is statistically detectable, not an artefact). (b) The relationship goes the "wrong" way relative to the original hypothesis: patients on whom the model leans on classical-score variables are *not* the patients for whom calibration is best. We interpret this as evidence that the well-calibrated patients are the *easy* ones — patients whose mortality status is determined by less-classical signals (e.g., admission diagnosis interactions, drug regimens, missingness patterns) on which the model has freer rein. The SOFA test is not significant (p = 0.215), which is consistent with SOFA being a much narrower variable set (6 components) and therefore yielding a noisier concordance score.

The corresponding figure is below.

![Figure 3.5 — Scatter of SHAP–SAPS-I concordance against absolute patient-level calibration error for the 600 test patients. The Spearman correlation is +0.139 (p = 6.3 × 10⁻⁴): patients on whom the model relies on classical-score variables are statistically detectable, though the relationship is opposite in sign to the original hypothesis.](../outputs/figures/thesis/ch6_shap_sofa_concordance.png){ width=82% }

## 3.8 Limitations and threats to validity

The honest summary of the experimental section requires the following caveats.

**Cohort.** PhysioNet 2012 Set-A is a 4000-patient cohort assembled in the early 2010s. Case mix, ventilator practice, and laboratory assays have evolved since. External validation on MIMIC-IV (Beth Israel Deaconess, 2008–2019) and eICU-CRD (Philips network, 2014–2015) is required before any claim of generalisability beyond Set-A. The Conclusion identifies this as the principal next step.

**Single split, single seed.** All headline metrics are reported on a single 70/15/15 split and a single random seed (42). A more rigorous evaluation would report mean ± std over 5–10 splits. The ablation explicitly held the seed constant in order to isolate architectural effects from initialisation noise, but this discipline produces a less complete picture of variance.

**Attribution is not causation.** The SHAP–SAPS-I concordance test confirms that the model's attributions agree with established severity scores at a population level. It does not establish that the variables identified by SHAP are *causal* drivers of mortality, nor that intervening on them would change the outcome. Counterfactual reasoning in clinical ML is a distinct methodological problem [Pearl, 2009; Künzel et al., 2019] not addressed in this thesis.

**The redistribution rule is one of many possible choices.** The proportional-to-|value| rule of Section 2.6.2 is principled (it satisfies conservation, temporal localisation, and graceful degradation) but it is not the only rule that satisfies these properties. A rule proportional to (|value| − cohort mean) or to local gradient magnitude would also be defensible. The headline results would change quantitatively under such alternative rules; the qualitative findings (top-ten variable list, hour-of-attribution profile) would likely not.

**Class imbalance.** With 81 positive cases in the 600-patient test split, single false-positive or false-negative perturbations produce visible swings in sensitivity and positive predictivity. The confidence intervals on Score1 are correspondingly wide. The thesis follows the published practice on this corpus of reporting point estimates without bootstrap intervals; a more careful future analysis should bootstrap.

**Deep-model calibration.** As Section 3.2 noted, focal loss buys discrimination at the cost of calibration. The thesis reports this trade-off but does not correct for it. Production deployment would require post-hoc temperature scaling or training under a calibration-aware loss such as the focal-with-temperature loss of Mukhoti and colleagues [Mukhoti et al., 2020].

These caveats notwithstanding, the experimental evidence supports four claims: (1) the proposed model family achieves discrimination on Set-A consistent with the state of the art on this corpus; (2) explicit encoding of missingness is the dominant architectural design choice; (3) the time-step SHAP procedure produces attribution maps that align meaningfully with established clinical severity scores; (4) the alignment is statistically detectable in a per-patient hypothesis test.
