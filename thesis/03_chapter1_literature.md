# CHAPTER 1. LITERATURE REVIEW

This chapter situates the thesis at the intersection of three established research traditions: classical severity scoring in critical care medicine, machine learning approaches to clinical time series, and the still-maturing field of explainable artificial intelligence (XAI). The aim is not encyclopaedic coverage but enough orientation for a reader who is comfortable with one of these traditions to understand the contributions of the other two and to see why their combination is non-trivial.

## 1.1 Mortality prediction in the intensive care unit

### 1.1.1 The clinical problem

An adult patient admitted to an intensive care unit in a contemporary tertiary hospital generates, conservatively, tens of thousands of measurements in the first day: bedside monitors sample heart rate, blood pressure, oxygen saturation, and respiratory rate at the minute scale; the laboratory returns dozens of chemistry and haematology panels; nurses chart fluid balance, sedation level, and Glasgow Coma Scale hourly. From this torrent the attending intensivist must construct, by the end of the morning ward round, a working estimate of how sick the patient is, how that compares to other patients on the unit, and which interventions will most reduce the conditional probability of in-hospital death. Severity scores formalise the first two operations.

### 1.1.2 The APACHE family

The Acute Physiology and Chronic Health Evaluation (APACHE) system, introduced by Knaus and colleagues in 1981 [Knaus et al., 1981], assigned weights to the worst values of twelve physiological variables in the first twenty-four hours, added points for chronic health status and age, and predicted hospital mortality through a logistic regression calibrated on a cohort of approximately 800 patients. Successive revisions — APACHE II (1985), APACHE III (1991), APACHE IV (2006) — expanded the variable list, recalibrated the regression coefficients to changing case mixes, and incorporated diagnosis-specific terms. The most recent version, APACHE IV, achieves area under the receiver operating characteristic curve (AUROC) of approximately 0.88 on the cohort of 110 000 admissions on which it was developed [Zimmerman et al., 2006], although external validation studies typically report values in the 0.80–0.85 range.

### 1.1.3 The SAPS family

The Simplified Acute Physiology Score, developed in parallel by Le Gall and colleagues in France [Le Gall et al., 1984; 1993; 2005], pursued the same goal with a smaller variable list. SAPS-I uses fourteen variables and is the score originally embedded in the PhysioNet 2012 Challenge as a baseline. SAPS-II uses seventeen variables and was for many years the most widely deployed score in Europe. SAPS-3 abandons the worst-value-in-24-hours convention in favour of admission-time variables only, on the argument that one of the points of a severity score is to be available before treatment effects confound the prediction.

### 1.1.4 SOFA

The Sequential Organ Failure Assessment score, introduced at a 1994 European Society of Intensive Care Medicine consensus conference and formalised by Vincent and colleagues [Vincent et al., 1996], differs in purpose from APACHE and SAPS. It is a *dynamic* score: each of six organ systems (respiratory, coagulation, hepatic, cardiovascular, central nervous system, renal) is assigned an integer 0–4, and the score is recomputed daily. It was not designed as a mortality predictor but has been widely repurposed as one. SOFA remains the score embedded in the consensus definition of sepsis [Singer et al., 2016]. The thesis uses SOFA component variables — creatinine, bilirubin, platelets, MAP, Glasgow Coma Scale, FiO2/PaO2 ratio — as one of the two reference yardsticks against which the SHAP attributions are validated.

### 1.1.5 Limitations of the classical scores

Three limitations of the classical scores motivate the present thesis. First, they collapse the trajectory of each variable into a single worst-value summary, discarding information about the *order* and *rate* of physiological change. A patient whose creatinine rises monotonically from 1.0 to 3.0 mg/dL over twenty-four hours has the same SAPS contribution as a patient with a single spurious creatinine of 3.0 at hour eight that resolves by hour twelve, even though their prognoses differ. Second, they treat missing observations as exchangeable with measured-and-normal observations, which contradicts an essential property of clinical data: variables that are *not* measured are not measured because the responsible clinician judged, implicitly, that they were not needed [Sharafoddini et al., 2019; Agniel et al., 2018]. Third, the regression coefficients are static once calibrated, which renders them vulnerable to dataset shift as case mix, treatment patterns, and documentation conventions change over decades.

## 1.2 Machine learning on ICU time series

### 1.2.1 Pre-deep-learning era

Before approximately 2015 the dominant machine-learning approach to ICU mortality prediction was to extract a fixed-length vector of summary statistics from each patient's first-day trajectory and feed it to a logistic regression, random forest, or gradient-boosted tree. Pirracchio and colleagues' Super Learner [Pirracchio et al., 2015] is a well-known representative: a stacked ensemble of multiple base learners trained on physiology, demographics, and admission diagnoses, achieving AUROC of 0.88 on MIMIC-II — roughly five points above SAPS-II on the same cohort. Johnson and colleagues independently confirmed this margin on the same data [Johnson et al., 2017]. The XGBoost model evaluated in Chapter 3 is the direct descendant of these tree-ensemble baselines.

### 1.2.2 Recurrent neural networks

Lipton and colleagues' seminal paper on diagnosing diseases from clinical time series [Lipton et al., 2015] demonstrated that LSTM networks trained end-to-end on raw multivariate observations match or exceed feature-engineered baselines on paediatric ICU data. Choi and colleagues' RETAIN architecture [Choi et al., 2016] introduced a two-level attention mechanism explicitly designed to make recurrent models interpretable for clinical use. Che and colleagues' GRU-D [Che et al., 2018] modified the GRU cell to incorporate two mechanisms for handling missingness: a binary mask on whether each variable was observed at each time-step and a decay function that interpolates between the last observation and the empirical mean as time without observation grows. GRU-D demonstrated, on MIMIC-III, that the encoding of *when* measurements occur is a first-class signal — a finding the ablation study of Chapter 3 confirms on PhysioNet 2012.

### 1.2.3 Temporal convolutional networks

Bai, Kolter, and Koltun's 2018 paper [Bai et al., 2018] argued that dilated causal convolutional networks (TCNs) match recurrent architectures on a wide range of sequence-modelling tasks while training faster and producing more stable gradients. The TCN baseline in Chapter 3 follows the canonical residual block structure of that paper, with dilations 1, 2, 4, 8 stacked over a 48-hour input window producing a 31-hour effective receptive field at the output layer.

### 1.2.4 Transformer-based architectures

The Transformer [Vaswani et al., 2017] dispenses with recurrence and convolution entirely, replacing them with multi-head self-attention. For irregularly sampled or fixed-grid clinical time series, two principal adaptations have appeared. Song and colleagues' SAnD [Song et al., 2018] applies a Transformer encoder to dense MIMIC-III observations and obtains improvements over RETAIN on multiple downstream tasks. Tipirneni and Reddy's STraTS [Tipirneni & Reddy, 2022] handles sparsity by representing each observation as a triplet of (variable, time, value) and applying continuous-value embedding rather than positional encoding. BEHRT [Li et al., 2020] and Med-BERT [Rasmy et al., 2021] extend the BERT recipe to longitudinal electronic-health-record sequences for diagnosis prediction, demonstrating that Transformer pre-training transfers usefully to clinical tasks.

The model evaluated in this thesis is a pre-norm Transformer encoder with four layers, four attention heads, model dimension 64, feed-forward dimension 256, and a *learnable sinusoidal positional encoding* that we refer to as Time-Aware Positional Encoding. The frequencies of the sinusoids are learnable parameters, rather than the fixed geometric series of the original Transformer formulation, on the hypothesis that the optimal time scale for ICU physiology differs from that of natural language and should be discovered by gradient descent. A CLS token in the BERT style summarises each patient's encoded trajectory for the final classification head.

### 1.2.5 The PhysioNet 2012 corpus

The PhysioNet/Computing in Cardiology Challenge 2012 [Silva et al., 2012] released 8000 ICU records (Set-A, Set-B, Set-C of 4000 each), of which Set-A is publicly labelled. Each record covers the first 48 hours after ICU admission and contains a heterogeneous set of approximately 37 physiological and laboratory variables along with six admission descriptors (age, gender, height, ICU type, weight, time). The Challenge defines two evaluation events: Event 1 maximises Score1 = min(Sensitivity, +Predictivity) over the choice of operating threshold, and Event 2 evaluates calibration via the Hosmer–Lemeshow statistic. The official SAPS-I baseline on this cohort achieves Score1 = 0.3097 and HL-H = 35.21. The corpus has been a standard benchmark for ICU mortality models for over a decade; published deep-learning results on Set-A typically report AUROC in the 0.83–0.86 range [Harutyunyan et al., 2019; Sheikhalishahi et al., 2020].

## 1.3 Explainable artificial intelligence

### 1.3.1 Why explanation matters in critical care

The argument for explainability in clinical artificial intelligence has been made in two distinct vocabularies. One vocabulary, exemplified by the European Union's General Data Protection Regulation (Recital 71) and the United States Food and Drug Administration's Software-as-a-Medical-Device guidance, is regulatory: a deployed clinical model must furnish a meaningful explanation of its outputs. The other vocabulary, exemplified by the AI-in-medicine literature [Tonekaboni et al., 2019; Holzinger et al., 2019], is epistemic: a physician on a ward round cannot be expected to defer to a black-box probability against their own diagnostic intuition, and an opaque model that conflicts with that intuition is functionally a model that goes unused.

### 1.3.2 LIME

Ribeiro, Singh, and Guestrin's Local Interpretable Model-Agnostic Explanations (LIME) [Ribeiro et al., 2016] explains the prediction of an arbitrary classifier at a given input by fitting a locally weighted linear surrogate on a synthetic perturbation neighbourhood. LIME's strength is its agnosticism: it requires only black-box access to the model. Its weaknesses, which have been widely catalogued, are instability under perturbation sampling, sensitivity to the choice of kernel width, and the absence of a coherent additivity principle that would allow attributions across inputs to be compared.

### 1.3.3 SHAP

Lundberg and Lee's SHapley Additive exPlanations (SHAP) [Lundberg & Lee, 2017] extend the Shapley value from cooperative game theory to feature attribution. The Shapley value is the unique allocation of the total "payout" of a coalitional game to its players that satisfies four axioms: efficiency, symmetry, dummy-player, and additivity. Translated to machine learning, the players are input features, the payout is the difference between the model's prediction and its average prediction, and the Shapley value of feature *i* is the average marginal contribution of *i* over all permutations of the other features. SHAP inherits three properties from this construction:

1. *Local accuracy:* the explanation sums to the model's deviation from baseline.
2. *Missingness:* features absent from the input receive zero attribution.
3. *Consistency:* if a model is changed so that a feature contributes more, that feature's attribution does not decrease.

The TreeSHAP algorithm [Lundberg et al., 2020] computes exact Shapley values for tree ensembles in polynomial time, which makes SHAP applicable at production scale for gradient-boosted tree models like the XGBoost classifier evaluated in this thesis. For deep networks, DeepSHAP combines DeepLIFT [Shrikumar et al., 2017] with the SHAP framework to provide approximate but theoretically grounded attributions.

### 1.3.4 SHAP for time-series data

A subtle but important limitation of the standard SHAP estimator is that its output lives in the same coordinate system as its input. When the input is a flat feature vector — for example, the 228-dimensional vector of summary statistics used by the XGBoost model — SHAP produces one number per summary statistic. "Last-observed heart rate" gets one attribution; "mean heart rate" gets a separate attribution. Neither attribution tells the clinician *when* heart rate mattered. The temporal axis has been compressed out of the feature representation and SHAP cannot recover it from a model that never saw it.

Two families of solutions have appeared. The first is to run SHAP on a model that ingests the raw 3D tensor — typically a Transformer or convolutional network — using KernelSHAP or DeepSHAP. This is theoretically clean but computationally prohibitive at the patient-cohort scale (KernelSHAP runtime scales with the product of the number of features and the perturbation budget). The second, which is the approach of this thesis, is to retain the computational efficiency of TreeSHAP on a flat-feature tree model while *redistributing* the variable-level attribution back onto the temporal axis using a deterministic rule derived from the raw trajectory. The construction is detailed in Section 2.6.

### 1.3.5 Attention as explanation

A separate strand of the interpretability literature argues that attention weights from Transformer or attention-augmented recurrent models are themselves an explanation. The position is contested — Jain and Wallace [Jain & Wallace, 2019] showed that attention weights are not consistent across runs and cannot in general be interpreted as faithful explanations, while Wiegreffe and Pinter [Wiegreffe & Pinter, 2019] argued that under certain conditions attention can serve as a plausible explanation. The thesis is agnostic on the philosophical question; Section 3.5 reports attention head specialisation patterns from the trained Transformer as a *descriptive* finding, not as a primary explanation method.

### 1.3.6 The clinical-validation gap

Two practical observations about the SHAP-for-ICU literature motivated the validation experiment of Section 2.7. First, the overwhelming majority of papers present SHAP attributions as illustrative figures — "the model emphasised lactate, which is clinically sensible" — without any quantitative comparison to an established clinical yardstick. Second, when quantitative comparisons appear, they are typically global (mean absolute attribution of variable X across the cohort) rather than per-patient (does the model agree with SOFA *for this patient*). The thesis addresses both gaps by defining a per-patient concordance score against SOFA and SAPS-I component variables and testing the statistical relationship between concordance and calibration error.

## 1.4 Synthesis and position of this work

The literature reviewed above suggests four conclusions that frame the rest of the thesis.

1. Classical severity scores remain the regulatory and clinical *baseline* but are limited by their assumption that the first twenty-four hours can be summarised by a worst-value vector. Modern data make this assumption unnecessary.

2. Deep learning on raw multivariate ICU trajectories has demonstrably superior discrimination on benchmarks like MIMIC-III and PhysioNet 2012, but the magnitude of the gain depends critically on whether the model is given access to missingness information.

3. SHAP is the most theoretically grounded post-hoc explanation method currently available, but its standard formulation loses the temporal axis on time-series data. Recovering that axis without paying the computational cost of DeepSHAP on raw 3D tensors is an open methodological problem.

4. The clinical validation of post-hoc explanations is under-developed. The dominant evaluation in published work is visual plausibility, which is insufficient grounds for clinical deployment.

This thesis contributes to (2) by quantifying the effect of explicit missingness encoding on PhysioNet 2012, contributes to (3) by developing the time-step SHAP redistribution procedure, and contributes to (4) by introducing the SOFA / SAPS-I concordance test.
