# Master's Thesis — Explainable ML for ICU Mortality Prediction

**Author:** Ihsan Naqvi
**University:** National Research Tomsk State University (TSU)
**Year:** 2026

## Contents

| File | Section |
|:-----|:--------|
| `00_titlepage.md` | Title page |
| `01_abstract.md` | Abstract & keywords |
| `02_introduction.md` | Introduction (background, problem, RQs, aim, novelty, structure) |
| `03_chapter1_literature.md` | Chapter 1 — Literature review |
| `04_chapter2_methodology.md` | Chapter 2 — Methodology and data |
| `05_chapter3_results.md` | Chapter 3 — Results, findings, discussion |
| `06_conclusion.md` | Conclusion |
| `07_references.md` | References (45 entries) |
| `08_appendices.md` | Appendices A–E |

## Compiling to a single document

To produce a single Word or PDF file from the markdown sources, run (Pandoc required):

```powershell
# PDF (requires LaTeX)
pandoc 00_titlepage.md 01_abstract.md 02_introduction.md `
       03_chapter1_literature.md 04_chapter2_methodology.md `
       05_chapter3_results.md 06_conclusion.md 07_references.md `
       08_appendices.md -o thesis.pdf --toc --toc-depth=2

# Word
pandoc 00_titlepage.md 01_abstract.md 02_introduction.md `
       03_chapter1_literature.md 04_chapter2_methodology.md `
       05_chapter3_results.md 06_conclusion.md 07_references.md `
       08_appendices.md -o thesis.docx --toc --toc-depth=2
```

## Results pack

All headline numbers in the thesis are reproducible from JSON result files in `d:/icu-xai/outputs/`:

- XGBoost: AUROC = 0.8798, Score1 = 0.5422, HL-H = 37.67
- Transformer: AUROC = 0.8542, Score1 = 0.5301, HL-H = 165.17
- TCN: AUROC = 0.8191, Score1 = 0.4253, HL-H = 205.78
- SAPS-I baseline: Score1 = 0.3097, HL-H = 35.21
- Ablation: removing missingness drops Score1 by 6.6 pp (largest single-component effect)
- Clinical validation: SHAP–SAPS-I concordance mean = 0.498, ρ vs |error| = 0.139, p = 6.3 × 10⁻⁴

## Figure pack

Seven 300-DPI figures in `d:/icu-xai/outputs/figures/thesis/`. See Appendix D.
