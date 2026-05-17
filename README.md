# ICU XAI Research

**Explainable AI for ICU Patient Outcomes Prediction**

This repository implements explainable machine learning models for predicting critical patient outcomes in the ICU using SHAP-based time-step level interpretability.

## Project Structure

```
icu-xai/
├── data/
│   ├── raw/              # PhysioNet extracted files
│   └── processed/        # Cleaned dataframes
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_preprocessing.ipynb  # Data Cleaning & Feature Engineering
│   ├── 03_models.ipynb         # Model Training (XGBoost, TCN, Transformer)
│   └── 04_shap_analysis.ipynb  # SHAP-based Explainability
│
├── src/
│   ├── __init__.py       # Package initialization
│   ├── load_data.py      # Data loading functions
│   ├── preprocess.py     # Preprocessing pipeline
│   ├── models.py         # Model definitions
│   └── shap_utils.py     # SHAP utilities
│
├── outputs/
│   ├── figures/          # SHAP plots, ROC curves
│   └── models/           # Saved model weights
│
├── requirements.txt      # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Setup

### 1. Create Virtual Environment
```bash
python -m venv venv
source venv/Scripts/activate  # On Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure in VS Code
- Press `Ctrl + Shift + P`
- Type: `Python: Select Interpreter`
- Choose: `./venv/Scripts/python.exe`

## Usage

Execute notebooks in order:

1. **01_eda.ipynb** - Explore your ICU dataset
2. **02_preprocessing.ipynb** - Clean and engineer features
3. **03_models.ipynb** - Train XGBoost, TCN, and Transformer
4. **04_shap_analysis.ipynb** - Generate SHAP-based explanations

## Key Features

- **Multiple model architectures**: XGBoost, TCN, Transformer
- **Time-step level explainability**: Identify critical moments in patient trajectories
- **SHAP integration**: Feature importances per timestep
- **Evaluation metrics**: AUC-ROC, confusion matrices, feature importance plots

## Models

### XGBoost
Gradient boosting baseline for tabular feature-based predictions.

### Temporal Convolutional Network (TCN)
Captures temporal dependencies with dilated convolutions.

### Transformer
Attention-based architecture for sequence-to-outcome prediction.

## Explainability

SHAP (SHapley Additive exPlanations) values reveal:
- Which features are most important for each prediction
- How features evolve over time in patient trajectories
- Critical time windows for intervention

## References

- SHAP: [arxiv.org/abs/1705.07874](https://arxiv.org/abs/1705.07874)
- PhysioNet: [physionet.org](https://physionet.org)

## License

MIT License
