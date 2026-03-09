# Disease Prediction Modeling Pipeline using EHR Data

An end-to-end machine learning pipeline for **disease prediction from Electronic Health Records (EHRs)**, using **superimposed preeclampsia (PE) in chronic hypertension pregnancies** as a case study.

> **Note:** This repository uses a **synthetic dataset** generated to mirror the structure of a real clinical cohort. No real patient data is included. Hospital names and identifiable information are de-identified.

---

## Research Project: Superimposed Preeclampsia Prediction

Superimposed PE — the development of preeclampsia in women with pre-existing chronic hypertension — is one of the most challenging hypertensive disorders of pregnancy to diagnose. This pipeline models early identification of superimposed PE from routinely collected EHR variables.

| Attribute | Detail |
|-----------|--------|
| **Index condition** | Chronic hypertension (ICD-10: O10, I10) |
| **Outcome** | Superimposed preeclampsia (ICD-10: O11) |
| **Study period** | June 2018 – July 2021 |
| **Age range** | 12–50 years at delivery |
| **Data source** | Synthetic EHR (de-identified structure) |

### ICD-10-CM Cohort Definitions

| Cohort | Codes |
|--------|-------|
| Chronic hypertension | O10, I10 |
| Superimposed PE | O11, O11.1–O11.9 |
| Eclampsia | O15 |
| Gestational hypertension | O13 |
| Preeclampsia | O14.0–O14.9 |

---

## Models Implemented

### Classical Machine Learning
| Model | Description |
|-------|-------------|
| Logistic Regression | L1, L2, ElasticNet regularisation |
| SVM | RBF, Linear, Polynomial kernels (probability calibrated) |
| Random Forest | Bagging ensemble of decision trees |
| XGBoost | Gradient boosted trees (Chen & Guestrin, 2016) |
| LightGBM | Histogram-based gradient boosting (Ke et al., 2017) |
| CatBoost | Ordered boosting for categorical features |
| Decision Tree | Interpretable single-tree baseline |
| K-Nearest Neighbours | Distance-weighted voting |
| Naïve Bayes | Gaussian generative model |
| LDA | Linear Discriminant Analysis |

### Deep Learning
| Model | Description |
|-------|-------------|
| MLP | Multi-Layer Perceptron with batch norm, dropout, residual connections |
| TabNet | Attentive feature selection (Arik & Pfister, NeurIPS 2021) |
| FT-Transformer | Feature Tokenizer + Transformer (Gorishniy et al., NeurIPS 2021) |
| LSTM | Bidirectional LSTM for sequential/longitudinal EHR |
| 1D-CNN | 1D Convolutional network over feature vectors |

### Ensemble
| Strategy | Description |
|----------|-------------|
| Soft Voting | Average of class probabilities |
| Hard Voting | Majority class vote |
| Stacking | OOF meta-learning with a holdout meta-learner |
| Blending | Holdout-set meta-learning |
| Bayesian Model Averaging | Evidence-weighted model combination |
| Rank Averaging | Robust rank-transform averaging |

---

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **AUROC** | Area under the ROC curve |
| **AUPRC** | Area under the Precision-Recall curve |
| **Sensitivity** | True positive rate (recall) |
| **Specificity** | True negative rate |
| **PPV** | Positive predictive value (precision) |
| **NPV** | Negative predictive value |
| **F1** | Harmonic mean of precision and recall |
| **Brier Score** | Probabilistic calibration |
| **MCC** | Matthews Correlation Coefficient |
| **ECE** | Expected Calibration Error |
| **95% CI** | Bootstrap confidence intervals (n=1000) |
| **DeLong test** | Statistical comparison of AUROC pairs |

Statistical comparisons: **Kruskal-Wallis** (continuous), **χ² test** (categorical), significance threshold p < 0.05.


---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/your-username/disease-prediction-modeling.git
cd disease-prediction-modeling
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
python data/synthetic/generate_synthetic_data.py
# Options:
python data/synthetic/generate_synthetic_data.py --n 3000 --seed 99
```

### 3. Run Full Pipeline

```bash
# Step 1: Preprocess
python pipelines/01_data_preprocessing.py

# Step 2: Classical ML
python pipelines/02_classical_ml.py
python pipelines/02_classical_ml.py --tune        # with hyperparameter search

# Step 3: Deep Learning
python pipelines/03_deep_learning.py
python pipelines/03_deep_learning.py --models mlp tabnet --epochs 100

# Step 4: Ensembles
python pipelines/04_ensemble_models.py

# Step 5: Final comparison + SHAP
python pipelines/05_model_comparison.py --shap-model random_forest
```

### 4. Explore Notebooks

```bash
jupyter lab notebooks/
```

---

## Input Features

### Demographics
- Age at delivery, race/ethnicity, insurance type
- Parity, gravidity, multiple gestation, gestational age

### Clinical History
- Chronic hypertension duration (years)
- Pre-gestational diabetes, gestational diabetes
- Renal disease, autoimmune disorder
- Prior preeclampsia, prior preterm birth
- Pre-pregnancy BMI, smoking status

### Vital Signs
- Systolic/diastolic BP (1st trimester, 2nd trimester, max antepartum)

### Laboratory Values
- 24-hour urine protein (mg)
- Creatinine (mg/dL)
- Platelet count (K/µL)
- ALT, AST (U/L)
- Uric acid (mg/dL)
- Hemoglobin (g/dL)
- LDH (U/L)

### Derived Clinical Features (auto-generated)
- Pulse pressure, mean arterial pressure
- BP severity category (normal / mild / severe HTN)
- Proteinuria flag (≥300 mg), severe proteinuria flag (≥2000 mg)
- HELLP risk score (platelet + AST + LDH composite)
- Renal function score (creatinine + uric acid)
- Liver enzyme ratio (AST/ALT)
- Metabolic risk score (BMI + diabetes)

---

## Configuration

All settings are controlled via `config/config.yaml`:

```yaml
preprocessing:
  imputation_strategy: median    # median | mean | knn | iterative
  scaling: standard              # standard | minmax | robust | none
  encoding_method: onehot        # onehot | ordinal | target
  outlier_method: iqr            # iqr | zscore | none

deep_learning:
  max_epochs: 200
  patience: 20
  batch_size: 64
  dropout_rate: 0.3

evaluation:
  bootstrap_ci:
    n_iterations: 1000
    confidence_level: 0.95
```

---

## Extending to Other Diseases

This pipeline is disease-agnostic. To adapt for a different prediction task:

1. **Replace data**: Modify `data/synthetic/generate_synthetic_data.py` to match your cohort features and outcome.
2. **Update config**: Set `target_column` and feature lists in `config/config.yaml`.
3. **Run the same pipeline**: Steps 01–05 work without code changes.

Example adaptations:
- Gestational diabetes prediction
- Preterm birth risk scoring
- Sepsis early warning
- Hospital readmission prediction
- Cancer risk stratification from EHR

---

## Statistical Methods

| Analysis | Method |
|----------|--------|
| Continuous variable comparison | Kruskal-Wallis H-test |
| Categorical variable comparison | χ² test |
| Feature selection | Mutual information, ANOVA-F, Spearman correlation |
| Hyperparameter tuning | Randomised / Grid search with 5-fold CV |
| Confidence intervals | Bootstrap (n=1,000, 95% CI) |
| Model comparison | DeLong test for correlated ROC curves |
| Calibration | Hosmer-Lemeshow test, reliability diagram, ECE |
| Interpretability | SHAP (TreeExplainer / KernelExplainer), LIME, permutation importance |

---

## References

| Method | Citation |
|--------|----------|
| XGBoost | Chen & Guestrin, KDD 2016 |
| LightGBM | Ke et al., NeurIPS 2017 |
| CatBoost | Prokhorenkova et al., NeurIPS 2018 |
| TabNet | Arik & Pfister, AAAI 2021 |
| FT-Transformer | Gorishniy et al., NeurIPS 2021 |
| SHAP | Lundberg & Lee, NeurIPS 2017 |
| DeLong test | DeLong et al., Biometrics 1988 |
| SMOTE | Chawla et al., JAIR 2002 |

---

## License

MIT License — see `LICENSE` for details.

> **Privacy Notice:** This repository contains only synthetic data. Do not commit real patient EHR data to any public repository.
