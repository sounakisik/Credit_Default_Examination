# Independent Model Validation & Challenger Framework

This project mimics an independent Model Risk Management (MRM) validation of a
binary credit-default model.

## Business question
Can a credit-default model be approved for use, and what risks, limitations,
monitoring requirements, and challenger-model evidence should be documented
before approval?

## Dataset
UCI "Default of Credit Card Clients" dataset:
- 30,000 observations
- 23 explanatory variables
- binary default outcome

## Validation architecture
1. Data-quality and leakage audit
2. Champion model: Logistic Regression
3. Challenger model: XGBoost
4. Discrimination: ROC-AUC, PR-AUC, KS/Gini
5. Calibration: calibration curve, Brier score, calibration intercept/slope
6. Uncertainty: bootstrap confidence intervals
7. Stability: PSI and feature-shift diagnostics
8. Stress testing: controlled synthetic distribution shift
9. Explainability: coefficients/odds ratios and SHAP
10. Sensitivity and reasonableness checks
11. Model-risk recommendation: Approve / Approve with Conditions / Reject

## Run
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Start with `notebooks/01_data_audit.ipynb`.
