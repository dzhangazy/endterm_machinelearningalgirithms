# Endterm analysis record

The final analysis is produced dynamically by `notebooks/astana_housing_endterm.ipynb`.
After a successful clean run it writes:

- `reports/model_comparison_validation.csv`
- `reports/final_test_metrics.csv`
- figures under `visualizations/`
- `models/best_model.joblib` when the selected model is a scikit-learn model

## Methodological record

- Target: `log1p(price)`; original-price metrics are back-transformed.
- Split: fixed-seed 70/15/15 train/validation/test split using log-price quantile bins.
- Selection: five-fold cross-validation and validation metrics use training and validation data only.
- Test: used once only after selecting the model.
- Advanced models: Random Forest and Gradient Boosting with bounded randomized searches.
- Additional endterm work: standardized K-Means segmentation, numerical PCA, and an early-stopped feedforward ANN.

## Results from clean notebook execution

The notebook was executed from a clean kernel with the official CSV on
2026-11-10. Gradient Boosting had the best validation log-RMSE (0.1618) and
was therefore selected before the test set was touched. Its final held-out test
results were: RMSE(log) **0.1674**, MAE(log) **0.1252**, R2 **0.8921**, MAE
**6,107,228 KZT**, and MedAPE **9.27%**.

The ANN used the same validation partition and did not improve the result:
its validation RMSE(log) was 1.0203 and MedAPE was 61.90%. This is an observed
outcome, not a reason to change splits or selectively remove the model.

The authoritative machine-readable records are
`model_comparison_validation.csv` and `final_test_metrics.csv` in this folder.
