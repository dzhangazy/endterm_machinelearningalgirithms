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

The notebook was executed from a clean Python 3.14.10 kernel with the supplied
CSV on 2026-11-11. Gradient Boosting had the best validation log-RMSE (0.1624) and
was therefore selected before the test set was touched. Its final held-out test
results were: RMSE(log) **0.1672**, MAE(log) **0.1254**, R2 **0.8924**, MAE
**6,097,216 KZT**, and MedAPE **9.18%**. The validation MedAPE was **9.42%**.

K-Means selected **k = 5** with silhouette **0.277**. The highest-price group
was cluster 2 (median asking price 78.0M KZT; mean area 118.0 m²). PCA retained
33.8% of numerical-feature variance in its first two components; 9 and 11
components retained 90% and 95%, respectively.

The ANN used the same validation partition and did not improve the result:
it stopped after 33 epochs, with validation RMSE(log) **1.1019** and MedAPE
**64.44%**. This is an observed outcome, not a reason to change splits or
selectively remove the model.

Runtime versions: pandas 3.1.1, NumPy 2.5.4, scikit-learn 1.9.1,
Matplotlib 3.11.2, seaborn 0.13.2, joblib 1.6.0 and TensorFlow 2.22.0-rc0.

The authoritative machine-readable records are
`model_comparison_validation.csv` and `final_test_metrics.csv` in this folder.
