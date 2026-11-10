# Astana Apartment Price Prediction — Endterm Project

Machine Learning Algorithms | Group SE-2431  
Team: Nurkhan Kuanyshov and Bakytzhan Jangazy

## Overview

This project estimates Astana apartment asking prices from listing size, layout, district, floor, building, coordinate, and amenity information. The task is regression: the original target is `price`, while models learn `log1p(price)`. Our research question is: can we predict the asking price of an apartment in Astana using its area, location, rooms, floor, building characteristics and amenities?

It extends the preserved midterm work. Asking-price predictions are decision support, not official valuations.

## Team contributions

| Member | Technical responsibility |
| --- | --- |
| Nurkhan Kuanyshov | Dataset preparation, EDA, cleaning, feature engineering, notebook structure, documentation, clustering, and PCA. |
| Bakytzhan Jangazy | Supervised evaluation, cross-validation/tuning, ANN, error analysis, comparison, and presentation preparation. |

These are identifiable project sections, not claims about commits or finished results.

## Dataset

Source: [Astana Real Estate Dataset by turarr on Kaggle](https://www.kaggle.com/datasets/turarr/astana-real-estate-dataset). It contains Krisha.kz advertisements. The notebook derives actual size, columns, missingness and cleaned size from the supplied CSV; it does not hardcode them.

The dataset downloads automatically from Kaggle when the notebook runs. A local `astana_real_estate_dataset.csv` is optional and takes priority; in Kaggle notebooks, an attached input is used first. The source is a third-party Kaggle collection of Krisha.kz advertisements, not an official government dataset. See [DATA.md](DATA.md) for details. No synthetic data is generated.

## Deliverables

```text
astana_housing_midterm.ipynb              # preserved original work
notebooks/astana_housing_endterm.ipynb    # endterm notebook
scripts/build_endterm_notebook.py         # version-controlled notebook source
visualizations/{eda,clustering,pca,ann,evaluation}/
models/                                   # created by a successful run
reports/endterm_analysis.md
reports/defense_questions.md
data/README.md
```

## Endterm methods

- Cleaning: numeric conversion, floor extraction, duplicate removal, realistic ranges, target trimming, and identifier/leakage removal.
- Features: house age, floor ratio, first/top-floor flags, area per room, and coordinate-based distance to centre only when real coordinates exist.
- EDA: raw/log target distributions, area-price relation, district prices, correlations and missingness.
- Evaluation: fixed 70/15/15 train/validation/test split with log-price quantile stratification and training-only preprocessing pipelines.
- Regression: median baseline, Linear Regression, Decision Tree, KNN, Random Forest, Gradient Boosting, and a feedforward ANN.
- Clustering: scaled K-Means chosen from elbow inertia and silhouette evidence; price is exploratory only, never a supervised input.
- PCA: training-only standardized numerical PCA with scree, cumulative variance and 2D cluster plot.
- ANN: 128–64–32 ReLU layers, dropout, Adam/MSE, EarlyStopping, ReduceLROnPlateau and history plot.
- Error analysis: actual/predicted, residuals, price segments, districts, and largest errors.

## Installation and execution

The recorded run used Python 3.14.7 and TensorFlow 2.22.0-rc0. `requirements.txt` currently specifies minimum versions, not a full lockfile, so fresh installs may resolve differently. The notebook prints the runtime versions of Python, pandas, scikit-learn, NumPy, Matplotlib, seaborn, joblib and TensorFlow.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/build_endterm_notebook.py
python scripts/verify_endterm_notebook.py
```

A successful verification executes the notebook top-to-bottom, downloading the public dataset if needed, and creates visualization PNGs, `reports/model_comparison_validation.csv`, `reports/final_test_metrics.csv`, and `models/best_model.joblib` when a scikit-learn model is selected. The first download outside Kaggle requires internet access; KaggleHub caches the data outside the repository.

## Results

The endterm notebook was run from a clean kernel on the supplied Kaggle CSV. Gradient
Boosting was selected on validation RMSE(log) and achieved the following final
held-out test results: **RMSE(log) 0.1674**, **MAE(log) 0.1252**, **R2 0.8921**,
**MAE 6.11M KZT**, and **MedAPE 9.27%**. It improves considerably over the
median baseline's validation MedAPE of 33.93%. The ANN was evaluated fairly but
did not improve the traditional models (validation RMSE(log) 1.0203 versus
0.1618 for Gradient Boosting). Historical midterm metrics remain available in
the original notebook and are not substituted for endterm evidence.

## Limitations and ethics

Listing prices are not verified transactions. Advertisements can be duplicated, inaccurate, incomplete, and time-specific; premium homes can depend on unrecorded renovation, view or building details. Accuracy can vary by district and price segment. Feature importance is association, not causation, and outputs must not be treated as official appraisals.

## Defense

[reports/defense_questions.md](reports/defense_questions.md) contains 35 student-level questions and result-specific prompts for the recorded data run.
