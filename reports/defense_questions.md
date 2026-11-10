# Defense questions and answers

These answers describe the implemented method. For questions about the winning
model or exact values, consult the generated validation and test tables from the
endterm run; do not quote the historical midterm figures as endterm results.

1. **What is regression?** Regression predicts a continuous numeric outcome. Here it estimates apartment price.
2. **Why this dataset?** It directly represents the Astana listing market and includes price plus property characteristics relevant to the research question.
3. **Why is `price` the target?** Asking price is the outcome that buyers and sellers want to estimate from a listing's characteristics.
4. **Why model `log1p(price)`?** Prices have a long upper tail. Logging makes the target less skewed and reduces the dominance of very expensive listings.
5. **What does `log1p` mean?** It calculates `log(1 + price)`, which is safe even if a zero occurred.
6. **What is data leakage?** Leakage is information unavailable at prediction time entering training, which creates unrealistically good metrics.
7. **How is leakage prevented?** Data are split before fitting, price-like columns are removed, and imputing/scaling/encoding occur inside training-fitted pipelines.
8. **Why three splits?** Training fits models, validation selects among them, and the untouched test split estimates final generalization.
9. **Why use quantile-bin stratification?** It keeps low-, middle-, and high-price listings represented across regression splits.
10. **What is cross-validation?** It repeatedly fits on different training folds and estimates performance on the held-out fold, using training data only.
11. **What is GridSearchCV or RandomizedSearchCV?** They evaluate hyperparameter combinations through cross-validation and select the best CV score.
12. **How does Linear Regression work?** It learns a weighted linear combination of processed features that minimizes squared error.
13. **How does a Decision Tree work?** It repeatedly splits feature values into groups with more similar target prices.
14. **How does KNN work?** It predicts from nearby processed listings; scaling is important because it uses distances.
15. **How does Random Forest work?** It averages many randomized decision trees, reducing the instability of one tree.
16. **What is Gradient Boosting?** It builds shallow trees sequentially, with each new tree correcting part of previous errors.
17. **What is clustering?** It groups observations by feature similarity without a target prediction objective.
18. **Why may clustering use price here?** It is exploratory market segmentation, not a feature used to predict price in the supervised model.
19. **How does K-Means work?** It assigns listings to the nearest centroid, then updates centroids until assignments stabilize.
20. **What is the Elbow Method?** It plots within-cluster inertia by k and looks for diminishing improvement as clusters are added.
21. **What is a silhouette score?** It measures whether a listing is closer to its own cluster than to neighboring clusters; higher is generally better.
22. **What is PCA?** Principal Component Analysis rotates correlated numeric features into new directions ordered by variance explained.
23. **What does explained variance mean?** It is the proportion of numerical-feature variation retained by a component or set of components.
24. **Why standardize before K-Means and PCA?** Otherwise features with larger numerical units, such as price, dominate distance and variance calculations.
25. **What is an ANN?** A feedforward neural network learns nonlinear transformations through weighted layers of neurons.
26. **What is ReLU?** `max(0, x)`; it gives the hidden ANN layers a simple nonlinearity.
27. **Why is the output activation linear?** Price on the log scale is continuous and should not be restricted to a fixed range.
28. **What is Adam?** An adaptive gradient-based optimizer used to update ANN weights efficiently.
29. **What is overfitting?** It is when training performance keeps improving while validation performance becomes worse.
30. **What does EarlyStopping do?** It stops ANN training when validation loss does not improve and restores the best validation weights.
31. **Why might ANN perform worse than a tree model or Linear Regression?** Tabular datasets of this size can favor simpler models; ANN performance depends on architecture, regularization and representation.
32. **Which model is best?** The notebook selects the lowest validation log-RMSE model, then reports that selected model once on test data.
33. **Why are feature importances not causal effects?** They show how a fitted model used predictive associations, not what would happen if one property characteristic changed in real life.
34. **What are the main dataset limitations?** Asking rather than sale prices, potential duplicates, seller-entry errors, missing attributes, and a time-specific sample.
35. **What is the practical value?** It provides a transparent comparison estimate for listings, with uncertainty and responsible-use limits clearly stated.

## Result-specific questions to prepare from the endterm run

- Which model had the lowest validation RMSE, and what was its final test RMSE, R², MAE in KZT, and MedAPE?
- How much did it improve on the median baseline using the same metric?
- What k did K-Means select, what was its silhouette score, and how do cluster median prices differ?
- How much variance do the first two PCA components explain, and how many retain 90% and 95%?
- At what epoch did the ANN stop, and did its training versus validation loss indicate overfitting?
- Which price segment and which adequately represented district had the greatest median percentage error?

## Actual result flash cards

- The selected model is **Gradient Boosting**, chosen by validation RMSE(log) = **0.1624**.
- Its final test RMSE(log) is **0.1672**, R² is **0.8924**, MAE is **6.10M KZT**, and MedAPE is **9.18%**.
- The median baseline validation MedAPE is **33.93%**; the selected model's validation MedAPE is **9.42%**.
- The ANN did not win: validation RMSE(log) **0.1708** and MedAPE **9.70%**. Explain that neural networks are not automatically best for moderate-sized tabular data.
- K-Means selected **5 clusters** with silhouette **0.277**; the large-apartment cluster had median price about **78.0M KZT** and mean area about **118 m²**.
- PCA's first two components explain **33.8%** of numerical-feature variance; **9** and **11** components retain 90% and 95%.
- The run used Python **3.14.10** and TensorFlow **2.22.0-rc0**.
