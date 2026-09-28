# House Prices

End-to-end regression using the supplied dataset `datasets/housing.csv`.

- Target: `MEDV`
- Detected schema, missingness, descriptive statistics, and target distribution: `eda_profile.json`; visual EDA: `eda_target_distribution.png` and `eda_correlation_heatmap.png`.
- Candidate model scores: `model_comparison.csv`; randomized tuning results: `tuning_results.json`.
- Final fitted preprocessing and estimator pipeline: `model_pipeline.joblib`.
- Actual final selection rationale and held-out metrics: `run_summary.json`.
- Held-out permutation feature importance: `feature_importance.csv`; numeric correlations: `numeric_correlations.csv`.

Run from the repository root with `python model_selection/run_experiments.py`. Five-fold CV is computed on the training split, and candidates share one untouched test split. The test set is reserved for final reporting.

## Selection rationale

Tuned CatBoost had the lowest five-fold CV RMSE among evaluated candidates (3.234); on the held-out set it achieved MAE 1.862, RMSE 3.017, and R² 0.876. `MEDV` is in $1000s. Boston Housing is an old educational dataset and should not be used for real property valuation or policy decisions.
