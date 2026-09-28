# Heart Disease

End-to-end binary classification using the supplied dataset `datasets/heart_failure_clinical_records_dataset.csv`.

- Target: `DEATH_EVENT`
- Detected schema, missingness, descriptive statistics, and target distribution: `eda_profile.json`; visual EDA: `eda_target_distribution.png` and `eda_correlation_heatmap.png`.
- Candidate model scores: `model_comparison.csv`; randomized tuning results: `tuning_results.json`.
- Final fitted preprocessing and estimator pipeline: `model_pipeline.joblib`.
- Actual final selection rationale and held-out metrics: `run_summary.json`.
- Held-out permutation feature importance: `feature_importance.csv`; numeric correlations: `numeric_correlations.csv`.

Run from the repository root with `python model_selection/run_experiments.py`. Five-fold CV is computed on the training split, and candidates share one untouched test split. The test set is reserved for final reporting.

## Selection rationale

The tuned Gradient Boosting classifier had the strongest weighted-F1 CV result in the experiment (0.8598). It is within one standard error of the strongest alternatives and the chosen tuned model’s held-out weighted F1 is 0.793 (accuracy 0.800, ROC-AUC 0.833). The dataset is small (299 rows), so the holdout estimate is uncertain. The target measures follow-up mortality, not heart disease diagnosis.
