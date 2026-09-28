# Personal Machine Learning Library and Portfolio

A practical ML cheat sheet plus four end-to-end projects built from the supplied real-world datasets. The experiments determine each target type from its meaning and use a shared held-out test split per project. All preprocessing is fitted inside scikit-learn Pipelines, keeping imputation, encoding, and scaling inside each training fold.

## Repository map

- `datasets/` — the four original data files.
- `projects/heart_disease/` — heart failure mortality classification.
- `projects/house_prices/` — Boston Housing median-value regression.
- `projects/loan_approval/` — loan approval classification.
- `projects/student_performance/` — exam score regression.
- `preprocessing/` — reusable imputing, encoding, scaling, and feature selection examples.
- `model_selection/` — experiment runner, CV, randomized tuning, and results.
- `evaluation/` — reusable metric examples.
- `algorithms_from_scratch/` — NumPy learning implementations.

## Setup and run

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
python model_selection/run_experiments.py
```

The run evaluates every supported model with five-fold CV on the training split, compares them on one untouched 20% test split, tunes the top three by CV using `RandomizedSearchCV`, chooses from CV scores, and saves the winning fitted preprocessing + estimator pipeline as `projects/<project>/model_pipeline.joblib`. It writes model comparisons, tuning results, and run metadata beside each project. Seed is 42. Run from any current directory; paths are resolved from this file.

Optional XGBoost, LightGBM, and CatBoost models are automatically included when their packages are installed. The experiment summary records which were available. Avoid loading joblib files from untrusted sources. Boston Housing is an old, geographically and socially biased dataset with sensitive historical features; it is used here for learning, not real lending, valuation, or policy decisions. Heart data predicts the provided `DEATH_EVENT` outcome, not whether someone has heart disease.

## Quick lookup

- Logistic Regression / classifiers / regressors: `model_selection/run_experiments.py` (`model_zoo`).
- `StandardScaler`, `OneHotEncoder`, `SimpleImputer`, `ColumnTransformer`, `Pipeline`: `preprocessing/cheatsheet.py`.
- `cross_validate`, `RandomizedSearchCV`, train/test split: `model_selection/run_experiments.py`.
- Accuracy, precision, recall, F1, ROC-AUC, confusion matrix, MAE, MSE, RMSE, R²: `evaluation/metrics.py`.
- Linear/logistic regression, KNN, K-Means, decision tree, Naive Bayes, PCA, gradient descent from scratch: `algorithms_from_scratch/implementations.py`.

## Dataset and target map

| Project | Supplied file | Target | Problem |
|---|---|---|---|
| Heart disease | `heart_failure_clinical_records_dataset.csv` | `DEATH_EVENT` | Binary classification; mortality follow-up outcome |
| House prices | `housing.csv` | `MEDV` | Regression; Boston-area median owner-occupied value in $1000s |
| Loan approval | `loan_approval_dataset.csv` | `loan_status` | Binary classification |
| Student performance | `StudentPerformanceFactors.csv` | `Exam_Score` | Regression; numeric score prediction |

Run-specific row counts, missingness, comparisons, tuned settings, selected models, and actual test metrics are generated from the supplied files and saved in each project directory. No scores are hard-coded.

## Deployment path

Each final `model_pipeline.joblib` accepts a pandas DataFrame containing the same feature columns used during training (without the target; loan input also excludes `loan_id`). A future API can validate a JSON payload, call `pipeline.predict(...)` (and `predict_proba(...)` for classifiers), and serialize the response. Keep feature names/types and training metadata with the API.

## Algorithms covered

Classification: Logistic Regression, KNN, Decision Tree, Random Forest, SVM, Gaussian Naive Bayes, Gradient Boosting, AdaBoost, Extra Trees, HistGradientBoosting, XGBoost, LightGBM, CatBoost. Regression: Linear and Polynomial Regression, Ridge, Lasso, Elastic Net, Decision Tree, Random Forest, SVR, Gradient Boosting, AdaBoost, KNN, Extra Trees, HistGradientBoosting, XGBoost, LightGBM, CatBoost. XGBoost, LightGBM, and CatBoost are included in `requirements.txt` and run as part of the comparisons.

The final choice follows the one-standard-error rule on CV: choose the simpler model when the score difference is within one standard error, and prefer tuned settings when the same estimator is selected. The per-dataset actual comparison and selection rationale are in [`model_selection/model_comparison_summary.md`](model_selection/model_comparison_summary.md) and each project README.
