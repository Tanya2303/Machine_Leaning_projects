# Experimental model comparison and selections

Actual results on the supplied datasets. Each project uses one seeded 80/20 test split shared by every model, with five-fold CV on the training split. Classifiers use weighted F1; regressors use RMSE. The final choice applies a one-standard-error rule: among models close to the best CV score, prefer the simpler model, and use tuned settings when available for that estimator. Detailed rankings, test metrics, tuning settings, and confusion matrices are in each project folder.

| Project | Selected model | Selected CV result | Held-out test result |
|---|---|---|---|
| heart_disease | Gradient Boosting | weighted F1 0.8598 (baseline fold SD 0.0420) | Accuracy 0.800; F1 0.793; ROC-AUC 0.833 |
| house_prices | CatBoost | RMSE 3.234 (baseline fold SD 0.336) | MAE 1.862; RMSE 3.017; R² 0.876 |
| loan_approval | Decision Tree | weighted F1 0.9994 (baseline fold SD 0.0012) | Accuracy 1.000; F1 1.000; ROC-AUC 1.000 |
| student_performance | Ridge | RMSE 2.052 (baseline fold SD 0.437) | MAE 0.452; RMSE 1.804; R² 0.770 |

## Dataset notes

- Heart dataset (299 rows) predicts the supplied `DEATH_EVENT` follow-up outcome; it does not diagnose heart disease. The small positive class means recall and ROC-AUC may vary substantially across splits.
- The 506-row headerless `housing.csv` is Boston Housing. `MEDV` is a historical median value in thousands of dollars; this dataset has known fairness and population limitations and is for education only.
- Loan approval achieved unusually high scores on this split, including a perfect holdout for the selected Decision Tree. This is dataset-specific evidence and does not establish real-world lending performance.
- Student Performance selects regularized Ridge because its CV result is within one standard error of the best while remaining easier to explain and deploy. Its test performance is close to the tuned SVR.
- Feature importance is calculated after model selection for interpretation. It does not influence model selection.
