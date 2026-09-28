# Model selection workflow

`run_experiments.py` loads the supplied four datasets, writes EDA summaries, builds type-aware preprocessing pipelines, evaluates all relevant available candidates with five-fold CV, performs a fixed 80/20 split (seed 42), tunes the leading candidates with randomized CV search, then saves the selected fitted pipeline and reports holdout metrics. Classification selection uses weighted F1; regression selection uses CV RMSE. Every preprocessing operation is refit inside each fold. The test set is used only for final metrics and descriptive permutation importance, never for selection.

Run from the repository root: `python model_selection/run_experiments.py`.

`GridSearchCV` versus `RandomizedSearchCV` examples are in `tuning_cheatsheet.py`. `SelectKBest` examples (classification and regression scoring) are in `preprocessing/cheatsheet.py`. The loan pipeline calculates total assets and loan-to-income/assets ratios inside the saved pipeline, so web callers provide the original raw features.
