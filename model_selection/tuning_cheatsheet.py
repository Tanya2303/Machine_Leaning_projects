"""Callable CV tuning helpers and short examples. Always pass a training-only Pipeline."""
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def grid_search_cv(pipeline, param_grid, X_train, y_train, *, scoring, cv=5, n_jobs=-1):
    """Exhaustively compare a small grid; returns a refitted search object."""
    search=GridSearchCV(pipeline,param_grid=param_grid,scoring=scoring,cv=cv,n_jobs=n_jobs,refit=True)
    return search.fit(X_train,y_train)

def randomized_search_cv(pipeline, param_distributions, X_train, y_train, *, scoring, n_iter=30, cv=5, random_state=42, n_jobs=-1):
    """Sample a broad space efficiently; returns a refitted search object."""
    search=RandomizedSearchCV(pipeline,param_distributions=param_distributions,n_iter=n_iter,
        scoring=scoring,cv=cv,random_state=random_state,n_jobs=n_jobs,refit=True)
    return search.fit(X_train,y_train)

# Example:
# search=grid_search_cv(pipeline, {'model__C':[0.1, 1, 10]}, X_train, y_train,
#                       scoring='f1_weighted')
# print(search.best_params_, search.best_score_)
# Use search.best_estimator_ on the held-out test set exactly once.
