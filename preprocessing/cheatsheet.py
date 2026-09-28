"""Reusable sklearn preprocessing building blocks; fit only inside train/CV pipelines."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, mutual_info_regression

def tabular_preprocessor(numeric_columns, categorical_columns, scaler='standard'):
    scalers={'standard':StandardScaler,'minmax':MinMaxScaler,'robust':RobustScaler,None:None}
    scale=scalers[scaler]() if scaler else 'passthrough'
    numeric=Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',scale)])
    categorical=Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))])
    return ColumnTransformer([('numeric',numeric,numeric_columns),('categorical',categorical,categorical_columns)])

# Feature selection example: put SelectKBest(mutual_info_classif, k=...) in a Pipeline
# after the preprocessor. Use mutual_info_regression for regression targets.

# Polynomial feature engineering example (place in a Pipeline to avoid leakage):
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.linear_model import Ridge
# Pipeline([('preprocess', tabular_preprocessor(nums, cats)),
#           ('poly', PolynomialFeatures(degree=2, include_bias=False)), ('model', Ridge())])
# Feature selection example, also inside the Pipeline:
# Pipeline([('preprocess', tabular_preprocessor(nums, cats)),
#           ('select', SelectKBest(mutual_info_regression, k=10)), ('model', Ridge())])

def feature_selector(task='classification', k=10):
    """Create task-appropriate univariate selection for a leak-safe Pipeline."""
    if task == 'classification':
        return SelectKBest(score_func=mutual_info_classif, k=k)
    if task == 'regression':
        return SelectKBest(score_func=mutual_info_regression, k=k)
    raise ValueError("task must be 'classification' or 'regression'")
