"""Metric cookbook. Choose metrics that match the task and its error costs."""
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score)
import numpy as np

def classification_metrics(y_true, y_pred, y_score=None):
    out={'accuracy':accuracy_score(y_true,y_pred), 'precision_weighted':precision_score(y_true,y_pred,average='weighted',zero_division=0),
         'recall_weighted':recall_score(y_true,y_pred,average='weighted',zero_division=0), 'f1_weighted':f1_score(y_true,y_pred,average='weighted',zero_division=0),
         'confusion_matrix':confusion_matrix(y_true,y_pred).tolist()}
    if y_score is not None: out['roc_auc']=roc_auc_score(y_true,y_score)
    return out

def regression_metrics(y_true,y_pred):
    mse=mean_squared_error(y_true,y_pred)
    return {'mae':mean_absolute_error(y_true,y_pred),'mse':mse,'rmse':np.sqrt(mse),'r2':r2_score(y_true,y_pred)}
