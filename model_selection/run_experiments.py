"""Run repeatable, leakage-safe model comparisons on the four supplied datasets."""
from __future__ import annotations
import json, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier,
    ExtraTreesRegressor, GradientBoostingClassifier, GradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier,
    HistGradientBoostingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.model_selection import (RandomizedSearchCV, cross_validate, train_test_split)
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures, FunctionTransformer
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets'
SEED=42

# Whitespace-separated UCI Boston Housing file supplied without a header.
HOUSING_COLS=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
PROJECTS={
 'heart_disease': {'file':'heart_failure_clinical_records_dataset.csv','target':'DEATH_EVENT','task':'classification','title':'Heart Failure Mortality Risk'},
 'house_prices': {'file':'housing.csv','target':'MEDV','task':'regression','title':'Boston Housing Price'},
 'loan_approval': {'file':'loan_approval_dataset.csv','target':'loan_status','task':'classification','title':'Loan Approval'},
 'student_performance': {'file':'StudentPerformanceFactors.csv','target':'Exam_Score','task':'regression','title':'Student Exam Score'},
}

def load_data(key):
 c=PROJECTS[key]; path=DATA/c['file']
 if key=='house_prices':
  df=pd.read_csv(path,sep=r'\s+',header=None,names=HOUSING_COLS)
 elif key=='loan_approval':
  df=pd.read_csv(path); df.columns=df.columns.str.strip()
  # Identifier has no predictive meaning and can encode row order.
  df=df.drop(columns=['loan_id'])
  df['loan_status']=df['loan_status'].astype(str).str.strip()
 else: df=pd.read_csv(path)
 return df

def split_xy(df, target):
 return df.drop(columns=[target]),df[target]

def engineer_features(X, project):
 """Deterministic, leakage-safe, tabular feature engineering."""
 X=X.copy()
 if project=='loan_approval':
  assets=X['residential_assets_value']+X['commercial_assets_value']+X['luxury_assets_value']+X['bank_asset_value']
  X['total_assets']=assets
  X['loan_to_income_ratio']=X['loan_amount']/X['income_annum'].replace(0,np.nan)
  X['loan_to_assets_ratio']=X['loan_amount']/assets.replace(0,np.nan)
 return X

def make_preprocessor(X, scale=True):
 nums=X.select_dtypes(include=np.number).columns.tolist()
 cats=[c for c in X.columns if c not in nums]
 num_steps=[('imputer',SimpleImputer(strategy='median'))]
 if scale: num_steps.append(('scaler',StandardScaler()))
 cat_steps=[('imputer',SimpleImputer(strategy='most_frequent')),
            ('onehot',OneHotEncoder(handle_unknown='ignore', sparse_output=False))]
 return ColumnTransformer([('numeric',Pipeline(num_steps),nums),('categorical',Pipeline(cat_steps),cats)], remainder='drop')

def model_zoo(task, n):
 if task=='classification':
  models={
   'Logistic Regression':LogisticRegression(max_iter=3000,class_weight='balanced'),
   'KNN':KNeighborsClassifier(), 'Decision Tree':DecisionTreeClassifier(class_weight='balanced',random_state=SEED),
   'Random Forest':RandomForestClassifier(n_estimators=250,class_weight='balanced',random_state=SEED,n_jobs=-1),
   'SVM':SVC(class_weight='balanced',probability=True,random_state=SEED), 'Naive Bayes':GaussianNB(),
   'Gradient Boosting':GradientBoostingClassifier(random_state=SEED),
   'AdaBoost':AdaBoostClassifier(random_state=SEED),
   'Extra Trees':ExtraTreesClassifier(n_estimators=250,class_weight='balanced',random_state=SEED,n_jobs=-1),
   'Hist Gradient Boosting':HistGradientBoostingClassifier(random_state=SEED),
  }
  try:
   from xgboost import XGBClassifier
   models['XGBoost']=XGBClassifier(n_estimators=250, max_depth=4, learning_rate=.05, eval_metric='logloss', random_state=SEED, n_jobs=-1)
  except ImportError: pass
  try:
   from lightgbm import LGBMClassifier
   models['LightGBM']=LGBMClassifier(n_estimators=250, learning_rate=.05, verbosity=-1, random_state=SEED, n_jobs=-1)
  except ImportError: pass
  try:
   from catboost import CatBoostClassifier
   models['CatBoost']=CatBoostClassifier(iterations=250, verbose=False, random_seed=SEED, allow_writing_files=False)
  except ImportError: pass
 else:
  models={
   'Linear Regression':LinearRegression(),
   'Polynomial Regression (degree 2)':Pipeline([('poly',PolynomialFeatures(degree=2,include_bias=False)),('reg',Ridge())]),
   'Ridge':Ridge(), 'Lasso':Lasso(max_iter=5000), 'Elastic Net':ElasticNet(max_iter=5000),
   'Decision Tree Regressor':DecisionTreeRegressor(random_state=SEED),
   'Random Forest Regressor':RandomForestRegressor(n_estimators=250,random_state=SEED,n_jobs=-1),
   'SVR':SVR(), 'Gradient Boosting':GradientBoostingRegressor(random_state=SEED),
   'AdaBoost':AdaBoostRegressor(random_state=SEED),
   'KNN Regressor':KNeighborsRegressor(),
   'Extra Trees':ExtraTreesRegressor(n_estimators=250,random_state=SEED,n_jobs=-1),
   'Hist Gradient Boosting':HistGradientBoostingRegressor(random_state=SEED),
  }
  try:
   from xgboost import XGBRegressor
   models['XGBoost']=XGBRegressor(n_estimators=250,max_depth=4,learning_rate=.05,random_state=SEED,n_jobs=-1)
  except ImportError: pass
  try:
   from lightgbm import LGBMRegressor
   models['LightGBM']=LGBMRegressor(n_estimators=250,learning_rate=.05,verbosity=-1,random_state=SEED,n_jobs=-1)
  except ImportError: pass
  try:
   from catboost import CatBoostRegressor
   models['CatBoost']=CatBoostRegressor(iterations=250,verbose=False,random_seed=SEED,allow_writing_files=False)
  except ImportError: pass
 return models

def save_eda_plots(df, X, y, task, out):
 """Write a target distribution and numeric correlation heatmap for EDA."""
 fig,ax=plt.subplots(figsize=(7,4))
 if task=='classification': y.value_counts().sort_index().plot.bar(ax=ax,color='#4c78a8'); ax.set_ylabel('Rows')
 else: ax.hist(y,bins=25,color='#4c78a8',edgecolor='white'); ax.set_ylabel('Rows')
 ax.set_title('Target distribution'); ax.set_xlabel(str(y.name)); fig.tight_layout(); fig.savefig(out/'eda_target_distribution.png',dpi=140); plt.close(fig)
 nums=engineer_features(X, out.name).select_dtypes(include=np.number)
 fig,ax=plt.subplots(figsize=(max(8,min(15,.55*len(nums.columns)+5)),max(6,min(13,.48*len(nums.columns)+4))))
 sns.heatmap(nums.corr(),cmap='vlag',center=0,ax=ax,xticklabels=True,yticklabels=True)
 ax.set_title('Numeric feature correlations'); fig.tight_layout(); fig.savefig(out/'eda_correlation_heatmap.png',dpi=140); plt.close(fig)

def score_holdout(pipe, X, y, task):
 pred=pipe.predict(X)
 if task=='classification':
  labels=np.unique(y); positive=labels[-1]
  result={'accuracy':accuracy_score(y,pred),'precision_weighted':precision_score(y,pred,average='weighted',zero_division=0),
   'recall_weighted':recall_score(y,pred,average='weighted',zero_division=0),'f1_weighted':f1_score(y,pred,average='weighted',zero_division=0),
   'confusion_matrix':confusion_matrix(y,pred).tolist()}
  if len(labels)==2:
   try: result['roc_auc']=roc_auc_score(y,pipe.predict_proba(X)[:,list(pipe.classes_).index(positive)])
   except Exception: pass
  return result
 mse=mean_squared_error(y,pred)
 return {'mae':mean_absolute_error(y,pred),'mse':mse,'rmse':float(np.sqrt(mse)),'r2':r2_score(y,pred)}

def run_one(key, cfg):
 out=ROOT/'projects'/key; out.mkdir(parents=True,exist_ok=True)
 df=load_data(key); target=cfg['target']; X,y=split_xy(df,target); task=cfg['task']
 label_mapping=None
 if task=='classification' and y.dtype == object:
  encoder=LabelEncoder(); y=pd.Series(encoder.fit_transform(y),index=y.index,name=y.name); label_mapping={str(i):str(label) for i,label in enumerate(encoder.classes_)}
 # EDA artifacts: schema, nulls, descriptive statistics, target shape, and numeric correlations.
 profile={'shape':list(df.shape),'dtypes':{k:str(v) for k,v in df.dtypes.items()},'missing_values':{k:int(v) for k,v in df.isna().sum().items()},'describe':df.describe(include='all').replace({np.nan:None}).to_dict(),'target':target,'target_counts':{str(k):int(v) for k,v in y.value_counts(dropna=False).items()} if task=='classification' else {'min':float(y.min()),'median':float(y.median()),'max':float(y.max()),'mean':float(y.mean())}}
 (out/'eda_profile.json').write_text(json.dumps(profile,indent=2,default=str))
 engineer_features(X,key).select_dtypes(include=np.number).corr().to_csv(out/'numeric_correlations.csv')
 save_eda_plots(df,X,y,task,out)
 stratify=y if task=='classification' and y.value_counts().min()>=2 else None
 Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=stratify)
 pre=make_preprocessor(engineer_features(Xtr,key))
 rows=[]; fitted={}
 scoring='f1_weighted' if task=='classification' else 'neg_root_mean_squared_error'
 for name,est in model_zoo(task,len(Xtr)).items():
  pipe=Pipeline([('feature_engineering',FunctionTransformer(engineer_features,kw_args={'project':key},validate=False)),('preprocess',clone(pre)),('model',est)])
  started=time.time()
  try:
   cv=cross_validate(pipe,Xtr,ytr,cv=5,scoring=scoring,n_jobs=1,error_score='raise')
   pipe.fit(Xtr,ytr); metrics=score_holdout(pipe,Xte,yte,task)
   row={'model':name,'cv_score_mean':float(np.mean(cv['test_score'])),'cv_score_std':float(np.std(cv['test_score'])),'fit_seconds':round(time.time()-started,2),**metrics}
   rows.append(row); fitted[name]=pipe
   print(key,name,round(row['cv_score_mean'],4),flush=True)
  except Exception as e:
   rows.append({'model':name,'error':f'{type(e).__name__}: {e}'})
 results=pd.DataFrame(rows)
 valid=results[results.get('cv_score_mean').notna()].copy()
 # Keep top three by train-fold CV score; tune only promising models using training rows.
 valid=valid.sort_values('cv_score_mean',ascending=False)
 tuned=[]
 for _,r in valid.head(3).iterrows():
  name=r['model']; base=fitted[name]
  # Broad but intentionally bounded distributions; no test-set information enters tuning.
  if task=='classification':
   params={'model__max_depth':[None,3,5,8,12],'model__min_samples_leaf':[1,2,4,8], 'model__n_estimators':[100,250,400]}
   if name=='Logistic Regression': params={'model__C':np.logspace(-3,2,12)}
   elif name=='KNN': params={'model__n_neighbors':[3,5,7,9,13,17,21], 'model__weights':['uniform','distance']}
   elif name=='SVM': params={'model__C':np.logspace(-2,2,10),'model__gamma':['scale','auto']}
   elif name=='Naive Bayes': params={'model__var_smoothing':np.logspace(-11,-7,9)}
   elif name=='Hist Gradient Boosting': params={'model__learning_rate':[.03,.05,.1,.2],'model__max_iter':[100,200,350],'model__max_leaf_nodes':[15,31,63]}
   elif name in ['XGBoost']: params={'model__learning_rate':[.03,.05,.1],'model__max_depth':[3,5,7],'model__n_estimators':[100,250,400],'model__subsample':[.8,1.0]}
   elif name=='LightGBM': params={'model__learning_rate':[.03,.05,.1],'model__max_depth':[-1,3,5,8],'model__n_estimators':[100,250,400],'model__num_leaves':[15,31,63]}
   elif name=='CatBoost': params={'model__learning_rate':[.03,.05,.1],'model__depth':[4,6,8],'model__iterations':[100,250,400]}
   elif name=='AdaBoost': params={'model__learning_rate':[.03,.05,.1,.2,1.0],'model__n_estimators':[50,100,200,350]}
   elif name=='Gradient Boosting': params={'model__learning_rate':[.03,.05,.1,.2],'model__max_depth':[2,3,5],'model__n_estimators':[100,200,350]}
   elif name in ['Decision Tree']: params={'model__max_depth':[None,3,5,8,12],'model__min_samples_leaf':[1,2,4,8]}
   else: params={'model__max_features':['sqrt',.5,1.0],'model__min_samples_leaf':[1,2,4],'model__n_estimators':[150,300,500]}
  else:
   params={'model__max_depth':[None,3,5,8,12],'model__min_samples_leaf':[1,2,4,8],'model__n_estimators':[100,250,400]}
   if name in ['Ridge','Lasso','Elastic Net']: params={'model__alpha':np.logspace(-3,2,15)}
   elif name=='Linear Regression': continue
   elif name=='KNN Regressor': params={'model__n_neighbors':[3,5,7,9,13,17,21],'model__weights':['uniform','distance']}
   elif name=='SVR': params={'model__C':np.logspace(-1,2,8),'model__epsilon':[.05,.1,.2,.5],'model__kernel':['rbf','linear']}
   elif name=='Hist Gradient Boosting': params={'model__learning_rate':[.03,.05,.1,.2],'model__max_iter':[100,200,350],'model__max_leaf_nodes':[15,31,63]}
   elif name in ['XGBoost']: params={'model__learning_rate':[.03,.05,.1],'model__max_depth':[3,5,7],'model__n_estimators':[100,250,400],'model__subsample':[.8,1.0]}
   elif name=='LightGBM': params={'model__learning_rate':[.03,.05,.1],'model__max_depth':[-1,3,5,8],'model__n_estimators':[100,250,400],'model__num_leaves':[15,31,63]}
   elif name=='CatBoost': params={'model__learning_rate':[.03,.05,.1],'model__depth':[4,6,8],'model__iterations':[100,250,400]}
   elif name=='AdaBoost': params={'model__learning_rate':[.03,.05,.1,.2,1.0],'model__n_estimators':[50,100,200,350]}
   elif name=='Gradient Boosting': params={'model__learning_rate':[.03,.05,.1,.2],'model__max_depth':[2,3,5],'model__n_estimators':[100,200,350]}
   elif name=='Decision Tree Regressor': params={'model__max_depth':[None,3,5,8,12],'model__min_samples_leaf':[1,2,4,8]}
  try:
   search=RandomizedSearchCV(base,params,n_iter=min(12, max(1,np.prod([len(v) for v in params.values()]))),scoring=scoring,cv=5,random_state=SEED,n_jobs=1,refit=True)
   search.fit(Xtr,ytr); metrics=score_holdout(search.best_estimator_,Xte,yte,task)
   tuned.append({'model':name,'best_cv_score':float(search.best_score_),'best_params':search.best_params_,**metrics})
   if not hasattr(run_one,'_tuned'): run_one._tuned={}
   run_one._tuned[(key,name)]=search.best_estimator_
  except Exception as e: tuned.append({'model':name,'error':f'{type(e).__name__}: {e}'})
 if task=='regression':
  results['cv_rmse_mean']=-results['cv_score_mean']; results['cv_rmse_std']=results['cv_score_std']
 results.to_csv(out/'model_comparison.csv',index=False)
 pd.DataFrame(tuned).to_json(out/'tuning_results.json',orient='records',indent=2)
 # One-standard-error rule: if candidates are statistically close on CV, prefer simpler deployment.
 candidates=[]
 baseline_std={r['model']:float(r['cv_score_std']) for _,r in valid.iterrows()}
 for _,r in valid.iterrows(): candidates.append((float(r.cv_score_mean),r.model,fitted[r.model],'baseline'))
 for r in tuned:
  if 'best_cv_score' in r and (key,r['model']) in getattr(run_one,'_tuned',{}): candidates.append((r['best_cv_score'],r['model'],run_one._tuned[(key,r['model'])],'tuned'))
 cv_winner=max(candidates,key=lambda x:x[0])
 uncertainty=baseline_std.get(cv_winner[1],0.0)/np.sqrt(5)
 eligible=[c for c in candidates if c[0]>=cv_winner[0]-uncertainty]
 simplicity=(['Logistic Regression','Naive Bayes','Decision Tree','KNN','SVM','AdaBoost','Gradient Boosting','Random Forest','Extra Trees','Hist Gradient Boosting','LightGBM','XGBoost','CatBoost'] if task=='classification' else ['Ridge','Linear Regression','Lasso','Elastic Net','Polynomial Regression (degree 2)','SVR','KNN Regressor','Decision Tree Regressor','AdaBoost','Gradient Boosting','Random Forest Regressor','Extra Trees','Hist Gradient Boosting','LightGBM','XGBoost','CatBoost'])
 rank={name:i for i,name in enumerate(simplicity)}
 best=min(eligible,key=lambda c:(rank.get(c[1],len(rank)), 0 if c[3]=='tuned' else 1, -c[0]))
 model_path=out/'model_pipeline.joblib'; dump(best[2],model_path)
 # Held-out permutation importance explains the selected model; it is never used for selection.
 imp=permutation_importance(best[2],Xte,yte,n_repeats=8,random_state=SEED,n_jobs=1,scoring=scoring)
 pd.DataFrame({'feature':X.columns,'importance_mean':imp.importances_mean,'importance_std':imp.importances_std}).sort_values('importance_mean',ascending=False).to_csv(out/'feature_importance.csv',index=False)
 # Feature/data summary is part of auditable run outputs.
 summary={'dataset':cfg['file'],'rows':int(len(df)),'columns':int(df.shape[1]),'target':target,'task':task,
  'target_distribution':{str(k):int(v) for k,v in y.value_counts(dropna=False).items()} if task=='classification' else {'min':float(y.min()),'median':float(y.median()),'max':float(y.max())},
  'null_counts':{k:int(v) for k,v in df.isna().sum().items()},'categorical_features':X.select_dtypes(exclude=np.number).columns.tolist(),
  'numeric_features':engineer_features(X,key).select_dtypes(include=np.number).columns.tolist(),'raw_input_features':X.columns.tolist(),'engineered_features':[c for c in engineer_features(X,key).columns if c not in X.columns],'test_size':.2,'random_state':SEED,
  'final_model':best[1],'selection_source':best[3],'cv_best_model':cv_winner[1],'cv_best_score':cv_winner[0],'one_standard_error_tolerance':float(uncertainty),'selection_rule':'Among models within one standard error of the best CV score, prefer the simpler/deployment-friendlier model; prefer tuned settings for that estimator.','selection_metric':'5-fold CV weighted F1' if task=='classification' else '5-fold CV RMSE (lower is better)','final_cv_score':best[0],'final_cv_rmse':float(-best[0]) if task=='regression' else None,'target_label_mapping':label_mapping,'final_test_metrics':score_holdout(best[2],Xte,yte,task),
  'optional_boosters_available':[m for m in ['XGBoost','LightGBM','CatBoost'] if m in set(results.model)],
  'optional_boosters_missing':[m for m in ['XGBoost','LightGBM','CatBoost'] if m not in set(results.model)]}
 (out/'run_summary.json').write_text(json.dumps(summary,indent=2))
 print('FINAL',key,best[1],best[3],summary['final_test_metrics'],flush=True)
 return summary

if __name__=='__main__':
 all_summaries=[]
 for key,cfg in PROJECTS.items(): all_summaries.append(run_one(key,cfg))
 (ROOT/'model_selection'/'all_results.json').write_text(json.dumps(all_summaries,indent=2))
