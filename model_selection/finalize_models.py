"""Rebuild final pipelines from saved CV/tuning reports using a one-SE simplicity rule.
Normally run_experiments.py performs this directly; this utility reselects without repeating CV.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from model_selection.run_experiments import (PROJECTS,load_data,split_xy,engineer_features,
    make_preprocessor,model_zoo,score_holdout,SEED)

CLASS_ORDER=['Logistic Regression','Naive Bayes','Decision Tree','KNN','SVM','AdaBoost','Gradient Boosting','Random Forest','Extra Trees','Hist Gradient Boosting','LightGBM','XGBoost','CatBoost']
REG_ORDER=['Ridge','Linear Regression','Lasso','Elastic Net','Polynomial Regression (degree 2)','SVR','KNN Regressor','Decision Tree Regressor','AdaBoost','Gradient Boosting','Random Forest Regressor','Extra Trees','Hist Gradient Boosting','LightGBM','XGBoost','CatBoost']

def finalize(key,cfg):
 out=ROOT/'projects'/key; task=cfg['task']; df=load_data(key); X,y=split_xy(df,cfg['target']); mapping=None
 if task=='classification' and y.dtype==object:
  enc=LabelEncoder(); y=pd.Series(enc.fit_transform(y),index=y.index,name=y.name); mapping={str(i):str(v) for i,v in enumerate(enc.classes_)}
 strat=y if task=='classification' and y.value_counts().min()>=2 else None
 Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=SEED,stratify=strat)
 table=pd.read_csv(out/'model_comparison.csv').dropna(subset=['cv_score_mean'])
 tuned=json.loads((out/'tuning_results.json').read_text())
 candidates=[]
 for _,r in table.iterrows(): candidates.append({'score':float(r.cv_score_mean),'std':float(r.cv_score_std),'name':r.model,'source':'baseline','params':{}})
 for r in tuned:
  if 'best_cv_score' in r and isinstance(r.get('best_params'),dict): candidates.append({'score':float(r['best_cv_score']),'std':np.nan,'name':r['model'],'source':'tuned','params':r['best_params']})
 winner=max(candidates,key=lambda x:x['score']); base=table[table.model==winner['name']].iloc[0]; tolerance=float(base.cv_score_std/np.sqrt(5))
 eligible=[c for c in candidates if c['score']>=winner['score']-tolerance]
 rank={n:i for i,n in enumerate(CLASS_ORDER if task=='classification' else REG_ORDER)}
 selected=min(eligible,key=lambda c:(rank.get(c['name'],len(rank)),0 if c['source']=='tuned' else 1,-c['score']))
 estimator=model_zoo(task,len(Xtr))[selected['name']]
 pre=make_preprocessor(engineer_features(Xtr,key))
 pipe=Pipeline([('feature_engineering',FunctionTransformer(engineer_features,kw_args={'project':key},validate=False)),('preprocess',pre),('model',estimator)])
 if selected['params']: pipe.set_params(**selected['params'])
 pipe.fit(Xtr,ytr); dump(pipe,out/'model_pipeline.joblib')
 scoring='f1_weighted' if task=='classification' else 'neg_root_mean_squared_error'
 imp=permutation_importance(pipe,Xte,yte,n_repeats=8,random_state=SEED,n_jobs=1,scoring=scoring)
 pd.DataFrame({'feature':X.columns,'importance_mean':imp.importances_mean,'importance_std':imp.importances_std}).sort_values('importance_mean',ascending=False).to_csv(out/'feature_importance.csv',index=False)
 summary=json.loads((out/'run_summary.json').read_text())
 summary.update({'final_model':selected['name'],'selection_source':selected['source'],'selected_cv_score':selected['score'],
  'final_cv_rmse':float(-selected['score']) if task=='regression' else None,'cv_best_model':winner['name'],'cv_best_score':winner['score'],
  'one_standard_error_tolerance':tolerance,'selection_rule':'Among models within one standard error of the best CV score, prefer the simpler/deployment-friendlier model; prefer tuned settings for that estimator.',
  'target_label_mapping':mapping,'final_test_metrics':score_holdout(pipe,Xte,yte,task),
  'raw_input_features':X.columns.tolist(),'engineered_features':[c for c in engineer_features(X,key).columns if c not in X.columns]})
 (out/'run_summary.json').write_text(json.dumps(summary,indent=2))
 print(key,summary['final_model'],summary['selection_source'],summary['final_test_metrics'])

if __name__=='__main__':
 for key,cfg in PROJECTS.items(): finalize(key,cfg)
