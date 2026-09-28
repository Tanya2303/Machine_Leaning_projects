"""Regenerate the supplied-data EDA plots without fitting models."""
from model_selection.run_experiments import ROOT,PROJECTS,load_data,split_xy,save_eda_plots
for key,cfg in PROJECTS.items():
    df=load_data(key); X,y=split_xy(df,cfg['target'])
    save_eda_plots(df,X,y,cfg['task'],ROOT/'projects'/key)
    print(f"Wrote EDA plots for {key}")
