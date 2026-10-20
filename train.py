import argparse, json
from pathlib import Path
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from src.data import load_dataset, split_dataset
from src.evaluate import choose_threshold, metrics, plots

def model_candidates(seed=42):
    return {
      "LogisticRegression_balanced": Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(class_weight="balanced",max_iter=2000,random_state=seed))]),
      "RandomForest_balanced": Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestClassifier(n_estimators=250,min_samples_leaf=2,class_weight="balanced_subsample",n_jobs=-1,random_state=seed))]),
      "LogisticRegression_SMOTE": ImbPipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("smote",SMOTE(random_state=seed,k_neighbors=3)),("model",LogisticRegression(max_iter=2000,random_state=seed))])
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="data/creditcard.csv")
    ap.add_argument("--max-rows",type=int,default=None,help="Development-only stratified sample")
    args=ap.parse_args()
    X,y=load_dataset(args.data,args.max_rows)
    Xtr,Xv,Xte,ytr,yv,yte=split_dataset(X,y)
    print(f"Rows={len(X)}, features={X.shape[1]}, fraud rate={y.mean():.4%}")
    trained={}
    for name,m in model_candidates().items():
        print("Training",name)
        try:
            m.fit(Xtr,ytr); pv=m.predict_proba(Xv)[:,1]; t=choose_threshold(yv,pv); vm=metrics(yv,pv,t)
            trained[name]=(m,t,vm)
            print(f" validation F1={vm['fraud_f1']:.4f}, PR-AUC={vm['average_precision_pr_auc']:.4f}, threshold={t:.4f}")
        except Exception as e: print("Skipped",name,"because:",e)
    if not trained: raise RuntimeError("No model trained successfully.")
    best=max(trained,key=lambda n:trained[n][2]["average_precision_pr_auc"])
    m,t,_=trained[best]; p=m.predict_proba(Xte)[:,1]; result=metrics(yte,p,t)
    result.update({"selected_model":best,"feature_columns":list(X.columns),
      "selection_rule":"Highest validation average precision; threshold selected by validation F1","random_state":42})
    Path("models").mkdir(exist_ok=True); Path("reports").mkdir(exist_ok=True)
    joblib.dump({"model":m,"threshold":t,"feature_columns":list(X.columns),"model_name":best},"models/best_model.joblib")
    Path("reports/metrics.json").write_text(json.dumps(result,indent=2))
    pd.DataFrame([{"model":n,"validation_f1":v[2]["fraud_f1"],"validation_pr_auc":v[2]["average_precision_pr_auc"],"threshold":v[1]} for n,v in trained.items()]).to_csv("reports/model_comparison.csv",index=False)
    core=m.named_steps.get("model"); imp=getattr(core,"feature_importances_",None)
    if imp is None and hasattr(core,"coef_"): imp=abs(core.coef_[0])
    plots(yte,p,t,Path("reports"),list(X.columns),imp)
    print("\nFINAL HOLD-OUT TEST RESULTS")
    for k in ["selected_model","threshold","fraud_precision","fraud_recall","fraud_f1","average_precision_pr_auc","roc_auc","accuracy"]:
        print(f"{k}: {result[k]}")
    print("Saved model to models/best_model.joblib and report to reports/")

if __name__=="__main__": main()
