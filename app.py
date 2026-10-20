from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Credit Card Fraud Detection",page_icon="🛡️",layout="wide")
st.title("🛡️ Credit Card Fraud Detection")
st.caption("Research/educational prototype — not a bank decision system.")
path=Path("models/best_model.joblib")
if not path.exists():
    st.warning("Train first: python train.py --data data/creditcard.csv")
    st.stop()
bundle=joblib.load(path); model=bundle["model"]; threshold=float(bundle["threshold"]); features=bundle["feature_columns"]
st.sidebar.write("**Model:**",bundle.get("model_name","trained model"))
st.sidebar.write("**Threshold:**",f"{threshold:.4f}")
st.sidebar.info("A model score is not proof of fraud. Do not use for real financial decisions.")
single,batch,report=st.tabs(["Single transaction","Batch CSV","Evaluation report"])
with single:
    st.write("Enter values for each feature. For the classic Kaggle dataset, these include Time, V1–V28 and Amount.")
    vals={}; cols=st.columns(3)
    for i,f in enumerate(features):
        with cols[i%3]: vals[f]=st.number_input(f,value=(100.0 if f.lower()=="amount" else 0.0),format="%.6f",key=f)
    if st.button("Analyze transaction",type="primary"):
        prob=float(model.predict_proba(pd.DataFrame([[vals[f] for f in features]],columns=features))[0,1])
        st.metric("Estimated fraud probability",f"{prob:.2%}")
        if prob>=threshold: st.error("Flagged for review by the model")
        else: st.success("Not flagged at the current threshold")
with batch:
    up=st.file_uploader("Upload CSV with all required feature columns",type=["csv"])
    if up:
        try:
            df=pd.read_csv(up); missing=[f for f in features if f not in df.columns]
            if missing: st.error("Missing columns: "+", ".join(missing))
            elif df.empty:
                st.error("The uploaded CSV has no transaction rows.")
            else:
                try:
                    numeric = df[features].apply(pd.to_numeric, errors="raise")
                    if numeric.isna().any().any():
                        st.error("Feature columns contain missing values. Clean the CSV and try again.")
                    else:
                        probs=model.predict_proba(numeric)[:,1]; out=df.copy()
                        out["fraud_probability"]=probs; out["flagged_for_review"]=probs>=threshold
                        st.dataframe(out.head(100),use_container_width=True)
                        st.download_button("Download scored CSV",out.to_csv(index=False).encode(), "fraud_predictions.csv","text/csv")
                except (ValueError, TypeError) as e:
                    st.error(f"Feature columns must contain numeric values: {e}")
        except Exception as e: st.error(f"Could not read this CSV: {e}")
with report:
    p=Path("reports/metrics.json")
    if p.exists():
        m=json.loads(p.read_text())
        c=st.columns(4)
        c[0].metric("Precision",f"{m['fraud_precision']:.3f}"); c[1].metric("Recall",f"{m['fraud_recall']:.3f}")
        c[2].metric("F1",f"{m['fraud_f1']:.3f}"); c[3].metric("PR-AUC",f"{m['average_precision_pr_auc']:.3f}")
        st.write(f"Test rows: {m['total_count']} | Fraud prevalence: {m['fraud_prevalence']:.3%}")
        for f in ["reports/confusion_matrix.png","reports/precision_recall_curve.png","reports/feature_importance.png"]:
            if Path(f).exists(): st.image(f,use_container_width=True)
    else: st.info("Run training to generate metrics and plots.")
