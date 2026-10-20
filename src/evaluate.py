import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_score, recall_score,
 f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, classification_report)

def choose_threshold(y, prob):
    precision, recall, thresholds = precision_recall_curve(y, prob)
    if not len(thresholds): return .5
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])

def metrics(y, prob, threshold):
    pred = (np.asarray(prob) >= threshold).astype(int)
    return {"threshold": float(threshold), "accuracy": float(accuracy_score(y,pred)),
      "fraud_precision": float(precision_score(y,pred,zero_division=0)),
      "fraud_recall": float(recall_score(y,pred,zero_division=0)),
      "fraud_f1": float(f1_score(y,pred,zero_division=0)),
      "average_precision_pr_auc": float(average_precision_score(y,prob)),
      "roc_auc": float(roc_auc_score(y,prob)), "total_count": int(len(y)),
      "fraud_count": int(np.sum(np.asarray(y)==1)), "fraud_prevalence": float(np.mean(y)),
      "confusion_matrix": confusion_matrix(y,pred).tolist(),
      "classification_report": classification_report(y,pred,output_dict=True,zero_division=0)}

def plots(y, prob, threshold, out, names, importances=None):
    out.mkdir(parents=True, exist_ok=True)
    ConfusionMatrixDisplay(confusion_matrix(y,(np.asarray(prob)>=threshold).astype(int)),
      display_labels=["Legitimate","Fraud"]).plot(values_format="d")
    plt.tight_layout(); plt.savefig(out/"confusion_matrix.png",dpi=150); plt.close()
    p,r,_ = precision_recall_curve(y,prob)
    plt.figure(); plt.plot(r,p); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall curve")
    plt.tight_layout(); plt.savefig(out/"precision_recall_curve.png",dpi=150); plt.close()
    if importances is not None:
        df = pd.DataFrame({"feature":names,"importance":importances}).nlargest(20,"importance").sort_values("importance")
        plt.figure(figsize=(9,7)); plt.barh(df.feature,df.importance); plt.title("Top feature importance")
        plt.tight_layout(); plt.savefig(out/"feature_importance.png",dpi=150); plt.close()
