import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, recall_score,
                             precision_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix)

def summarize(df):
    y, p, s = df["y_true"], df["pred"], df["score"]
    return {
        "accuracy": accuracy_score(y, p),
        "balanced_acc": balanced_accuracy_score(y, p),
        "sensitivity_PD": recall_score(y, p, pos_label=1),
        "specificity_healthy": recall_score(y, p, pos_label=0),
        "precision_PD": precision_score(y, p, pos_label=1, zero_division=0),
        "f1_PD": f1_score(y, p, pos_label=1),
        "roc_auc": roc_auc_score(y, s),
        "pr_auc_healthy": average_precision_score(1 - y, -s),
    }

def per_fold(df):
    rows = [summarize(g) for _, g in df.groupby("fold")]
    return pd.DataFrame(rows).agg(["mean", "std"]).T


def subject_level(df):
    
    subj = df.groupby("subject_id").agg(y_true=("y_true", "first"), vote=("pred", "mean"))
    subj["pred"] = (subj["vote"] >= 0.5).astype(int)
    n_ties = int((subj["vote"] == 0.5).sum())
    return subj, confusion_matrix(subj["y_true"], subj["pred"]), n_ties