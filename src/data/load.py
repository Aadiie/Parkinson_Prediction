import pandas as pd

DUPLICATES = ["Jitter:DDP", "Shimmer:DDA"]  
NON_FEATURES = ["name", "status", "subject_id", "session_id"]

def load_data(path):
    df = pd.read_csv(path)
    parts = df["name"].str.split("_")
    df["session_id"] = parts.str[1]
    df["subject_id"] = parts.str[2]

    
    assert parts.str.len().eq(4).all(), "unexpected name format"
    assert df.groupby("subject_id")["status"].nunique().eq(1).all(), "a subject has mixed labels"

    X = df.drop(columns=NON_FEATURES + DUPLICATES)
    y = df["status"]
    groups = df["subject_id"]
    return X, y, groups