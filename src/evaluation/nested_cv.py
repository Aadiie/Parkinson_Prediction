import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from src.models.pipeline import get_search_space, SEED


def get_scores(estimator, X):
    
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


def run_nested_cv(name, X, y, groups, outer_splits=5, inner_splits=3, scoring="roc_auc"):
    pipe, grid = get_search_space(name)
    outer = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=SEED)

    scores = np.full(len(y), np.nan)
    pred = np.full(len(y), -1)
    fold_id = np.full(len(y), -1)
    best = []

    for fold, (tr, te) in enumerate(outer.split(X, y, groups), start=1):
        inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=SEED)
        search = GridSearchCV(pipe, grid, scoring=scoring, cv=inner, n_jobs=-1, refit=True)
        search.fit(X.iloc[tr], y.iloc[tr], groups=groups.iloc[tr])   # inner folds grouped too

        scores[te] = get_scores(search, X.iloc[te])
        pred[te] = search.predict(X.iloc[te])
        fold_id[te] = fold
        best.append({"fold": fold, "inner_auc": search.best_score_, **search.best_params_})

    oof = pd.DataFrame({"y_true": y.values, "score": scores, "pred": pred,
                        "fold": fold_id, "subject_id": groups.values})
    assert (oof["fold"] > 0).all(), "some rows never got a prediction"
    return oof, pd.DataFrame(best)