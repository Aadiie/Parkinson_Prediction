import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from sklearn.inspection import permutation_importance

from src.models.pipeline import get_search_space, SEED


def get_scores(estimator, X):
    """Probability of PD if the model has one, otherwise SVM's signed distance.
    Both rank recordings the same way, which is all ROC-AUC / PR-AUC need."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    return estimator.decision_function(X)


def run_nested_cv(name, X, y, groups, outer_splits=5, inner_splits=3,
                  scoring="roc_auc", seed=SEED, importance=False):
    pipe, grid = get_search_space(name)
    outer = StratifiedGroupKFold(n_splits=outer_splits, shuffle=True, random_state=seed)

    scores = np.full(len(y), np.nan)
    pred = np.full(len(y), -1)
    fold_id = np.full(len(y), -1)
    best, imps = [], []

    for fold, (tr, te) in enumerate(outer.split(X, y, groups), start=1):
        inner = StratifiedGroupKFold(n_splits=inner_splits, shuffle=True, random_state=seed)
        search = GridSearchCV(pipe, grid, scoring=scoring, cv=inner, n_jobs=-1, refit=True)
        search.fit(X.iloc[tr], y.iloc[tr], groups=groups.iloc[tr])   # inner folds grouped too

        scores[te] = get_scores(search, X.iloc[te])
        pred[te] = search.predict(X.iloc[te])
        fold_id[te] = fold
        best.append({"fold": fold, "inner_auc": search.best_score_, **search.best_params_})

        if importance:   # measured on the held-out fold, never on training data
            r = permutation_importance(search.best_estimator_, X.iloc[te], y.iloc[te],
                                       scoring="roc_auc", n_repeats=30, random_state=seed)
            imps.append(pd.Series(r.importances_mean, index=X.columns, name=fold))

    oof = pd.DataFrame({"y_true": y.values, "score": scores, "pred": pred,
                        "fold": fold_id, "subject_id": groups.values})
    assert (oof["fold"] > 0).all(), "some rows never got a prediction"
    imp = pd.concat(imps, axis=1) if importance else None
    return oof, pd.DataFrame(best), imp