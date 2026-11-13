import json
from pathlib import Path

import joblib
import sklearn
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV

from src.models.pipeline import get_search_space, SEED


def train_final(name, X, y, groups, out_dir="../models"):
    
    pipe, grid = get_search_space(name)
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1, refit=True)
    search.fit(X, y, groups=groups)

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    joblib.dump(search.best_estimator_, out / f"{name}_final.joblib")

    meta = {
        "model": name,
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "features_in_order": list(X.columns),
        "sklearn_version": sklearn.__version__,
        "warning": "Performance = nested CV results, not best_score_.",
    }
    (out / f"{name}_final.json").write_text(json.dumps(meta, indent=2))
    return search