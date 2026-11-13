from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

SEED = 42


def build_pipeline(model):
    return Pipeline([
        ("power", PowerTransformer(method="yeo-johnson", standardize=False)),
        ("scale", RobustScaler()),
        ("select", SelectKBest(score_func=f_classif, k="all")),
        ("model", clone(model)),
    ])


MODELS = {
    "logreg": (LogisticRegression(class_weight="balanced", max_iter=5000),
               {"model__C": [0.001, 0.003, 0.01, 0.03, 0.1, 1, 10]}),
    "svm_rbf": (SVC(kernel="rbf", class_weight="balanced"),
                {"model__C": [0.01, 0.03, 0.1, 1, 10, 100],
                 "model__gamma": ["scale", 0.01, 0.1]}),
    "knn": (KNeighborsClassifier(),
            {"model__n_neighbors": [5, 11, 15, 21, 31],
             "model__weights": ["uniform", "distance"]}),
    "rf": (RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  random_state=SEED, n_jobs=1),
           {"model__max_depth": [None, 3, 5],
            "model__min_samples_leaf": [1, 3, 5, 10]}),
}


def get_search_space(name):
    model, grid = MODELS[name]
    return build_pipeline(model), {**grid, "select__k": [8, 12, "all"]}