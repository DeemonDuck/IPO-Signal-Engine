from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def get_models():
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced"))
        ]),

        "Random Forest": Pipeline([
            ("model", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            ))
        ]),

        "Gradient Boosting": Pipeline([
            ("model", GradientBoostingClassifier(
                n_estimators=200,
                random_state=42
            ))
        ]),

        "XGBoost": Pipeline([
            ("model", XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                eval_metric="logloss"
            ))
        ]),

        "LightGBM": Pipeline([
            ("model", LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                random_state=42,
                verbosity=-1,
                force_col_wise=True
            ))
        ]),

        "CatBoost": Pipeline([
            ("model", CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=5,
                verbose=0
            ))
        ]),

        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                probability=True,
                class_weight="balanced",
                kernel="rbf",
                random_state=42
            ))
        ])
    }

    return models