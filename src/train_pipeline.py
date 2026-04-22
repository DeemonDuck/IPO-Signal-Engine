import pandas as pd
import yaml
import warnings
import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report

from models import get_models

warnings.filterwarnings("ignore")


# -----------------------------
# Load Config
# -----------------------------
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# -----------------------------
# Load Data
# -----------------------------
def load_data(path):
    return pd.read_csv(path)


# -----------------------------
# Prepare Data
# -----------------------------
def prepare_data(df, target_col, test_size, random_state):
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


# -----------------------------
# Train & Evaluate
# -----------------------------
def train_and_evaluate(models, X_train, X_test, y_train, y_test):

    results = []
    best_model = None
    best_score = 0

    for name, model in models.items():
        print(f"\nTraining: {name}")

        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=5,
            scoring="roc_auc"
        )

        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = model.predict(X_test)

        roc = roc_auc_score(y_test, probs)

        print(f"CV ROC-AUC: {cv_scores.mean():.4f}")
        print(f"Test ROC-AUC: {roc:.4f}")
        print(classification_report(y_test, preds))

        results.append({
            "Model": name,
            "CV_ROC_AUC": cv_scores.mean(),
            "Test_ROC_AUC": roc
        })

        # Track best model
        if roc > best_score:
            best_score = roc
            best_model = model

    results_df = pd.DataFrame(results).sort_values(
        "Test_ROC_AUC", ascending=False
    )

    return results_df, best_model


# -----------------------------
# Save Model
# -----------------------------
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"\nBest model saved at: {path}")


# -----------------------------
# Main Pipeline
# -----------------------------
def run_pipeline(config):

    data_path = config["data"]["processed_path"]
    target_col = "Apply_Label"

    test_size = config["model"]["test_size"]
    random_state = config["model"]["random_state"]

    model_output_path = "models/best_model.pkl"

    df = load_data(data_path)

    X_train, X_test, y_train, y_test = prepare_data(
        df, target_col, test_size, random_state
    )

    models = get_models()

    results_df, best_model = train_and_evaluate(
        models, X_train, X_test, y_train, y_test
    )

    print("\nFinal Model Comparison:")
    print(results_df.reset_index(drop=True))

    save_model(best_model, model_output_path)


# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    config = load_config("config.yaml")
    run_pipeline(config)