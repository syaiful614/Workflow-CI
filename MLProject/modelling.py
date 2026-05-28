"""
modelling.py (untuk MLProject)
File ini dipanggil oleh MLflow Project runner via conda/docker environment.
Mendukung parameter CLI sesuai definisi di MLProject file.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import json
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from sklearn.model_selection import cross_val_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Heart Disease model via MLflow Project")
    parser.add_argument("--n_estimators",      type=int,   default=200,   help="RF n_estimators")
    parser.add_argument("--max_depth",         type=int,   default=8,     help="RF max_depth")
    parser.add_argument("--min_samples_split", type=int,   default=5,     help="RF min_samples_split")
    parser.add_argument("--max_features",      type=str,   default="sqrt", help="RF max_features")
    parser.add_argument("--train_path",        type=str,   default="heart_preprocessing/train.csv")
    parser.add_argument("--test_path",         type=str,   default="heart_preprocessing/test.csv")
    return parser.parse_args()


def load_data(train_path, test_path):
    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)
    X_train = train.drop("target", axis=1)
    y_train = train["target"]
    X_test  = test.drop("target", axis=1)
    y_test  = test["target"]
    return X_train, X_test, y_train, y_test


def plot_cm(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"],
                yticklabels=["No Disease", "Disease"])
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_roc(y_true, y_prob, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC Curve"); plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    os.makedirs("mlproject_artifacts", exist_ok=True)

    # Setup MLflow (DagsHub jika env var tersedia)
    dagshub_token = os.environ.get("DAGSHUB_TOKEN", "")
    dagshub_user  = os.environ.get("DAGSHUB_USERNAME", "")
    dagshub_repo  = os.environ.get("DAGSHUB_REPO", "Heart_Disease_MLOps")

    if dagshub_token and dagshub_user:
        import dagshub
        dagshub.init(repo_owner=dagshub_user, repo_name=dagshub_repo, mlflow=True)
    else:
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))

    mlflow.set_experiment("Heart_Disease_CI_Pipeline")

    X_train, X_test, y_train, y_test = load_data(args.train_path, args.test_path)

    with mlflow.start_run(run_name="CI_RandomForest"):
        # Train
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else None,
            min_samples_split=args.min_samples_split,
            max_features=args.max_features,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # CV scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

        # Manual logging
        mlflow.log_params({
            "n_estimators":      args.n_estimators,
            "max_depth":         args.max_depth,
            "min_samples_split": args.min_samples_split,
            "max_features":      args.max_features,
        })

        metrics = {
            "accuracy":         accuracy_score(y_test, y_pred),
            "precision":        precision_score(y_test, y_pred),
            "recall":           recall_score(y_test, y_pred),
            "f1_score":         f1_score(y_test, y_pred),
            "roc_auc":          roc_auc_score(y_test, y_prob),
            "cv_mean_accuracy": cv_scores.mean(),
            "cv_std_accuracy":  cv_scores.std(),
        }
        mlflow.log_metrics(metrics)

        # Artifacts
        cm_path  = "mlproject_artifacts/confusion_matrix.png"
        roc_path = "mlproject_artifacts/roc_curve.png"
        plot_cm(y_test, y_pred, cm_path)
        plot_roc(y_test, y_prob, roc_path)

        # Classification report
        report_path = "mlproject_artifacts/classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(y_test, y_pred))

        for p in [cm_path, roc_path, report_path]:
            mlflow.log_artifact(p)

        # Log model
        mlflow.sklearn.log_model(model, artifact_path="model")

    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    # Save model locally for serving (Kriteria 4)
    import pickle
    model_path = "mlproject_artifacts/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {model_path}")
    logger.info("✅ CI Pipeline completed successfully!")


if __name__ == "__main__":
    main()
