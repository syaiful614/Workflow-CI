"""
modelling.py - untuk MLflow Project CI Pipeline
"""
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import cross_val_score

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--min_samples_split", type=int, default=5)
    parser.add_argument("--max_features", type=str, default="sqrt")
    parser.add_argument("--train_path", type=str, default="MLProject/heart_preprocessing/train.csv")
    parser.add_argument("--test_path", type=str, default="MLProject/heart_preprocessing/test.csv")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs("mlproject_artifacts", exist_ok=True)

    mlflow.set_tracking_uri("mlruns")
    mlflow.set_experiment("Heart_Disease_CI")

    train = pd.read_csv(args.train_path)
    test  = pd.read_csv(args.test_path)
    X_train = train.drop("target", axis=1)
    y_train = train["target"]
    X_test  = test.drop("target", axis=1)
    y_test  = test["target"]

    with mlflow.start_run(run_name="CI_RandomForest"):
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth if args.max_depth > 0 else None,
            min_samples_split=args.min_samples_split,
            max_features=args.max_features,
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        cv = cross_val_score(model, X_train, y_train, cv=5)

        mlflow.log_params({
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "max_features": args.max_features,
        })
        mlflow.log_metrics({
            "accuracy":         accuracy_score(y_test, y_pred),
            "precision":        precision_score(y_test, y_pred),
            "recall":           recall_score(y_test, y_pred),
            "f1_score":         f1_score(y_test, y_pred),
            "roc_auc":          roc_auc_score(y_test, y_prob),
            "cv_mean_accuracy": cv.mean(),
        })

        mlflow.sklearn.log_model(model, artifact_path="model")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
        print("CI Pipeline selesai!")

if __name__ == "__main__":
    main()