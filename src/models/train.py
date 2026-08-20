import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
from src.processing.preprocessor import build_preprocessor
from src.models.save_load import save_pipeline
from sklearn.pipeline import Pipeline
import numpy as np

def simple_train(data_path, target_col="Churn"):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=[target_col])
    X = df.drop(columns=[target_col])
    y = df[target_col].map({True:1, False:0}) if df[target_col].dtype == bool else df[target_col]
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=100, n_jobs=-1)
    }

    for name, model in models.items():
        pipe = Pipeline([('pre', preprocessor), ('clf', model)])
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        roc = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        print(f"{name} ROC-AUC CV mean: {roc.mean():.4f}")

    # fit best (choose rf here)
    final_pipe = Pipeline([('pre', preprocessor), ('clf', models['rf'])])
    final_pipe.fit(X, y)
    save_pipeline(final_pipe, "models/pipeline.joblib")
    print("Saved pipeline to models/pipeline.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="path to CSV")
    parser.add_argument("--target", default="Churn")
    args = parser.parse_args()
    simple_train(args.data, target_col=args.target)