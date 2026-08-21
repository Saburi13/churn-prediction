import os
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.pipeline import Pipeline
# ensure local src is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from processing.preprocessor import build_preprocessor
from joblib import dump


def load_data(path: str):
    return pd.read_csv(path)


def train_and_evaluate(data_path: str, target_col: str = 'Churn'):
    df = load_data(data_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column {target_col} not found in data")

    # Drop identifier
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    preprocessor, num_cols, cat_cols = build_preprocessor(X)

    models = {
        'logreg': LogisticRegression(max_iter=1000, class_weight='balanced', solver='lbfgs'),
        'rf': RandomForestClassifier(n_estimators=100, n_jobs=-1, class_weight='balanced')
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, clf in models.items():
        pipe = Pipeline([('pre', preprocessor), ('clf', clf)])
        roc = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        pr = cross_val_score(pipe, X, y, cv=cv, scoring='average_precision', n_jobs=-1)
        results[name] = {'roc_auc_mean': float(np.mean(roc)), 'pr_auc_mean': float(np.mean(pr))}
        print(f"{name} CV ROC-AUC: {results[name]['roc_auc_mean']:.4f}, PR-AUC: {results[name]['pr_auc_mean']:.4f}")

    # Select best by PR-AUC
    best_name = max(results.keys(), key=lambda k: results[k]['pr_auc_mean'])
    best_clf = models[best_name]
    print(f"Selected best model: {best_name}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    final_pipelines = {}
    test_metrics = {}
    for name, clf in models.items():
        final_pipe = Pipeline([('pre', build_preprocessor(X_train)[0]), ('clf', clf)])
        final_pipe.fit(X_train, y_train)
        y_proba = final_pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        metrics = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'roc_auc': float(roc_auc_score(y_test, y_proba)),
            'pr_auc': float(average_precision_score(y_test, y_proba)),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        final_pipelines[name] = final_pipe
        test_metrics[name] = metrics
        print(f'{name} test metrics:', metrics)

    # Save both pipelines and preserve pipeline.joblib as the Logistic Regression default.
    os.makedirs('models', exist_ok=True)
    for name, pipeline in final_pipelines.items():
        dump(pipeline, os.path.join('models', f'model_{name}.joblib'))
    dump(final_pipelines['logreg'], os.path.join('models', 'pipeline.joblib'))
    meta = {'default_model': 'logreg', 'models': test_metrics, 'model': best_name, 'metrics': test_metrics[best_name]}
    with open(os.path.join('models', 'model_info.json'), 'w', encoding='utf8') as f:
        json.dump(meta, f, indent=2)

    print('Saved model_logreg.joblib, model_rf.joblib, pipeline.joblib, and model_info.json')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/clean_telco.csv')
    parser.add_argument('--target', default='Churn')
    args = parser.parse_args()
    train_and_evaluate(args.data, args.target)
