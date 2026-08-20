import os
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.pipeline import Pipeline


def main(pipeline_path='models/pipeline.joblib', data_path='data/clean_telco.csv', out_dir='reports'):
    os.makedirs(out_dir, exist_ok=True)
    print('Loading pipeline...')
    pipe = joblib.load(pipeline_path)
    print('Loading data...')
    df = pd.read_csv(data_path)
    X = df.drop(columns=['Churn'], errors='ignore')
    # drop id if present
    if 'customerID' in X.columns:
        X = X.drop(columns=['customerID'])

    # use a small sample for SHAP to keep runtimes reasonable
    sample = X.sample(n=min(200, len(X)), random_state=42)

    # For models inside a sklearn Pipeline, explain after preprocessing (numeric features)
    if isinstance(pipe, Pipeline):
        preproc = pipe[:-1]
        model = pipe[-1]
    else:
        preproc = None
        model = pipe

    if preproc is not None:
        print('Transforming sample with pipeline preprocessor...')
        X_trans = preproc.transform(sample)
    else:
        X_trans = sample.values

    # attempt to get transformed feature names
    feature_names = None
    try:
        if preproc is not None:
            feature_names = preproc.get_feature_names_out(sample.columns)
    except Exception:
        try:
            feature_names = preproc.get_feature_names_out()
        except Exception:
            feature_names = [f'f{i}' for i in range(X_trans.shape[1])]

    print('Creating SHAP LinearExplainer (suitable for linear models)...')
    try:
        explainer = shap.LinearExplainer(model, X_trans, feature_perturbation='interventional')
    except Exception:
        try:
            explainer = shap.LinearExplainer(model, X_trans)
        except Exception:
            explainer = None

    if explainer is None:
        # fallback: use KernelExplainer on the whole pipeline (slower)
        print('Falling back to KernelExplainer (slower) on full pipeline...')
        predict_fn = lambda x: pipe.predict_proba(pd.DataFrame(x, columns=sample.columns))[:, 1]
        explainer = shap.KernelExplainer(predict_fn, sample.iloc[:50])
        shap_values = explainer.shap_values(sample.iloc[:50])
        X_for_plot = sample.iloc[:50]
    else:
        print('Computing SHAP values on transformed features...')
        try:
            # new API
            shap_expl = explainer(X_trans)
            shap_values = shap_expl.values
        except Exception:
            # older API
            shap_values = explainer.shap_values(X_trans)
        X_for_plot = X_trans

    # Summary plot
    plt.figure(figsize=(10, 6))
    try:
        shap.summary_plot(shap_values, X_for_plot, feature_names=feature_names, show=False)
    except Exception:
        # try without feature names
        shap.summary_plot(shap_values, X_for_plot, show=False)
    summary_path = os.path.join(out_dir, 'shap_summary.png')
    plt.savefig(summary_path, bbox_inches='tight')
    plt.close()
    print('Saved summary plot to', summary_path)

    # Dependence plot for top feature (use index)
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = int(np.argmax(mean_abs))
    top_feature = feature_names[top_idx] if feature_names is not None else str(top_idx)
    print('Top feature by mean |SHAP|:', top_feature)
    plt.figure(figsize=(8, 6))
    try:
        shap.dependence_plot(top_idx, shap_values, X_for_plot, feature_names=feature_names, show=False)
    except Exception:
        try:
            shap.dependence_plot(top_feature, shap_values, X_for_plot, feature_names=feature_names, show=False)
        except Exception:
            pass
    dep_path = os.path.join(out_dir, f'shap_dependence_{top_feature}.png')
    plt.savefig(dep_path, bbox_inches='tight')
    plt.close()
    print('Saved dependence plot to', dep_path)


if __name__ == '__main__':
    main()
