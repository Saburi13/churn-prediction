from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import CustomerFeatures
from api.logging import log_prediction
import os
import sys
# ensure `src/` is importable when running the api package
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from processing.preprocessor import build_preprocessor
import joblib
import time
import hashlib
import json
import pandas as pd
from typing import Any
from pydantic import ValidationError
from sklearn.pipeline import Pipeline

app = FastAPI(title="Churn Prediction API")
model_pipeline = None
model_pipelines = {}
model_meta = {"version": "0.1", "default_model": "logreg", "models": {}, "metrics": {}}

# Allow local dev frontend (Vite) to call the API
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_model():
    global model_pipeline, model_pipelines
    try:
        model_pipelines = _load_model_artifacts()
        model_pipeline = model_pipelines.get('logreg')
        _load_model_metadata()
    except Exception:
        model_pipeline = None


def _load_model_metadata():
    try:
        with open('models/model_info.json', 'r', encoding='utf8') as f:
            info = json.load(f)
            model_meta.update({
                'version': info.get('model', model_meta['version']),
                'default_model': info.get('default_model', 'logreg'),
                'models': info.get('models', {}),
                'metrics': info.get('metrics', {})
            })
    except Exception:
        pass


def _rebuild_pipeline(saved):
    """Rebuild preprocessing around a saved classifier for sklearn compatibility."""
    try:
        df = pd.read_csv('data/clean_telco.csv')
        X_df = df.drop(columns=['Churn'], errors='ignore')
        preproc, _, _ = build_preprocessor(X_df)
        preproc.fit(X_df)
        clf = saved.named_steps.get('clf', saved)
        return Pipeline([('pre', preproc), ('clf', clf)])
    except Exception:
        return saved


def _load_model_artifacts():
    loaded = {}
    for name, path in (
        ('logreg', 'models/model_logreg.joblib'),
        ('rf', 'models/model_rf.joblib'),
    ):
        try:
            saved = joblib.load(path)
            loaded[name] = _rebuild_pipeline(saved)
        except Exception:
            pass
    if not loaded:
        saved = joblib.load('models/pipeline.joblib')
        loaded['logreg'] = _rebuild_pipeline(saved)
    return loaded

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": bool(model_pipelines)}

@app.get("/model/info")
def model_info():
    return model_meta


@app.post("/model/reload")
def reload_model():
    """Reload the pipeline from disk without restarting the server."""
    global model_pipeline, model_pipelines
    try:
        model_pipelines = _load_model_artifacts()
        model_pipeline = model_pipelines.get('logreg')
        _load_model_metadata()
        return {"status": "reloaded", "model_loaded": bool(model_pipelines), "models": list(model_pipelines)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict(payload: dict):
    if not model_pipelines and model_pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        start = time.time()
        # Accept either top-level feature fields or a wrapper {"features": {...}}
        if 'features' in payload and isinstance(payload['features'], dict):
            features_in = payload['features']
        else:
            features_in = payload

        requested_model = payload.get('model', 'logreg')
        model_aliases = {
            'logistic_regression': 'logreg',
            'logistic regression': 'logreg',
            'random_forest': 'rf',
            'random forest': 'rf',
        }
        requested_model = model_aliases.get(str(requested_model).lower(), str(requested_model).lower())
        selected_pipeline = model_pipelines.get(requested_model)
        if selected_pipeline is None and requested_model == 'logreg':
            selected_pipeline = model_pipeline
        if selected_pipeline is None:
            raise HTTPException(status_code=400, detail=f"Unknown model '{requested_model}'. Choose 'logreg' or 'rf'.")

        # Validate using Pydantic model to get clear errors
        try:
            cf = CustomerFeatures.parse_obj(features_in)
        except ValidationError as ve:
            # Return validation errors from Pydantic
            raise HTTPException(status_code=400, detail=json.loads(ve.json()) if hasattr(ve, 'json') else str(ve))

        features = cf.dict()
        # Drop customerID before feeding into pipeline
        if 'customerID' in features:
            features.pop('customerID', None)

        df = pd.DataFrame([features])
        try:
            proba = float(selected_pipeline.predict_proba(df)[0, 1])
        except Exception:
            # Fallback: reconstruct preprocessor and use saved classifier to avoid
            # SimpleImputer pickle incompatibilities across sklearn versions.
            try:
                artifact = f"models/model_{requested_model}.joblib"
                saved = joblib.load(artifact if os.path.exists(artifact) else 'models/pipeline.joblib')
                clf = saved.named_steps.get('clf', saved)
            except Exception:
                raise

            # rebuild and fit preprocessor on cleaned training data
            try:
                from processing.preprocessor import build_preprocessor
                Xtrain = pd.read_csv('data/clean_telco.csv').drop(columns=['Churn'], errors='ignore')
                preproc, _, _ = build_preprocessor(Xtrain)
                preproc.fit(Xtrain)
                X_input = preproc.transform(df)
                proba = float(clf.predict_proba(X_input)[0, 1])
            except Exception as e:
                raise
        prediction = int(proba >= 0.5)
        payload_hash = hashlib.sha256(str(features).encode()).hexdigest()
        log_prediction(timestamp=time.time(), input_hash=payload_hash, input=features, model=requested_model, prob=proba, pred=prediction, duration=time.time()-start)
        return {"model": requested_model, "probability": proba, "prediction": prediction}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))