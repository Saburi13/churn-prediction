import joblib
from typing import Any

def save_pipeline(pipeline: Any, path: str):
    joblib.dump(pipeline, path)

def load_pipeline(path: str):
    return joblib.load(path)