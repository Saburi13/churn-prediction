import json
import time

def log_prediction(**kwargs):
    entry = {"ts": kwargs.get("timestamp", time.time())}
    entry.update({k: kwargs[k] for k in kwargs if k != "timestamp"})
    print(json.dumps(entry))