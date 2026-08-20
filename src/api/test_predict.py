import json
import pandas as pd
import urllib.request
import urllib.error

# load first row from cleaned data
df = pd.read_csv('data/clean_telco.csv')
row = df.drop(columns=['Churn'], errors='ignore').iloc[0].to_dict()
payload = json.dumps({'features': row}).encode('utf-8')
req = urllib.request.Request('http://localhost:8000/predict', data=payload, headers={'Content-Type':'application/json'}, method='POST')
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        print(resp.read().decode())
except urllib.error.HTTPError as e:
    print('HTTPError', e.code, e.read().decode())
except Exception as e:
    print('Error', e)
