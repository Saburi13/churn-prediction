import os
import pandas as pd


def clean_telco(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    # Coerce TotalCharges to numeric (some entries are empty strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # For customers with tenure == 0, TotalCharges should be 0
    if 'tenure' in df.columns:
        mask = df['TotalCharges'].isna() & (df['tenure'] == 0)
        df.loc[mask, 'TotalCharges'] = 0.0
    # Fill any remaining TotalCharges NaNs with median
    if df['TotalCharges'].isna().any():
        median_tc = df['TotalCharges'].median()
        df['TotalCharges'] = df['TotalCharges'].fillna(median_tc)

    # Normalize Churn to binary 0/1 if present
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

    # Ensure output dir exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned CSV to {output_path}. Shape: {df.shape}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--in', dest='input', default='src/data/telco_churn.csv')
    parser.add_argument('--out', dest='output', default='data/clean_telco.csv')
    args = parser.parse_args()
    clean_telco(args.input, args.output)
