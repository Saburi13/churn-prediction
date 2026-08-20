import argparse
import json
import os
import pandas as pd
try:
    from src.data.load_data import load_csv
except Exception:
    def load_csv(path: str):
        return pd.read_csv(path)


def profile(df: pd.DataFrame, target: str | None = None) -> dict:
    out = {}
    out['shape'] = df.shape
    out['dtypes'] = df.dtypes.apply(lambda x: x.name).to_dict()
    out['missing_values'] = df.isnull().sum().to_dict()
    out['missing_percent'] = (df.isnull().mean() * 100).round(2).to_dict()
    out['duplicates'] = int(df.duplicated().sum())
    # categorical cardinality
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    out['categorical_cardinality'] = {c: int(df[c].nunique()) for c in cat_cols}
    # basic stats for numeric
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    out['numeric_summary'] = df[num_cols].describe().to_dict()
    if target and target in df.columns:
        out['target_counts'] = df[target].value_counts(dropna=False).to_dict()
        out['target_balance'] = (df[target].value_counts(normalize=True, dropna=False) * 100).round(2).to_dict()
    return out


def main(data_path: str, target: str | None, out_path: str | None):
    df = load_csv(data_path)
    report = profile(df, target=target)
    text = []
    text.append(f"Loaded: {data_path}")
    text.append(f"Shape: {report['shape']}")
    text.append('\nDtypes:')
    for k, v in report['dtypes'].items():
        text.append(f"  {k}: {v}")
    text.append('\nMissing values (count, %):')
    for k in report['missing_values']:
        text.append(f"  {k}: {report['missing_values'][k]} ({report['missing_percent'][k]}%)")
    text.append(f"\nDuplicates: {report['duplicates']}")
    text.append('\nCategorical cardinality:')
    for k, v in report['categorical_cardinality'].items():
        text.append(f"  {k}: {v}")
    if target and 'target_counts' in report:
        text.append('\nTarget distribution:')
        for k, v in report['target_counts'].items():
            text.append(f"  {k}: {v} ({report['target_balance'][k]}%)")

    output_text = '\n'.join(text)
    print(output_text)

    # save JSON and text
    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w', encoding='utf8') as f:
            json.dump(report, f, indent=2)
        txt_path = os.path.splitext(out_path)[0] + '.txt'
        with open(txt_path, 'w', encoding='utf8') as f:
            f.write(output_text)
        print(f"Saved profile JSON to {out_path} and text to {txt_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='path to CSV file')
    parser.add_argument('--target', default='Churn', help='target column name (optional)')
    parser.add_argument('--out', default='reports/dataset_profile.json', help='output JSON path')
    args = parser.parse_args()
    main(args.data, args.target, args.out)
