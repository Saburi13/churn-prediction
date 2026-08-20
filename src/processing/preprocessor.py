from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pandas as pd


def build_preprocessor(X: pd.DataFrame):
    numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
    # exclude target if present
    if 'Churn' in numeric_cols:
        numeric_cols.remove('Churn')
    categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    # remove id-like columns if present
    for col in ['customerID', 'CustomerID', 'id', 'ID']:
        if col in categorical_cols:
            categorical_cols.remove(col)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Build OneHotEncoder in a version-compatible way (sparse vs sparse_output)
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', ohe)
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ], remainder='drop')

    return preprocessor, numeric_cols, categorical_cols


if __name__ == '__main__':
    print('This module provides build_preprocessor(X) -> (preprocessor, num_cols, cat_cols)')
 