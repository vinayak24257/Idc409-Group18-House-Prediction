from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "ames_housing" / "train.csv"
MODEL_PATH = ROOT / "models" / "house_price_bundle.joblib"


def build_pipeline(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])

    model = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=20,
        l2_regularization=1.0,
        random_state=42,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "SalePrice" not in df.columns:
        raise ValueError("SalePrice column is missing.")

    df = df.dropna(subset=["SalePrice"]).copy()
    X = df.drop(columns=["SalePrice", "Id"], errors="ignore")
    y = df["SalePrice"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    validation_pipeline = build_pipeline(X_train)
    validation_pipeline.fit(X_train, y_train)
    predictions = validation_pipeline.predict(X_valid)

    rmse = float(np.sqrt(mean_squared_error(y_valid, predictions)))
    mae = float(mean_absolute_error(y_valid, predictions))
    r2 = float(r2_score(y_valid, predictions))

    print("Holdout validation metrics")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE:  ${mae:,.2f}")
    print(f"R2:   {r2:.4f}")

    # Retrain the deployment pipeline on all available labelled rows.
    final_pipeline = build_pipeline(X)
    final_pipeline.fit(X, y)

    # Keep one complete input row so the app always supplies every feature.
    # Use a fresh row with object-compatible columns so numeric medians
    # such as 9478.5 can be assigned without integer dtype errors.
    base_row = X.iloc[[0]].astype(object).copy()
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(
        include=["object", "str", "category", "bool"]
    ).columns.tolist()

    for col in numeric:
        median = X[col].median()
        if pd.notna(median):
            base_row.loc[base_row.index[0], col] = float(median)

    category_options = {}
    for col in categorical:
        modes = X[col].mode(dropna=True)
        if not modes.empty:
            base_row.loc[:, col] = modes.iloc[0]
        category_options[col] = sorted(
            X[col].dropna().astype(str).unique().tolist()
        )

    bundle = {
        "pipeline": final_pipeline,
        "feature_columns": X.columns.tolist(),
        "base_row": base_row,
        "category_options": category_options,
        "validation_metrics": {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        },
        "model_name": "HistGradientBoostingRegressor",
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model bundle: {MODEL_PATH}")


if __name__ == "__main__":
    main()
