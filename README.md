# Ames Housing Price Predictor


An educational Streamlit app that predicts house sale prices using the Ames Housing dataset and a scikit-learn HistGradientBoostingRegressor pipeline.

## Dataset
Place train.csv at data/raw/ames_housing/train.csv. The raw dataset is excluded from Git and must not be committed.
The data describes historical sales in Ames, Iowa, from 2006–2010. Predictions are in historical US dollars, not current property valuations.

## Install dependencies
Use the project virtual environment:
    .\.venv\Scripts\python.exe -m pip install -r .\requirements.txt

## Train the model
    .\.venv\Scripts\python.exe .\src\train_model.py

## Run the app
    .\.venv\Scripts\python.exe -m streamlit run .\app.py

## Holdout validation results
- RMSE: $28,821.19
- MAE: $16,668.04
- R-squared: 0.8917
These metrics describe the configured holdout split. The app display range of plus/minus 10% is illustrative, not a calibrated confidence interval.

## Project
IDC 409 — Data Science and Machine Learning | Group 18 — IISER Mohali
