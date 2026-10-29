# Ames Housing Price Predictor

An educational Streamlit application that predicts house sale prices using the Ames Housing dataset and an XGBoost regression model integrated with a scikit-learn preprocessing pipeline.

## Dataset

Place `train.csv` at `data/raw/ames_housing/train.csv`.

The raw dataset is excluded from Git and must not be committed to the repository.

The dataset contains historical house sales in Ames, Iowa, from 2006–2010. Predictions are expressed in historical US dollars and should not be interpreted as current property valuations.

## Install Dependencies

Install the required packages using the project's virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\requirements.txt
```

## Train the Model

Train and evaluate the models, then generate the model bundle used by the application:

```powershell
.\.venv\Scripts\python.exe .\src\train_model.py
```

Ensure the required dataset is available at the path specified above before training.

## Run the Application

Launch the Streamlit application:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\app.py
```

## Model Performance

The final XGBoost model was evaluated using the configured holdout validation split.

| Metric | Result |
|---|---:|
| R-squared (R²) | **0.9256** |
| RMSE | **$23,895.42** |
| MAE | **$15,126.71** |

The R² value of 0.9256 indicates that the model explains approximately 92.56% of the variation in sale prices in the holdout dataset.

### Model Comparison

The following models were evaluated using the same configured holdout split.

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Linear Regression | $29,475.52 | $18,287.85 | 0.8867 |
| Random Forest | $28,845.87 | $17,468.11 | 0.8915 |
| HistGradientBoosting | $28,821.19 | $16,668.04 | 0.8917 |
| **XGBoost** | **$23,895.42** | **$15,126.71** | **0.9256** |

XGBoost achieved the strongest results among the evaluated models on this holdout split.

**Note:** The app's displayed price range of ±10% is illustrative and is not a calibrated confidence interval. Predictions are educational estimates based on historical data and are not professional property valuations.

## Project Information

**Course:** IDC 409 — Data Science and Machine Learning  
**Group:** 18  
**Institution:** IISER Mohali
