# Data Contract

## Final modelling dataset

Dataset: Ames Housing
File: data/raw/ames_housing/train.csv
Observed dimensions: 1460 rows and 81 columns
Target: SalePrice
Target units: US dollars

Source:
https://www.kaggle.com/datasets/johniewesh/house-prices-data

Verify the downloaded dataset version and applicable licence.

## Modelling requirements

1. Exclude Id from predictive features unless its use is justified.
2. Split labelled data before fitting preprocessing transformations.
3. Fit imputers, encoders and scalers using training data only.
4. Use the same validation method when comparing candidate models.
5. Record the random seed, metrics and feature schema.
6. After model selection, retrain the approved pipeline on all
   labelled training rows for deployment.
7. Do not use the unlabelled Kaggle test file to calculate metrics.

## Missing values and outliers

Investigate missing values by feature. Distinguish structural
absence from data-entry errors.

Do not automatically remove rows or columns. Document and
justify decisions about missing values and unusual observations.

## Earlier Indian housing analysis

The earlier Indian housing dataset is a separate prototype.
Its EDA results must not be presented as results for the Ames model.

## Application requirements

The app must use the approved model pipeline and explicitly
document the inputs required for a prediction. Do not describe
predictions as current market prices or as Indian house prices.
