# Bank Customer Savings Prediction
My project from the competition in university of Novosibirsk
---

## Overview

This project predicts amount of customer's savings for a bank using tabular data.
The solution is built on **FLAML AutoML** with LightGBM, XGBoost, Random Forest, and Extra Trees as candidate estimators.
The target variable is log-transformed for training and inverse-transformed for final predictions.

---

## Data

You can download input_data from this link: "https://drive.google.com/drive/folders/1S4vPdqQEG2ngbnuECXb9gREWvcJs3lLm?usp=sharing".
Download the dataset from the competition platform and place it in:

```
~/input_data/
```

Required files:
- `train_main_df.parquet`
- `test_main_df.parquet`
- `train_target.csv`

---

## Feature Engineering

- Missing value features (fraction, count, binary flag)
- Row-level statistics (mean, std, min, max, zero fraction)
- Pairwise feature interactions (division of numeric feature pairs)
- Quantile binning of top numeric features
- Ordinal encoding of categorical features
- Removal of highly correlated features (threshold: 0.95)
- Removal of zero-variance features
- Anomaly removal via IsolationForest (contamination: 1%)

---

## Model

- **Framework:** FLAML AutoML
- **Task:** Regression
- **Metric:** RMSE (log space)
- **CV:** 5-fold cross-validation
- **Time budget:** 1200 seconds
- **Estimators:** LightGBM, XGBoost, Random Forest, Extra Trees

---

## Results

Predictions are saved to:
```
~/output_data/submission_flaml.csv
```

---

## Author

Bogdan Lomp
GitHub: [Bank Customer Savings Prediction](https://github.com/nskbogdanl/Bank-Customer-Savings-Prediction)
