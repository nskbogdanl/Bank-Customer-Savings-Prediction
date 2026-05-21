# Bank Customer Savings Prediction
My project from the competition at the university of Novosibirsk (it is from June 2025, so I don't fully remember everything, which is why some misunderstandings are possible. Also some decisions may be quite controversial, because it was my first ML project)
---

## Overview

This project predicts amount of customer's savings for a bank using tabular data.
The solution is built on **FLAML AutoML** with LightGBM, XGBoost, Random Forest, and Extra Trees as candidate estimators.
The target variable is log-transformed for training and inverse-transformed for final predictions.

---

## Data

You can download input_data from this link: 
"https://drive.google.com/drive/folders/1S4vPdqQEG2ngbnuECXb9gREWvcJs3lLm?usp=sharing".
Download the dataset from the google drive and place it in:

```
~/input_data/
```

Required files from dataset:
- `train_main_df.parquet`
- `test_main_df.parquet`
- `train_target.csv`

---

## Feature Engineering

- Missing value features (fraction, count, binary flag)
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
