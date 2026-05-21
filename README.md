# Bank Customer Savings Prediction

My project from the competition at the university of Novosibirsk (it is from June 2025, so I don't fully remember everything, which is why some misunderstandings are possible. Also some decisions may be quite controversial, because it was my first ML project)

---

## Overview

This project predicts the amount of customer savings for a bank using tabular data.
The solution is based on a custom preprocessing pipeline and an XGBoost Regressor.

---

## Data

Dataset:
[Download here](https://drive.google.com/drive/folders/1S4vPdqQEG2ngbnuECXb9gREWvcJs3lLm?usp=sharing)

Place the files inside:

```bash
~/input_data/
```

Required files:

* `train_main_df.parquet`
* `test_main_df.parquet`
* `train_target.csv`

---

## Project Structure

```text
project/
│
├── input_data/
│   ├── dictionary_main.csv
│   ├── train_main_df.parquet
│   ├── test_main_df.parquet
│   └── train_target.csv
│
├── output_data/
│   ├── xgb_model.pkl
│   ├── ordinal_encoder.pkl
│   ├── variance_selector.pkl
│   └── dropped_corr.json
│
└── main.py
```

---

## Preprocessing Pipeline

### 1. Missing Value Features

Additional features are created based on missing values:

* `missing_fraction` — fraction of missing values in a row
* `many_missing` — binary flag for rows with more than 50% missing values

---

### 2. Highly Correlated Feature Removal

Numeric features with correlation above `0.95` are removed to reduce redundancy.

Removed columns are saved into:

```text
output_data/dropped_corr.json
```

---

### 3. Anomaly Detection

Outliers are removed using:

* `IsolationForest`
* contamination rate: `1%`

---

### 4. Categorical Encoding

Categorical columns are encoded using:

* `OrdinalEncoder`
* unknown categories are encoded as `-1`

The fitted encoder is saved as:

```text
output_data/ordinal_encoder.pkl
```

---

### 5. Missing Value Filling

Remaining missing values are filled with:

```python
0
```

---

### 6. Zero-Variance Feature Removal

Features with no variance are removed using:

* `VarianceThreshold`

The selector is saved as:

```text
output_data/variance_selector.pkl
```

---

## Model

### XGBoost Regressor

Parameters:

```python
XGBRegressor(
    n_estimators=5000,
    max_depth=8,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
```

---

## Training

### Cross-Validation

* 5-Fold Cross-Validation
* Metric: RMSE
* Training target:

```python
np.log1p(target)
```

Validation is performed in log-space.

---

## Saved Artifacts

After training, the following files are generated:

```text
output_data/
├── xgb_model.pkl
├── ordinal_encoder.pkl
├── variance_selector.pkl
└── dropped_corr.json
```

---

## Running the Project

Install dependencies:

```bash
pip install pandas numpy scikit-learn xgboost pyarrow joblib
```

Run training:

```bash
python main.py
```

---

## Notes

* The project uses only tabular features from the provided dataset.
* Predictions should be inverse-transformed with:

```python
np.expm1(predictions)
```

---

## Author

Bogdan Lomp

GitHub Repository:
[Bank Customer Savings Prediction](https://github.com/nskbogdanl/Bank-Customer-Savings-Prediction)

