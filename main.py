import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import root_mean_squared_error
import json

# --- Paths ---
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "input_data"
OUTPUT_DIR = BASE_DIR / "output_data"

TRAIN_PATH = INPUT_DIR / "train_main_df.parquet"
TARGET_PATH = INPUT_DIR / "train_target.csv"
TEST_PATH = INPUT_DIR / "test_main_df.parquet"

def remove_highly_correlated_features(df, threshold=0.95):
    corr_matrix = df.select_dtypes(include=['int', 'float']).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    print(f"Removed correlated features (>|{threshold}|): {len(to_drop)} -> {to_drop}")
    return df.drop(columns=to_drop), to_drop


# --- Load data ---
train = pd.read_parquet(TRAIN_PATH)
target = pd.read_csv(TARGET_PATH)
test = pd.read_parquet(TEST_PATH)


# --- Missing value features ---
train['missing_fraction'] = train.isna().mean(axis=1)
test['missing_fraction'] = test.isna().mean(axis=1)
train['many_missing'] = (train['missing_fraction'] > 0.5).astype(int)
test['many_missing'] = (test['missing_fraction'] > 0.5).astype(int)

# --- Remove highly correlated features ---
train, dropped_corr = remove_highly_correlated_features(train, threshold=0.95)
test = test.drop(columns=dropped_corr, errors='ignore')

# Save dropped features
with open(OUTPUT_DIR / 'dropped_corr.json', 'w') as f:
    json.dump(dropped_corr, f)

# --- Remove anomalies ---
numeric_features = train.select_dtypes(include=['int', 'float']).columns
iso = IsolationForest(contamination=0.01, random_state=42)
mask = iso.fit_predict(train[numeric_features]) == 1
train = train.loc[mask].reset_index(drop=True)
target = target.loc[mask].reset_index(drop=True)

# --- Merge train and test ---
train['is_train'] = 1
test['is_train'] = 0
full_df = pd.concat([train, test], axis=0).reset_index(drop=True)

# --- Categorical features ---
cat_cols = full_df.select_dtypes(include='object').columns
full_df[cat_cols] = full_df[cat_cols].fillna('missing')
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
full_df[cat_cols] = encoder.fit_transform(full_df[cat_cols])
joblib.dump(encoder, OUTPUT_DIR / 'ordinal_encoder.pkl')

# --- Fill remaining missing values ---
full_df = full_df.fillna(0)

# --- Remove zero-variance features ---
is_train_col = full_df['is_train'].values
features = full_df.drop(columns=['is_train'])
selector = VarianceThreshold(threshold=0.0)
selected = selector.fit_transform(features)
selected_cols = features.columns[selector.get_support()]
full_df = pd.DataFrame(selected, columns=selected_cols)
full_df['is_train'] = is_train_col
joblib.dump(selector, OUTPUT_DIR / 'variance_selector.pkl')

# --- Split back into train and test ---
train = full_df[full_df['is_train'] == 1].drop(columns=['is_train']).reset_index(drop=True)
test = full_df[full_df['is_train'] == 0].drop(columns=['is_train']).reset_index(drop=True)

# --- Drop target column if present ---
if 'target' in train.columns:
    train = train.drop(columns=['target'])

X_train = train
y_train = np.log1p(target['target'])
X_test = test

# --- XGBoost model ---
model = XGBRegressor(
    n_estimators=5000,
    max_depth=8,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

# --- Cross-validation ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    model.fit(X_tr, y_tr)
    preds = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, preds)
    rmse_scores.append(rmse)
    print(f"Fold {fold} RMSE: {rmse:.4f}")

print("Mean RMSE:", np.mean(rmse_scores))

# --- Final training and model saving ---
model.fit(X_train, y_train)
joblib.dump(model, OUTPUT_DIR / 'xgb_model.pkl')