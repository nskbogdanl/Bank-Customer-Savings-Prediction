import pandas as pd
import numpy as np
import logging
from pathlib import Path
from flaml import AutoML
from sklearn.model_selection import KFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error
from itertools import combinations

# Paths
BASE_DIR = Path.home() / "Programms" / "Shift"
INPUT_DIR = BASE_DIR / "input_data"
OUTPUT_DIR = BASE_DIR
LOG_DIR = BASE_DIR

TRAIN_PATH = INPUT_DIR / "train_main_df.parquet"
TARGET_PATH = INPUT_DIR / "train_target.csv"
TEST_PATH = INPUT_DIR / "test_main_df.parquet"
SUBMISSION_PATH = OUTPUT_DIR / "submission_flaml.csv"
FLAML_LOG_PATH = LOG_DIR / "flaml.log"
DETAILED_LOG_PATH = LOG_DIR / "flaml_detailed.log"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(DETAILED_LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def remove_highly_correlated_features(df, threshold=0.95):
    corr_matrix = df.select_dtypes(include=['int', 'float']).corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    logger.info(f"Removed correlated features (>|{threshold}|): {len(to_drop)} -> {to_drop}")
    return df.drop(columns=to_drop), to_drop

logger.info("Loading data...")
train = pd.read_parquet(TRAIN_PATH)
target = pd.read_csv(TARGET_PATH)
test = pd.read_parquet(TEST_PATH)

logger.info("Creating missing value features...")
train['missing_fraction'] = train.isna().mean(axis=1)
test['missing_fraction'] = test.isna().mean(axis=1)
train['many_missing'] = (train['missing_fraction'] > 0.5).astype(int)
test['many_missing'] = (test['missing_fraction'] > 0.5).astype(int)
train['n_missing'] = train.isna().sum(axis=1)
test['n_missing'] = test.isna().sum(axis=1)

logger.info("Creating row statistics...")
numeric_train = train.select_dtypes(include=['int', 'float'])
numeric_test = test.select_dtypes(include=['int', 'float'])

for df, numeric in [(train, numeric_train), (test, numeric_test)]:
    df['row_mean'] = numeric.mean(axis=1)
    df['row_std'] = numeric.std(axis=1)
    df['row_max'] = numeric.max(axis=1)
    df['row_min'] = numeric.min(axis=1)
    df['non_zero_count'] = (numeric != 0).sum(axis=1)
    df['zero_fraction'] = (numeric == 0).mean(axis=1)

logger.info("Removing correlated features...")
train, dropped_corr = remove_highly_correlated_features(train, threshold=0.95)
test = test.drop(columns=dropped_corr, errors='ignore')

logger.info("Removing anomalies using IsolationForest...")
numeric_features = train.select_dtypes(include=['int', 'float']).columns
iso = IsolationForest(contamination=0.01, random_state=42)
mask = iso.fit_predict(train[numeric_features]) == 1
train = train.loc[mask].reset_index(drop=True)
target = target.loc[mask].reset_index(drop=True)

logger.info("Creating feature interactions (pairwise division)...")
numeric_cols = train.select_dtypes(include=['int', 'float']).columns[:20]
for col1, col2 in combinations(numeric_cols, 2):
    train[f'{col1}_div_{col2}'] = np.where(train[col2] != 0, train[col1] / train[col2], 0)
    test[f'{col1}_div_{col2}'] = np.where(test[col2] != 0, test[col1] / test[col2], 0)

logger.info("Binning numeric features...")
train_bins = {}
test_bins = {}
for col in numeric_cols[:10]:
    train_bins[f'{col}_bin'] = pd.qcut(train[col], q=4, labels=False, duplicates='drop')
    test_bins[f'{col}_bin'] = pd.qcut(test[col], q=4, labels=False, duplicates='drop')
train = pd.concat([train, pd.DataFrame(train_bins, index=train.index)], axis=1)
test = pd.concat([test, pd.DataFrame(test_bins, index=test.index)], axis=1)
train = train.copy()
test = test.copy()

logger.info("Encoding categorical features...")
train['is_train'] = 1
test['is_train'] = 0
full_df = pd.concat([train, test], axis=0).reset_index(drop=True)

cat_cols = full_df.select_dtypes(include='object').columns
full_df[cat_cols] = full_df[cat_cols].fillna('missing')
encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
full_df[cat_cols] = encoder.fit_transform(full_df[cat_cols])

logger.info("Filling remaining missing values with zeros...")
full_df = full_df.fillna(0)

logger.info("Removing zero-variance features...")
is_train_col = full_df['is_train'].values
features = full_df.drop(columns=['is_train'])
selector = VarianceThreshold(threshold=0.0)
selected = selector.fit_transform(features)
selected_cols = features.columns[selector.get_support()]
full_df = pd.DataFrame(selected, columns=selected_cols)
full_df['is_train'] = is_train_col

logger.info("Splitting back into train and test...")
train = full_df[full_df['is_train'] == 1].drop(columns=['is_train']).reset_index(drop=True)
test = full_df[full_df['is_train'] == 0].drop(columns=['is_train']).reset_index(drop=True)

if 'target' in train.columns:
    train = train.drop(columns=['target'])

X_train = train
y_train = np.log1p(target['target'])
X_test = test

logger.info("Starting AutoML with FLAML...")
automl = AutoML()

settings = {
    "time_budget": 1200,
    "metric": 'rmse',
    "task": 'regression',
    "log_file_name": str(FLAML_LOG_PATH),
    "eval_method": 'cv',
    "n_splits": 5,
    "seed": 42,
    "verbose": 3,
    "estimator_list": ["lgbm", "xgboost", "rf", "extra_tree"]
}

automl.fit(X_train=X_train, y_train=y_train, **settings)

logger.info("Best model: %s", automl.model.estimator)
logger.info("Best RMSE (log space): %.5f", automl.best_loss)

logger.info("Making predictions and applying inverse log transform...")
test_predict_log = automl.predict(X_test)
test_full_predict = np.expm1(test_predict_log)
test_full_predict[test_full_predict < 0] = 0

logger.info("Saving predictions...")
submission = pd.DataFrame({
    'id': pd.read_parquet(TEST_PATH)['id'],
    'target': test_full_predict
})
submission.to_csv(SUBMISSION_PATH, index=False)
logger.info("Done! Predictions saved to %s", SUBMISSION_PATH)