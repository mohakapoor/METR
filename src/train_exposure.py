import polars as pl
import xgboost as xgb
import numpy as np
import yaml
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit

CONFIG_PATH = "config.yaml"



train = pl.read_parquet("data/processed/train/nifty.parquet")
test = pl.read_parquet("data/processed/test/nifty.parquet")


with open(CONFIG_PATH,"r") as f:
    config = yaml.safe_load(f)

exp_feat = config["Exposure_Features"]
splits = config["N_Splits"]

X_train = train.select(exp_feat).to_pandas()
y_train = train["Label"].to_pandas()

X_test = test.select(exp_feat).to_pandas()
y_test = test["Label"].to_pandas()

tscv = TimeSeriesSplit(n_splits = splits)



model = xgb.XGBClassifier(
    max_depth = 3,
    random_state = 42,
    n_estimators = 100,
)

for fold,(train_index,val_index) in enumerate(tscv.split(X_train)):
    X_t,X_v = X_train.iloc[train_index],X_train.iloc[val_index]
    y_t,y_v = y_train.iloc[train_index],y_train.iloc[val_index]

    model = xgb.XGBClassifier(
        max_depth = 3,
        random_state = 42,
        n_estimators = 100,
    )

    model.fit(X_t,y_t)

    preds = model.predict(X_v)


    accuracy = accuracy_score(y_v, preds)
    print(f"Test Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_v, preds))



