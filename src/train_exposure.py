import polars as pl
import xgboost as xgb
import numpy as np
import yaml
from sklearn.metrics import accuracy_score, classification_report

CONFIG_PATH = "config.yaml"


train = pl.read_parquet("data/processed/train/nifty.parquet")
test = pl.read_parquet("data/processed/test/nifty.parquet")


with open(CONFIG_PATH,"r") as f:
    config = yaml.safe_load(f)

exp_feat = config["Exposure_Features"]

X_train = train.select(exp_feat).to_pandas()
y_train = train["Label"].to_pandas()

X_test = test.select(exp_feat).to_pandas()
y_test = test["Label"].to_pandas()


model = xgb.XGBClassifier(
    max_depth = 3,
    random_state = 42,
    n_estimators = 100,
)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

# Score
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

