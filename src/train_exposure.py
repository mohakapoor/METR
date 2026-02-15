import polars as pl
import xgboost as xgb
import numpy as np
import yaml
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import TimeSeriesSplit,GridSearchCV


train = pl.read_parquet("nifty_train.parquet")
test = pl.read_parquet("nifty_test.parquet")


exp_feat = ["Ret_1d","Ret_3d","Ret_5d","Ret_20d","Vol_5d","Vol_20d","Vol_Ratio","MA_Ratio","Close_Pos_Range","Intraday_Return","Price_vs_MA20",'MACD_Hist','RSI','ROC_10','BB_Pct',    "ATR_Pct","Ret_1d_Lag1","Ret_1d_Lag2","Ret_1d_Lag3"]


X_train = train.select(exp_feat).to_pandas()
y_train = train["Label"].to_pandas()

X_test = test.select(exp_feat).to_pandas()
y_test = test["Label"].to_pandas()





param_grid = {
    'max_depth' : [2,3,4,5],
    'learning_rate' : [0.01,0.03,0.05,0.1],
    'n_estimators' : [100,150,200,250,300],
    'subsample' : [1.0]
}

tscv = TimeSeriesSplit(n_splits = 5)

search = GridSearchCV(
    estimator=xgb.XGBClassifier(random_state=42, device='cuda'),  # Use GPU
    param_grid=param_grid,
    scoring = 'roc_auc',
    cv =tscv,
    verbose =3,
    n_jobs = 1, # <--- Limit CPU threads; GPU does the work
)


search.fit(X_train, y_train)
print(f"Best Score: {search.best_score_:.4f}")
print(f"Best Params: {search.best_params_}")
# 5. Use the Best Model for Final Test
best_model = search.best_estimator_
preds = best_model.predict(X_test)


accuracy = accuracy_score(y_test, preds)
print(f"Test Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, preds))




