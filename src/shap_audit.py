"""
shap_audit.py 
==============
- Assets: Nifty, Gold, USDINR
"""

import joblib
import pandas as pd
import polars as pl
import os
import shap
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# Configuration
ASSETS = ["nifty", "gold", "usdinr"]

def run_audit(asset):
    model_path = f"models/meta/{asset}_xgb_meta.joblib"
    data_path = f"data/processed/test/{asset}.parquet"

    print(f"\n{'='*60}")
    print(f"  ASSET: {asset.upper()}")
    print(f"{'='*60}")

    model = joblib.load(model_path)
    test = pl.read_parquet(data_path)

    X = test.select(model.feature_names_in_).to_pandas()
    y = test["Meta_Label"].to_pandas()
    print(f"   Analyzing {len(X)} test samples with {len(X.columns)} features...")

    # 3. SHAP Interpretation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # 4. Global Importance 
    vals = np.abs(shap_values).mean(0)
    feature_importance = pd.DataFrame(list(zip(X.columns, vals)), columns=['Feature', 'SHAP_Value'])
    feature_importance.sort_values(by=['SHAP_Value'], ascending=False, inplace=True)

    print("\n   Top SHAP Alpha Drivers (Impact on Win Probability):")
    print(f"   {'Rank':4s} | {'Feature':25s} | {'SHAP Value':10s}")
    print(f"   {'-'*45}")
    for i, (_, row) in enumerate(feature_importance.head(12).iterrows()):
        print(f"    #{i+1:3d} | {row['Feature']:25s} | {row['SHAP_Value']:.6f}")

    # 5. Domain Audit
    vix_drivers = feature_importance[feature_importance['Feature'].str.contains('VIX', case=False)]
    rs_drivers  = feature_importance[feature_importance['Feature'].str.contains('RS', case=False)]
    
    print("\n   Domain Distribution")
    print(f"   VIX Presence: {vix_drivers['SHAP_Value'].sum():.6f} (Top: {vix_drivers.iloc[0]['Feature'] if not vix_drivers.empty else 'None'})")
    print(f"   RS Presence:  {rs_drivers['SHAP_Value'].sum():.6f} (Top: {rs_drivers.iloc[0]['Feature'] if not rs_drivers.empty else 'None'})")

if __name__ == "__main__":
    try:
        for asset in ASSETS:
            run_audit(asset)
    except Exception as e:
        print(f"   [ERROR] SHAP audit failed: {e}")
