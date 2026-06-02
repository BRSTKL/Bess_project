"""
Forecast Layer - Baseline + LightGBM, Walk-Forward, Capture Rate
================================================================
Input: DataFrame from entsoe_pipeline.build_dataset()
Usage: python forecast.py
"""

import cvxpy as cp
import numpy as np
import pandas as pd
from bess_arbitrage import Battery, optimize_day

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def make_features(df):
    X = pd.DataFrame(index=df.index)
    X["hour"] = df.index.hour
    X["dow"] = df.index.dayofweek
    X["month"] = df.index.month
    X["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
    
    # Public Holidays in Germany
    import holidays
    de_holidays = holidays.Germany(years=list(df.index.year.unique()))
    X["is_holiday"] = df.index.map(lambda dt: 1 if dt.date() in de_holidays else 0)

    if "load_fc_mw" in df:
        X["load_fc"] = df["load_fc_mw"]
    for col in ["Solar", "Wind Onshore", "Wind Offshore"]:
        if col in df:
            X[col.replace(" ", "_").lower()] = df[col]
            
    res_cols = [c for c in ["Solar", "Wind Onshore", "Wind Offshore"] if c in df]
    if res_cols and "load_fc_mw" in df:
        X["res_ratio"] = df[res_cols].sum(axis=1) / df["load_fc_mw"]
        # Net Load (Tüketim - Yenilenebilir Enerji Üretimi)
        X["net_load"] = df["load_fc_mw"] - df[res_cols].sum(axis=1)
        
    p = df["price_eur_mwh"]
    X["price_lag24"] = p.shift(24)
    X["price_lag48"] = p.shift(48)
    X["price_lag168"] = p.shift(168)
    
    # Rolling Statistics (shifted by 24h to avoid lookahead bias during inference)
    X["price_roll24"] = p.shift(24).rolling(24).mean()
    X["price_roll24_std"] = p.shift(24).rolling(24).std()
    X["price_roll168_mean"] = p.shift(24).rolling(168).mean()
    X["price_roll168_std"] = p.shift(24).rolling(168).std()
    
    X["target"] = p
    return X


def baseline_forecast(df):
    return df["price_eur_mwh"].shift(168).rename("pred_baseline")


def lgbm_walkforward(feat, train_end):
    if not HAS_LGB:
        raise RuntimeError("lightgbm not installed: pip install lightgbm")
    data = feat.dropna()
    cut = pd.Timestamp(train_end, tz=data.index.tz)
    train, test = data[data.index < cut], data[data.index >= cut]
    if len(test) == 0:
        raise ValueError("Test set empty - train_end must be before data end.")
    feat_cols = [c for c in data.columns if c != "target"]
    model = lgb.LGBMRegressor(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    )
    model.fit(train[feat_cols], train["target"])
    pred = model.predict(test[feat_cols])
    return pd.Series(pred, index=test.index, name="pred_lgbm")


def xgboost_walkforward(feat, train_end):
    if not HAS_XGB:
        raise RuntimeError("xgboost not installed: pip install xgboost")
    data = feat.dropna()
    cut = pd.Timestamp(train_end, tz=data.index.tz)
    train, test = data[data.index < cut], data[data.index >= cut]
    if len(test) == 0:
        raise ValueError("Test set empty - train_end must be before data end.")
    feat_cols = [c for c in data.columns if c != "target"]
    model = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    model.fit(train[feat_cols], train["target"])
    pred = model.predict(test[feat_cols])
    return pd.Series(pred, index=test.index, name="pred_xgb")


def ensemble_walkforward(feat, train_end):
    if HAS_LGB and HAS_XGB:
        pred_lgb = lgbm_walkforward(feat, train_end)
        pred_xgb = xgboost_walkforward(feat, train_end)
        return pd.Series(0.5 * pred_lgb + 0.5 * pred_xgb, index=pred_lgb.index, name="pred_ensemble")
    elif HAS_LGB:
        return lgbm_walkforward(feat, train_end).rename("pred_ensemble")
    elif HAS_XGB:
        return xgboost_walkforward(feat, train_end).rename("pred_ensemble")
    else:
        raise RuntimeError("Neither lightgbm nor xgboost is installed.")


def backtest_with_forecast(real_price, pred_price, bat, dt_h=1.0):
    common = real_price.index.intersection(pred_price.index)
    real_price, pred_price = real_price[common], pred_price[common]
    daily_profit = []
    for day, idx in real_price.groupby(real_price.index.date).groups.items():
        rp = real_price.loc[idx].to_numpy()
        pp = pred_price.loc[idx].to_numpy()
        if len(rp) < 2 or np.isnan(pp).any():
            continue
        plan = optimize_day(pp, bat, dt_h)
        if plan["status"] not in ("optimal", "optimal_inaccurate"):
            continue
        real_profit = (rp @ plan["discharge"] - rp @ plan["charge"]
                       - bat.cycle_cost * plan["discharge"].sum())
        daily_profit.append(real_profit)
    arr = np.array(daily_profit)
    return {"total": float(arr.sum()),
            "daily_mean": float(arr.mean()) if len(arr) else 0.0,
            "n_days": len(arr)}


def evaluate(df, train_end=None):
    bat = Battery(capacity_mwh=2.0, power_mw=1.0, rte=0.88, cycle_cost_eur_mwh=4.0)
    real = df["price_eur_mwh"].dropna()
    if train_end is None:
        train_end = str(real.index[int(len(real) * 0.6)].date())

    pf = backtest_with_forecast(real, real, bat)
    base_pred = baseline_forecast(df)
    bl = backtest_with_forecast(real, base_pred, bat)

    print("=" * 60)
    print("FORECAST EVALUATION  (test start: %s)" % train_end)
    print("=" * 60)
    print("%-22s%12s%12s" % ("Strategy", "Total EUR", "Capture %"))
    print("-" * 46)
    print("%-22s%12.0f%11.0f%%" % ("Perfect foresight", pf["total"], 100.0))
    print("%-22s%12.0f%11.0f%%" % ("Baseline (lag168)", bl["total"],
                                    100 * bl["total"] / pf["total"]))

    if HAS_LGB or HAS_XGB:
        feat = make_features(df)
        mask = real.index >= pd.Timestamp(train_end, tz=real.index.tz)
        pf_t = backtest_with_forecast(real[mask], real[mask], bat)
        bl_t = backtest_with_forecast(real[mask], base_pred[mask], bat)
        
        print("\n  [Training Machine Learning Models...]")
        
        preds = {}
        if HAS_LGB:
            print("  Training LightGBM...")
            preds["LightGBM"] = lgbm_walkforward(feat, train_end)
        if HAS_XGB:
            print("  Training XGBoost...")
            preds["XGBoost"] = xgboost_walkforward(feat, train_end)
        if HAS_LGB and HAS_XGB:
            print("  Blending Ensemble...")
            preds["Ensemble (LGBM+XGB)"] = ensemble_walkforward(feat, train_end)
            
        print("\n  [Test window only - fair comparison]")
        print("%-25s%12s%12s" % ("Strategy", "Total EUR", "Capture %"))
        print("-" * 49)
        print("%-25s%12.0f%11.0f%%" % ("Perfect (test)", pf_t["total"], 100.0))
        print("%-25s%12.0f%11.0f%%" % ("Baseline (test)", bl_t["total"],
                                        100 * bl_t["total"] / pf_t["total"]))
                                        
        for name, pred in preds.items():
            ml_t = backtest_with_forecast(real[mask], pred, bat)
            print("%-25s%12.0f%11.0f%%" % (name + " (test)", ml_t["total"],
                                            100 * ml_t["total"] / pf_t["total"]))
    else:
        print("\n  (ML skipped - pip install lightgbm or xgboost to enable)")


if __name__ == "__main__":
    import pathlib
    cands = sorted(pathlib.Path("./entsoe_cache").glob("dataset_*.parquet"))
    if not cands:
        raise SystemExit("Run entsoe_pipeline.py first (it creates dataset_*.parquet).")
    df = pd.read_parquet(cands[-1])
    print("Loaded: %s  (%d hours)\n" % (cands[-1].name, len(df)))
    evaluate(df)
