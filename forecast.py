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

try:
    import catboost as cb
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    from sklearn.neural_network import MLPRegressor
    HAS_MLP = True
except ImportError:
    HAS_MLP = False


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


def lgbm_walkforward(feat, train_end, quantile=None):
    if not HAS_LGB:
        raise RuntimeError("lightgbm not installed: pip install lightgbm")
    data = feat.dropna()
    cut = pd.Timestamp(train_end, tz=data.index.tz)
    train, test = data[data.index < cut], data[data.index >= cut]
    if len(test) == 0:
        raise ValueError("Test set empty - train_end must be before data end.")
    feat_cols = [c for c in data.columns if c != "target"]
    
    if quantile is not None:
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=quantile,
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
        )
    else:
        model = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
        )
    model.fit(train[feat_cols], train["target"])
    pred = model.predict(test[feat_cols])
    name = f"pred_lgbm_q{int(quantile*100)}" if quantile is not None else "pred_lgbm"
    return pd.Series(pred, index=test.index, name=name)


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


def catboost_walkforward(feat, train_end, quantile=None):
    if not HAS_CAT:
        raise RuntimeError("catboost not installed: pip install catboost")
    data = feat.dropna()
    cut = pd.Timestamp(train_end, tz=data.index.tz)
    train, test = data[data.index < cut], data[data.index >= cut]
    if len(test) == 0:
        raise ValueError("Test set empty - train_end must be before data end.")
    feat_cols = [c for c in data.columns if c != "target"]
    
    if quantile is not None:
        model = cb.CatBoostRegressor(
            loss_function=f'Quantile:alpha={quantile}',
            iterations=400, learning_rate=0.03, depth=6,
            random_seed=42, verbose=0
        )
    else:
        model = cb.CatBoostRegressor(
            iterations=400, learning_rate=0.03, depth=6,
            random_seed=42, verbose=0
        )
    model.fit(train[feat_cols], train["target"])
    pred = model.predict(test[feat_cols])
    name = f"pred_cat_q{int(quantile*100)}" if quantile is not None else "pred_cat"
    return pd.Series(pred, index=test.index, name=name)


def mlp_walkforward(feat, train_end):
    if not HAS_MLP:
        raise RuntimeError("scikit-learn not installed.")
    data = feat.dropna()
    cut = pd.Timestamp(train_end, tz=data.index.tz)
    train, test = data[data.index < cut], data[data.index >= cut]
    if len(test) == 0:
        raise ValueError("Test set empty - train_end must be before data end.")
    feat_cols = [c for c in data.columns if c != "target"]
    
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feat_cols])
    X_test = scaler.transform(test[feat_cols])
    
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
        max_iter=500, random_state=42
    )
    model.fit(X_train, train["target"])
    pred = model.predict(X_test)
    return pd.Series(pred, index=test.index, name="pred_mlp")


def ensemble_walkforward(feat, train_end, quantile=None):
    if quantile is None:
        preds = []
        if HAS_LGB:
            preds.append(lgbm_walkforward(feat, train_end))
        if HAS_XGB:
            preds.append(xgboost_walkforward(feat, train_end))
        if HAS_CAT:
            preds.append(catboost_walkforward(feat, train_end))
        if HAS_MLP:
            preds.append(mlp_walkforward(feat, train_end))
            
        if not preds:
            raise RuntimeError("No forecasting models are installed/available.")
            
        base_series = preds[0]
        if len(preds) > 1:
            combined = sum(p.values for p in preds) / len(preds)
            return pd.Series(combined, index=base_series.index, name="pred_ensemble")
        return base_series.rename("pred_ensemble")
    else:
        # Quantile prediction anchoring spreads on ensemble median
        pred_base = ensemble_walkforward(feat, train_end, quantile=None)
        
        deltas = []
        if HAS_LGB:
            pred_lgb_mean = lgbm_walkforward(feat, train_end, quantile=None)
            pred_lgb_q = lgbm_walkforward(feat, train_end, quantile=quantile)
            deltas.append(pred_lgb_q - pred_lgb_mean)
        if HAS_CAT:
            pred_cat_mean = catboost_walkforward(feat, train_end, quantile=None)
            pred_cat_q = catboost_walkforward(feat, train_end, quantile=quantile)
            deltas.append(pred_cat_q - pred_cat_mean)
            
        if deltas:
            delta = sum(d.values for d in deltas) / len(deltas)
        else:
            std_val = feat.dropna()["target"].std()
            sign = 1.0 if quantile > 0.5 else -1.0
            delta = sign * 1.28 * std_val
            
        pred_q = pred_base + delta
        name = f"pred_ensemble_q{int(quantile*100)}"
        return pd.Series(pred_q, index=pred_base.index, name=name)


def backtest_with_forecast(real_price, pred_price, bat, dt_h=1.0, use_degradation=False, use_multi_market=False, fcr_price=18.0, idm_spread=10.0):
    common = real_price.index.intersection(pred_price.index)
    real_price, pred_price = real_price[common], pred_price[common]
    daily_profit = []
    for day, idx in real_price.groupby(real_price.index.date).groups.items():
        rp = real_price.loc[idx].to_numpy()
        pp = pred_price.loc[idx].to_numpy()
        if len(rp) < 2 or np.isnan(pp).any():
            continue
            
        T = len(pp)
        pp_idm = pp + idm_spread * np.sin(np.arange(T) * 2 * np.pi / 24)
        pp_fcr = np.full(T, fcr_price)
        
        plan = optimize_day(
            pp, bat, dt_h, 
            use_degradation=use_degradation,
            use_multi_market=use_multi_market,
            idm_prices=pp_idm,
            fcr_prices=pp_fcr
        )
        if plan["status"] not in ("optimal", "optimal_inaccurate"):
            continue
            
        if use_multi_market:
            real_dam = rp @ plan["d_dam"] - rp @ plan["c_dam"]
            rp_idm = rp + idm_spread * np.sin(np.arange(T) * 2 * np.pi / 24)
            real_idm = rp_idm @ plan["d_idm"] - rp_idm @ plan["c_idm"]
            real_fcr = pp_fcr @ plan["r_fcr"]
            
            gross_val = real_dam + real_idm + real_fcr
            cycle_wear = bat.cycle_cost * np.sum(plan["d_dam"] + plan["d_idm"])
        else:
            gross_val = rp @ plan["discharge"] - rp @ plan["charge"]
            cycle_wear = bat.cycle_cost * plan["discharge"].sum()
            
        if use_degradation:
            soc_threshold = 0.8 * bat.E
            soc_penalty_val = bat.degradation_coef_soc * np.sum(np.maximum(0, plan["soc"][1:] - soc_threshold))
            if use_multi_market:
                power_penalty_val = bat.degradation_coef_power * (np.sum((plan["c_dam"]+plan["c_idm"])**2) + np.sum((plan["d_dam"]+plan["d_idm"])**2))
            else:
                power_penalty_val = bat.degradation_coef_power * (np.sum(plan["charge"]**2) + np.sum(plan["discharge"]**2))
            deg_cost = soc_penalty_val + power_penalty_val
        else:
            deg_cost = 0.0

        real_profit = gross_val - cycle_wear - deg_cost
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

    if HAS_LGB or HAS_XGB or HAS_CAT or HAS_MLP:
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
        if HAS_CAT:
            print("  Training CatBoost...")
            preds["CatBoost"] = catboost_walkforward(feat, train_end)
        if HAS_MLP:
            print("  Training MLP Neural Network...")
            preds["MLP Neural Network"] = mlp_walkforward(feat, train_end)
            
        model_tags = []
        if HAS_LGB: model_tags.append("LGB")
        if HAS_XGB: model_tags.append("XGB")
        if HAS_CAT: model_tags.append("CAT")
        if HAS_MLP: model_tags.append("MLP")
        
        ensemble_label = f"Ensemble ({'+'.join(model_tags)})"
        print(f"  Blending {ensemble_label}...")
        preds[ensemble_label] = ensemble_walkforward(feat, train_end)
            
        print("\n  [Test window only - fair comparison]")
        print("%-28s%12s%12s" % ("Strategy", "Total EUR", "Capture %"))
        print("-" * 52)
        print("%-28s%12.0f%11.0f%%" % ("Perfect (test)", pf_t["total"], 100.0))
        print("%-28s%12.0f%11.0f%%" % ("Baseline (test)", bl_t["total"],
                                        100 * bl_t["total"] / pf_t["total"]))
                                        
        for name, pred in preds.items():
            ml_t = backtest_with_forecast(real[mask], pred, bat)
            print("%-28s%12.0f%11.0f%%" % (name + " (test)", ml_t["total"],
                                            100 * ml_t["total"] / pf_t["total"]))
    else:
        print("\n  (ML skipped - no machine learning packages installed)")


if __name__ == "__main__":
    import pathlib
    cands = sorted(pathlib.Path("./entsoe_cache").glob("dataset_*.parquet"))
    if not cands:
        raise SystemExit("Run entsoe_pipeline.py first (it creates dataset_*.parquet).")
    df = pd.read_parquet(cands[-1])
    print("Loaded: %s  (%d hours)\n" % (cands[-1].name, len(df)))
    evaluate(df)
