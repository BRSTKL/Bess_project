# 1. Enforce cvxpy import first to prevent DLL conflicts on Windows
import cvxpy as cp
import sys
sys.path.insert(0, ".")
# Fix Windows console UTF-8 printing issues
sys.stdout.reconfigure(encoding='utf-8')

import os
import pathlib
import datetime
import pandas as pd
import numpy as np

# Import BESS and forecasting modules
import bess_arbitrage
import entsoe_pipeline
import forecast

def get_live_inference_data(today, tomorrow):
    start_date = today - datetime.timedelta(days=10)
    start_str = start_date.strftime("%Y-%m-%d")
    # Fetch prices up to today's end to avoid querying unpublished future prices
    # end date in entsoe-py is exclusive, so we use tomorrow_str for prices to get all of today
    today_end_str = tomorrow.strftime("%Y-%m-%d")
    tomorrow_end_str = (tomorrow + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    
    print(f"  [API] Fetching historical prices from {start_str} to {today_end_str}...")
    prices = entsoe_pipeline.get_day_ahead_prices(start_str, today_end_str)
    
    print(f"  [API] Fetching load forecasts from {start_str} to {tomorrow_end_str}...")
    load = entsoe_pipeline.get_load_forecast(start_str, tomorrow_end_str)
    
    print(f"  [API] Fetching wind/solar forecasts from {start_str} to {tomorrow_end_str}...")
    res = entsoe_pipeline.get_renewables_forecast(start_str, tomorrow_end_str)
    
    df = pd.concat([prices, load, res], axis=1)
    df = df.resample("1h").mean()
    return df

def get_synthetic_inference_data(today, tomorrow):
    print("  [Offline] Generating synthetic data for inference...")
    start_date = today - datetime.timedelta(days=10)
    # Convert tomorrow (date) to datetime to allow adding hourly timedelta
    tomorrow_dt = datetime.datetime.combine(tomorrow, datetime.time.min)
    idx = pd.date_range(start_date, tomorrow_dt + datetime.timedelta(hours=23), freq="h", tz="Europe/Berlin")
    rng = np.random.default_rng(42)
    hours = idx.hour.to_numpy()
    
    # Prices (synthetic)
    shape = np.array([
        45, 40, 38, 36, 35, 40, 55, 75,
        70, 55, 35, 15, -5, -10, 5, 30,
        55, 80, 110, 120, 95, 75, 60, 50
    ])
    base_price = shape[hours].astype(float)
    price = base_price + rng.normal(0, 12, len(idx))
    
    # Null out tomorrow's prices to simulate real-world prediction
    tomorrow_mask = idx.date >= tomorrow
    price[tomorrow_mask] = np.nan
    
    # Load forecast
    load_shape = np.array([
        40000, 38000, 37000, 37000, 39000, 42000, 50000, 60000,
        65000, 64000, 62000, 60000, 58000, 58000, 59000, 60000,
        62000, 66000, 70000, 72000, 68000, 62000, 52000, 45000
    ])
    load = load_shape[hours].astype(float) + rng.normal(0, 2500, len(idx))
    
    # Solar/Wind forecasts
    solar_base = np.zeros(24)
    solar_base[7:18] = np.array([500, 1500, 3000, 5000, 6500, 7000, 6500, 5000, 3000, 1500, 500])
    solar = solar_base[hours].astype(float) + rng.uniform(0, 400, len(idx))
    solar[solar < 0] = 0
    
    wind_onshore = rng.uniform(5000, 25000, len(idx)) + np.sin(np.arange(len(idx)) / 24.0) * 5000
    wind_offshore = rng.uniform(1000, 8000, len(idx)) + np.cos(np.arange(len(idx)) / 48.0) * 2000
    
    df = pd.DataFrame({
        "price_eur_mwh": price,
        "load_fc_mw": load,
        "Solar": solar,
        "Wind Onshore": wind_onshore,
        "Wind Offshore": wind_offshore
    }, index=idx)
    return df

def run_prediction_and_schedule(gamma=0.0):
    print("=" * 60)
    print("🔮 TOMORROW'S LIVE PRICE FORECAST & BESS SCHEDULING")
    print(f"  [Risk Setting] Gamma: {gamma:.2f}")
    print("=" * 60)
    
    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)
    
    # Check if API key is set
    api_key = os.environ.get("ENTSOE_API_KEY", "").strip()
    is_live = False
    
    if api_key and len(api_key) > 10:
        try:
            df = get_live_inference_data(today, tomorrow)
            is_live = True
            print("  Successfully loaded live ENTSO-E data!")
        except Exception as e:
            print(f"  [WARN] Failed to fetch live data ({e}). Falling back to synthetic data.")
            df = get_synthetic_inference_data(today, tomorrow)
    else:
        print("  ENTSOE_API_KEY environment variable not set or invalid. Running in offline/synthetic mode.")
        df = get_synthetic_inference_data(today, tomorrow)
        
    # Generate ML features
    print("  Calculating feature engineering...")
    feat = forecast.make_features(df)
    
    # Train the models on all historical data (where target is available)
    train_data = feat.dropna(subset=["target"])
    test_data = feat[feat.index.date == tomorrow]
    
    if len(test_data) == 0:
        # Fallback if dates are shifted in timezone conversion
        test_data = feat.tail(24)
        tomorrow_date = test_data.index[0].date()
        print(f"  [WARN] Exact date match empty. Using final 24 hours of dataset (Target date: {tomorrow_date}).")
    else:
        tomorrow_date = tomorrow
        
    feat_cols = [c for c in feat.columns if c != "target"]
    
    # Check if we have enough training data
    if len(train_data) < 24:
        raise ValueError(f"Not enough training data. Need at least 24 hours of history, have {len(train_data)} hours.")
        
    print(f"  Training models on {len(train_data)} hours of history...")
    
    # 1. Train LightGBM
    pred_lgb = np.zeros(len(test_data))
    if forecast.HAS_LGB:
        import lightgbm as lgb
        model_lgb = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
        )
        model_lgb.fit(train_data[feat_cols], train_data["target"])
        pred_lgb = model_lgb.predict(test_data[feat_cols])
        print("  LightGBM model trained.")
        
    # 2. Train XGBoost
    pred_xgb = np.zeros(len(test_data))
    if forecast.HAS_XGB:
        import xgboost as xgb
        model_xgb = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
        )
        model_xgb.fit(train_data[feat_cols], train_data["target"])
        pred_xgb = model_xgb.predict(test_data[feat_cols])
        print("  XGBoost model trained.")

    # 3. Train CatBoost
    pred_cat = np.zeros(len(test_data))
    if forecast.HAS_CAT:
        import catboost as cb
        model_cat = cb.CatBoostRegressor(
            iterations=400, learning_rate=0.03, depth=6,
            random_seed=42, verbose=0
        )
        model_cat.fit(train_data[feat_cols], train_data["target"])
        pred_cat = model_cat.predict(test_data[feat_cols])
        print("  CatBoost model trained.")

    # 4. Train MLP Neural Network
    pred_mlp = np.zeros(len(test_data))
    if forecast.HAS_MLP:
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        train_data_mlp = train_data.dropna()
        
        if len(train_data_mlp) >= 24:
            X_train_scaled = scaler.fit_transform(train_data_mlp[feat_cols])
            X_test_scaled = scaler.transform(test_data[feat_cols].fillna(0.0))
            
            model_mlp = MLPRegressor(
                hidden_layer_sizes=(100, 50), activation='relu', solver='adam',
                max_iter=500, random_state=42
            )
            model_mlp.fit(X_train_scaled, train_data_mlp["target"])
            pred_mlp = model_mlp.predict(X_test_scaled)
            print("  MLP Neural Network model trained.")
        else:
            print("  [WARN] MLP skipped: not enough non-NaN training data.")
        
    # 5. Create Blended Ensemble
    preds = []
    model_tags = []
    if forecast.HAS_LGB:
        preds.append(pred_lgb)
        model_tags.append("LightGBM")
    if forecast.HAS_XGB:
        preds.append(pred_xgb)
        model_tags.append("XGBoost")
    if forecast.HAS_CAT:
        preds.append(pred_cat)
        model_tags.append("CatBoost")
    if forecast.HAS_MLP:
        preds.append(pred_mlp)
        model_tags.append("MLP")
        
    if preds:
        pred_prices = sum(preds) / len(preds)
        model_name = f"Ensemble ({' + '.join(model_tags)})"
    else:
        pred_prices = test_data["price_lag24"].to_numpy()
        model_name = "Lag-24h (ML Fallback)"
        
    # Quantile prediction using spreads anchored on the ensemble
    pred_p10 = None
    pred_p90 = None
    deltas = []
    
    if forecast.HAS_LGB:
        print("  Training LightGBM Quantile models (P10 & P90)...")
        import lightgbm as lgb
        model_lgb_p10 = lgb.LGBMRegressor(
            objective="quantile", alpha=0.1,
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
        )
        model_lgb_p10.fit(train_data[feat_cols], train_data["target"])
        p10_lgb = model_lgb_p10.predict(test_data[feat_cols])
        
        model_lgb_p90 = lgb.LGBMRegressor(
            objective="quantile", alpha=0.9,
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
        )
        model_lgb_p90.fit(train_data[feat_cols], train_data["target"])
        p90_lgb = model_lgb_p90.predict(test_data[feat_cols])
        deltas.append((p10_lgb - pred_lgb, p90_lgb - pred_lgb))
        
    if forecast.HAS_CAT:
        print("  Training CatBoost Quantile models (P10 & P90)...")
        import catboost as cb
        model_cat_p10 = cb.CatBoostRegressor(
            loss_function='Quantile:alpha=0.1',
            iterations=400, learning_rate=0.03, depth=6,
            random_seed=42, verbose=0
        )
        model_cat_p10.fit(train_data[feat_cols], train_data["target"])
        p10_cat = model_cat_p10.predict(test_data[feat_cols])
        
        model_cat_p90 = cb.CatBoostRegressor(
            loss_function='Quantile:alpha=0.9',
            iterations=400, learning_rate=0.03, depth=6,
            random_seed=42, verbose=0
        )
        model_cat_p90.fit(train_data[feat_cols], train_data["target"])
        p90_cat = model_cat_p90.predict(test_data[feat_cols])
        deltas.append((p10_cat - pred_cat, p90_cat - pred_cat))
        
    if deltas:
        delta_p10 = sum(d[0] for d in deltas) / len(deltas)
        delta_p90 = sum(d[1] for d in deltas) / len(deltas)
        pred_p10 = pred_prices + delta_p10
        pred_p90 = pred_prices + delta_p90
    else:
        std_dev = train_data["target"].std() if len(train_data) > 0 else 15.0
        pred_p10 = pred_prices - 1.28 * std_dev
        pred_p90 = pred_prices + 1.28 * std_dev
        
    print(f"\n🔮 Predicted Hourly Prices for tomorrow ({tomorrow_date}) using {model_name}:")
    print(f"  {'Hour':<6} {'P10 Low':<12} {'Expected':<12} {'P90 High':<12}")
    print("-" * 48)
    for i, (dt, val) in enumerate(test_data.index.to_series().items()):
        print(f"  {dt.strftime('%H:%M')}: €{pred_p10[i]:>9.2f} / €{pred_prices[i]:>9.2f} / €{pred_p90[i]:>9.2f}")
        
    # Run Battery Optimization on predicted prices
    print("\n🔋 Running BESS Arbitrage optimization on predicted prices...")
    bat = bess_arbitrage.Battery(capacity_mwh=2.0, power_mw=1.0, rte=0.88, cycle_cost_eur_mwh=4.0)
    res = bess_arbitrage.optimize_day(
        pred_prices, bat,
        prices_p10=pred_p10,
        prices_p90=pred_p90,
        gamma=gamma
    )
    
    if res["status"] in ("optimal", "optimal_inaccurate"):
        print("\n📈 TOMORROW'S OPTIMAL BATTERY SCHEDULE PLAN:")
        schedule = pd.DataFrame({
            "Pred_Price": pred_prices,
            "Pred_P10": pred_p10,
            "Pred_P90": pred_p90,
            "Charge_MW": res["charge"],
            "Discharge_MW": res["discharge"],
            "SOC_MWh": res["soc"][1:]
        }, index=test_data.index)
        
        schedule["Action"] = "Idle"
        schedule.loc[schedule["Charge_MW"] > 1e-3, "Action"] = "CHARGE 🔌"
        schedule.loc[schedule["Discharge_MW"] > 1e-3, "Action"] = "DISCHARGE ⚡"
        
        print(schedule[["Pred_Price", "Pred_P10", "Pred_P90", "Charge_MW", "Discharge_MW", "SOC_MWh", "Action"]].round(3).to_string())
        
        expected_profit = res["profit"]
        print("\n💰 FINANCIAL EXPECTATION:")
        print(f"  Expected Profit tomorrow: € {expected_profit:,.2f}")
        print(f"  Estimated Cycles: {(schedule['Discharge_MW'].sum() / bat.E):.2f} Cycles")
        
        # Save to csv
        out_path = pathlib.Path("tomorrow_schedule.csv")
        schedule.to_csv(out_path)
        print(f"\nSaved tomorrow's schedule to: {out_path.absolute()}")
    else:
        print(f"\n❌ Optimization failed! Status: {res['status']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict tomorrow's prices and BESS arbitrage schedule.")
    parser.add_argument("--gamma", type=float, default=0.0, help="Risk aversion coefficient (0.0 to 0.95)")
    args = parser.parse_args()
    
    run_prediction_and_schedule(gamma=args.gamma)
