# 1. Enforce cvxpy import first to prevent DLL conflicts on Windows
import cvxpy as cp
import sys
sys.path.insert(0, ".")

import os
import pathlib
import datetime
import pandas as pd
import numpy as np
import streamlit as st

# Import project files
import bess_arbitrage
import entsoe_pipeline
import forecast

# App layout & styling configuration
st.set_page_config(
    page_title="BESS Arbitrage & Forecasting Dashboard",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
    .main {
        background-color: #0f1116;
        color: #e2e8f0;
    }
    .stMetric {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #334155;
    }
    .stMetric label {
        color: #94a3b8 !important;
        font-weight: 600;
    }
    .stMetric div {
        color: #38bdf8 !important;
    }
    h1, h2, h3 {
        color: #f1f5f9;
        font-family: 'Inter', sans-serif;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to generate mock ENTSO-E data if offline
def generate_synthetic_entsoe_data(start_date, end_date):
    idx = pd.date_range(start_date, end_date, freq="h", tz="Europe/Berlin")
    rng = np.random.default_rng(42)
    hours = idx.hour.to_numpy()
    
    # Synthetic prices
    shape = np.array([
        45, 40, 38, 36, 35, 40, 55, 75,
        70, 55, 35, 15, -5, -10, 5, 30,
        55, 80, 110, 120, 95, 75, 60, 50
    ])
    base_price = shape[hours].astype(float)
    price_noise = rng.normal(0, 12, len(idx))
    price = base_price + price_noise

    # Load forecast
    load_shape = np.array([
        40000, 38000, 37000, 37000, 39000, 42000, 50000, 60000,
        65000, 64000, 62000, 60000, 58000, 58000, 59000, 60000,
        62000, 66000, 70000, 72000, 68000, 62000, 52000, 45000
    ])
    load = load_shape[hours].astype(float) + rng.normal(0, 2500, len(idx))

    # Solar forecast
    solar_base = np.zeros(24)
    solar_base[7:18] = np.array([500, 1500, 3000, 5000, 6500, 7000, 6500, 5000, 3000, 1500, 500])
    solar = solar_base[hours].astype(float) + rng.uniform(0, 400, len(idx))
    solar[solar < 0] = 0

    # Wind forecast
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

def generate_synthetic_inference_data_in_app(today, tomorrow):
    start_date = today - datetime.timedelta(days=10)
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

def predict_tomorrow_prices_in_app(df, today, tomorrow):
    feat = forecast.make_features(df)
    train_data = feat.dropna(subset=["target"])
    test_data = feat[feat.index.date == tomorrow]
    
    if len(test_data) == 0:
        test_data = feat.tail(24)
        
    feat_cols = [c for c in feat.columns if c != "target"]
    
    pred_lgb = np.zeros(len(test_data))
    if forecast.HAS_LGB:
        import lightgbm as lgb
        model_lgb = lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.03, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1
        )
        model_lgb.fit(train_data[feat_cols], train_data["target"])
        pred_lgb = model_lgb.predict(test_data[feat_cols])
        
    pred_xgb = np.zeros(len(test_data))
    if forecast.HAS_XGB:
        import xgboost as xgb
        model_xgb = xgb.XGBRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
        )
        model_xgb.fit(train_data[feat_cols], train_data["target"])
        pred_xgb = model_xgb.predict(test_data[feat_cols])
        
    if forecast.HAS_LGB and forecast.HAS_XGB:
        pred_prices = 0.5 * pred_lgb + 0.5 * pred_xgb
        model_name = "Ensemble (LightGBM + XGBoost)"
    elif forecast.HAS_LGB:
        pred_prices = pred_lgb
        model_name = "LightGBM"
    elif forecast.HAS_XGB:
        pred_prices = pred_xgb
        model_name = "XGBoost"
    else:
        pred_prices = test_data["price_lag24"].to_numpy()
        model_name = "Lag-24h (Fallback)"
        
    pred_p10 = None
    pred_p90 = None
    if forecast.HAS_LGB:
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
        
        if forecast.HAS_XGB:
            delta_p10 = p10_lgb - pred_lgb
            delta_p90 = p90_lgb - pred_lgb
            pred_p10 = pred_prices + delta_p10
            pred_p90 = pred_prices + delta_p90
        else:
            pred_p10 = p10_lgb
            pred_p90 = p90_lgb
    else:
        std_dev = train_data["target"].std() if len(train_data) > 0 else 15.0
        pred_p10 = pred_prices - 1.28 * std_dev
        pred_p90 = pred_prices + 1.28 * std_dev
        
    return (
        pd.Series(pred_prices, index=test_data.index, name="predicted_price"),
        pd.Series(pred_p10, index=test_data.index, name="predicted_p10"),
        pd.Series(pred_p90, index=test_data.index, name="predicted_p90"),
        model_name
    )


def plot_price_and_soc_plotly(df_plot, price_col="Predicted Price (€/MWh)", p10_col=None, p90_col=None, soc_col="SOC (MWh)"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # 1. Add P10-P90 Shaded Band (if available)
    if p10_col in df_plot.columns and p90_col in df_plot.columns:
        # P90 line (invisible/transparent, used as top of filled area)
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot[p90_col],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                name='P90 Upper Bound'
            ),
            secondary_y=False
        )
        
        # P10 line (filled to next y)
        fig.add_trace(
            go.Scatter(
                x=df_plot.index,
                y=df_plot[p10_col],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor='rgba(56, 189, 248, 0.15)', # Sleek semitransparent blue
                name='90% Confidence Interval (P10-P90)',
                showlegend=True
            ),
            secondary_y=False
        )
    
    # 2. Add Price line
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot[price_col],
            mode='lines+markers',
            line=dict(color='#38bdf8', width=3), # Sleek blue
            name=price_col
        ),
        secondary_y=False
    )
    
    # 3. Add SOC line
    fig.add_trace(
        go.Scatter(
            x=df_plot.index,
            y=df_plot[soc_col],
            mode='lines',
            line=dict(color='#10b981', width=2.5, dash='dash'), # Green dashed line
            name=soc_col
        ),
        secondary_y=True
    )
    
    # Update axes and layout
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#0f1116",
        paper_bgcolor="#0f1116",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        height=450
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#1e293b"
    )
    
    fig.update_yaxes(
        title_text="Price (€/MWh)",
        showgrid=True,
        gridcolor="#1e293b",
        secondary_y=False
    )
    
    fig.update_yaxes(
        title_text="SOC (MWh)",
        showgrid=False,
        secondary_y=True
    )
    
    return fig

# Main Title & Description
st.title("🔋 BESS Arbitrage & Price Forecasting Dashboard")
st.write("Optimize Battery Energy Storage System (BESS) charging and evaluate LightGBM price forecasting models.")

# --- SIDEBAR CONFIGURATION ---
st.sidebar.header("🔋 Battery Specifications")
cap = st.sidebar.slider("Capacity (MWh)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)
pow_rate = st.sidebar.slider("Power Rating (MW)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
rte = st.sidebar.slider("Round-Trip Efficiency (RTE)", min_value=0.5, max_value=1.0, value=0.88, step=0.01)
cycle_cost = st.sidebar.slider("Cycle Cost (€/MWh)", min_value=0.0, max_value=20.0, value=4.0, step=0.5)
soc_init = st.sidebar.slider("Initial State-of-Charge (SOC)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

st.sidebar.markdown("---")
st.sidebar.header("📉 Battery Degradation settings")
use_degradation = st.sidebar.toggle("Enable Dynamic Degradation", value=False)
deg_coef_soc = 0.5
deg_coef_power = 0.2
if use_degradation:
    deg_coef_soc = st.sidebar.slider("High SOC Stress Coeff (€/MWh)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
    deg_coef_power = st.sidebar.slider("Power C-Rate Stress Coeff (€/MW²h)", min_value=0.0, max_value=2.0, value=0.2, step=0.05)

st.sidebar.markdown("---")
st.sidebar.header("🌐 Multi-Market Settings")
use_multi_market = st.sidebar.toggle("Enable Multi-Market Co-Optimization", value=False)
fcr_price_val = 18.0
idm_spread_val = 10.0
if use_multi_market:
    fcr_price_val = st.sidebar.slider("FCR Capacity Price (€/MW/h)", min_value=5.0, max_value=50.0, value=18.0, step=1.0)
    idm_spread_val = st.sidebar.slider("Intraday Price Volatility Spread (€/MWh)", min_value=0.0, max_value=30.0, value=10.0, step=1.0)

st.sidebar.markdown("---")
st.sidebar.header("🛡️ Risk Settings")
gamma = st.sidebar.slider("Risk Aversion Coefficient (gamma)", min_value=0.0, max_value=0.95, value=0.0, step=0.05)

st.sidebar.markdown("---")
st.sidebar.header("📅 Data Configuration")
data_source = st.sidebar.selectbox("Data Source", ["Synthetic Data (Offline)", "ENTSO-E API (Live)"])

start_date = st.sidebar.date_input("Start Date", datetime.date(2025, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime.date(2025, 2, 1))

if start_date >= end_date:
    st.sidebar.error("Error: Start Date must be before End Date.")

# Set up ENTSO-E parameters
api_key = ""
if data_source == "ENTSO-E API (Live)":
    api_key_env = os.environ.get("ENTSOE_API_KEY", "").strip()
    api_key = st.sidebar.text_input("ENTSO-E API Key", value=api_key_env, type="password").strip()
    if api_key:
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else api_key
        st.sidebar.caption(f"🔑 Key Loaded: `{masked_key}` ({len(api_key)} chars)")
    else:
        st.sidebar.warning("API Key is required to fetch real data.")

# Run Optimization button
run_btn = st.sidebar.button("🚀 Run Analysis", use_container_width=True)

# Instantiate Battery object
bat = bess_arbitrage.Battery(
    capacity_mwh=cap,
    power_mw=pow_rate,
    rte=rte,
    soc_init_frac=soc_init,
    cycle_cost_eur_mwh=cycle_cost,
    degradation_coef_soc=deg_coef_soc,
    degradation_coef_power=deg_coef_power
)

# Load data helper
def get_data(source, start_str, end_str, api_key_val):
    if source == "Synthetic Data (Offline)":
        return generate_synthetic_entsoe_data(start_str, end_str)
    else:
        if not api_key_val:
            raise ValueError("ENTSO-E API key is missing. Enter it in the sidebar.")
        # Temporarily set environment variable
        os.environ["ENTSOE_API_KEY"] = api_key_val.strip()
        # Do not catch and print errors here; let the caller handle it gracefully
        return entsoe_pipeline.build_dataset(start_str, end_str)

# Core execution
if "df" not in st.session_state:
    # Initialize with default synthetic data on load
    st.session_state.df = generate_synthetic_entsoe_data("2025-01-01", "2025-02-01")
if "data_status" not in st.session_state:
    st.session_state.data_status = "synthetic"

if run_btn:
    with st.spinner("Loading data and running simulation..."):
        try:
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            st.session_state.df = get_data(data_source, start_str, end_str, api_key)
            st.session_state.data_status = "live" if data_source == "ENTSO-E API (Live)" else "synthetic"
            st.success("Data loaded and simulation completed!")
        except Exception as e:
            err_msg = str(e)
            import re
            
            # Mask the API key in the error message for privacy
            safe_err_msg = re.sub(r'securityToken=[a-zA-Z0-9\-]+', 'securityToken=********', err_msg)
            
            # Check if there is an XML response body from requests
            xml_reason = ""
            try:
                curr = e
                while curr:
                    if hasattr(curr, "response") and curr.response is not None:
                        resp_text = curr.response.text
                        if resp_text and ("<Reason>" in resp_text or "<text>" in resp_text):
                            import xml.etree.ElementTree as ET
                            # Clean namespace if present to make parsing easier
                            cleaned_xml = re.sub(r'\sxmlns="[^"]+"', '', resp_text)
                            root = ET.fromstring(cleaned_xml)
                            reasons = [elem.text for elem in root.findall(".//text")]
                            if reasons:
                                xml_reason = " | ".join(reasons)
                        break
                    curr = getattr(curr, "__cause__", None) or getattr(curr, "__context__", None)
            except Exception as xml_err:
                pass
                
            error_details = safe_err_msg
            if xml_reason:
                error_details += f"\n\n**ENTSO-E Hata Açıklaması (Reason):** `{xml_reason}`"
            
            if "401" in err_msg or "Unauthorized" in err_msg:
                st.sidebar.error("❌ ENTSO-E API Key Unauthorized (401)")
                st.error(f"🔑 **ENTSO-E API Anahtarı Yetkisiz (401 Error)**: Girilen API anahtarı geçersiz veya henüz aktifleştirilmemiş.\n\n"
                         f"**Sistem Hatası:** `{error_details}`\n\n"
                         "**Nasıl Düzeltilir?**\n"
                         "1. [ENTSO-E Transparency Portal](https://transparency.entsoe.eu/) adresine kayıt olun.\n"
                         "2. Kayıtlı e-posta adresinizden **transparency@entsoe.eu** adresine 'API access' konulu bir e-posta gönderin.\n"
                         "3. Hesabınız aktifleştirildikten sonra (genellikle birkaç saat sürer) token'ınız çalışacaktır.\n\n"
                         "**Geçici Çözüm (Fallback):** Analizin kesintiye uğramaması için seçtiğiniz tarih aralığına uygun **Sentetik (Yapay) Veri** otomatik olarak üretilmiştir. Arayüzü incelemeye devam edebilirsiniz.")
            else:
                st.sidebar.error(f"❌ ENTSO-E Bağlantı Hatası")
                st.error(f"⚠️ **ENTSO-E Veri Çekme Hatası**:\n\n`{error_details}`\n\n"
                         "**Geçici Çözüm (Fallback):** Sentetik veri otomatik olarak yüklenmiştir.")
            
            # Fallback action
            st.session_state.df = generate_synthetic_entsoe_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
            st.session_state.data_status = "fallback"

df = st.session_state.df

# Show active data status banner
if st.session_state.data_status == "live":
    st.info("🟢 **Aktif Veri:** Canlı ENTSO-E API Verisi")
elif st.session_state.data_status == "fallback":
    st.warning("⚠️ **Aktif Veri:** Sentetik Veri (ENTSO-E API bağlantısı yetkisiz veya başarısız olduğu için otomatik geçiş yapıldı).")
else:
    st.info("ℹ️ **Aktif Veri:** Çevrimdışı Sentetik Veri")


if df is not None:
    # Set up layout tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Tomorrow's Live Plan", "📊 Daily Optimization Plan", "📈 Multi-Day Backtest & Forecasting", "📁 Raw Data Viewer"])
    
    # --- TAB 1: TOMORROW'S LIVE PLAN ---
    with tab1:
        st.header("🔮 Tomorrow's Live Operation Plan (Live Optimizer)")
        st.write(
            "Bu modül, yarının fiyat eğrisini tahmin etmek için canlı hava tahmini, yük tahmini ve tarihsel fiyat "
            "verilerini ENTSO-E'den indirir. Ardından LightGBM & XGBoost Ensemble modelini eğitip yarının fiyatlarını tahmin eder "
            "ve bataryanın yarın için en karlı şarj/deşarj takvimini çıkartır."
        )
        
        # We can trigger tomorrow's optimization explicitly
        run_tomorrow = st.button("🔮 Run Tomorrow's Forecast & Optimization", use_container_width=True)
        
        if run_tomorrow or "tomorrow_results" in st.session_state:
            if run_tomorrow:
                with st.spinner("Fetching live data and running ML models for tomorrow..."):
                    today = datetime.date.today()
                    tomorrow = today + datetime.timedelta(days=1)
                    
                    # Set API key in environment if provided
                    if api_key:
                        os.environ["ENTSOE_API_KEY"] = api_key
                        
                    is_live = False
                    # Try to fetch live data
                    if api_key and len(api_key) > 10:
                        try:
                            import predict_tomorrow
                            df_tomorrow = predict_tomorrow.get_live_inference_data(today, tomorrow)
                            is_live = True
                            st.success("Successfully fetched live ENTSO-E data for tomorrow's forecast!")
                        except Exception as e:
                            st.warning(f"Failed to fetch live ENTSO-E data ({e}). Falling back to offline synthetic data.")
                            df_tomorrow = generate_synthetic_inference_data_in_app(today, tomorrow)
                    else:
                        st.info("ENTSO-E API Key not provided or invalid. Running in offline/synthetic mode.")
                        df_tomorrow = generate_synthetic_inference_data_in_app(today, tomorrow)
                        
                    # Predict tomorrow's prices
                    pred_series, pred_p10_series, pred_p90_series, model_name = predict_tomorrow_prices_in_app(df_tomorrow, today, tomorrow)
                    pred_prices = pred_series.to_numpy()
                    pred_p10 = pred_p10_series.to_numpy()
                    pred_p90 = pred_p90_series.to_numpy()
                    
                    # Simulating IDM and FCR prices for tomorrow
                    T = len(pred_prices)
                    tomorrow_idm = pred_prices + idm_spread_val * np.sin(np.arange(T) * 2 * np.pi / 24)
                    tomorrow_fcr = np.full(T, fcr_price_val)
                    
                    # Run battery optimization
                    res_tomorrow = bess_arbitrage.optimize_day(
                        pred_prices, bat, 
                        use_degradation=use_degradation,
                        use_multi_market=use_multi_market,
                        idm_prices=tomorrow_idm,
                        fcr_prices=tomorrow_fcr,
                        prices_p10=pred_p10,
                        prices_p90=pred_p90,
                        gamma=gamma
                    )
                    
                    st.session_state.tomorrow_results = {
                        "pred_series": pred_series,
                        "pred_p10_series": pred_p10_series,
                        "pred_p90_series": pred_p90_series,
                        "model_name": model_name,
                        "res": res_tomorrow,
                        "is_live": is_live,
                        "tomorrow_date": tomorrow
                    }
            
            # Display results
            t_res = st.session_state.tomorrow_results
            pred_series = t_res["pred_series"]
            pred_p10_series = t_res.get("pred_p10_series", None)
            pred_p90_series = t_res.get("pred_p90_series", None)
            model_name = t_res["model_name"]
            res = t_res["res"]
            tomorrow_date = t_res["tomorrow_date"]
            
            if res["status"] in ("optimal", "optimal_inaccurate"):
                daily_profit = res["profit"]
                total_charge_mwh = float(np.sum(res["charge"]))
                total_discharge_mwh = float(np.sum(res["discharge"]))
                
                # Metrics cards
                st.subheader(f"Metrics Summary for Tomorrow ({tomorrow_date})")
                if use_multi_market:
                    dam_prof = res["dam_profit"]
                    idm_prof = res["idm_profit"]
                    fcr_prof = res["fcr_profit"]
                    net_prof = res["profit"]
                    deg_cost = res.get("degradation_cost", 0.0)
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("DAM Profit", f"€ {dam_prof:,.2f}", help="Day-Ahead market net arbitrage profit")
                    col2.metric("IDM Profit", f"€ {idm_prof:,.2f}", help="Intraday market net arbitrage profit")
                    col3.metric("FCR Revenue", f"€ {fcr_prof:,.2f}", help="FCR Reserve Capacity payments")
                    if use_degradation:
                        col4.metric("Est. Degradation Cost", f"€ {deg_cost:,.2f}")
                    else:
                        col4.metric("Cycle Wear Cost", f"€ {res['cycle_cost']:,.2f}")
                    col5.metric("Net Profit", f"€ {net_prof:,.2f}", help="Total Revenue minus wear and degradation cost")
                elif use_degradation:
                    gross_profit = res["gross_profit"] - res["cycle_cost"]
                    deg_cost = res["degradation_cost"]
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Gross Profit", f"€ {gross_profit:,.2f}", help="Arbitrage profit before degradation penalty")
                    m2.metric("Est. Degradation Cost", f"€ {deg_cost:,.2f}", help="Convex SOC and Power C-rate penalty")
                    m3.metric("Net Profit", f"€ {daily_profit:,.2f}", help="Arbitrage profit minus degradation penalty")
                    m4.metric("Total Discharged", f"{total_discharge_mwh:.2f} MWh")
                    m5.metric("Cycle Equivalent", f"{(total_discharge_mwh / bat.E):.2f} Cycles")
                else:
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Expected Profit", f"€ {daily_profit:,.2f}")
                    m2.metric("Total Charge", f"{total_charge_mwh:.2f} MWh")
                    m3.metric("Total Discharge", f"{total_discharge_mwh:.2f} MWh")
                    m4.metric("Cycle Equivalent", f"{(total_discharge_mwh / bat.E):.2f} Cycles")
                
                # Plot data
                plot_df_data = {
                    "Predicted Price (€/MWh)": pred_series.values,
                    "SOC (MWh)": res["soc"][1:],
                    "Total Charge Power (MW)": res["charge"],
                    "Total Discharge Power (MW)": res["discharge"]
                }
                if pred_p10_series is not None:
                    plot_df_data["predicted_p10"] = pred_p10_series.values
                if pred_p90_series is not None:
                    plot_df_data["predicted_p90"] = pred_p90_series.values

                if use_multi_market:
                    plot_df_data["DAM Charge (MW)"] = res["c_dam"]
                    plot_df_data["DAM Discharge (MW)"] = res["d_dam"]
                    plot_df_data["IDM Charge (MW)"] = res["c_idm"]
                    plot_df_data["IDM Discharge (MW)"] = res["d_idm"]
                    plot_df_data["FCR Capacity (MW)"] = res["r_fcr"]
                    
                plot_df = pd.DataFrame(plot_df_data, index=pred_series.index)
                
                plot_df["Action"] = "Idle"
                plot_df.loc[plot_df["Total Charge Power (MW)"] > 1e-3, "Action"] = "CHARGE 🔌"
                plot_df.loc[plot_df["Total Discharge Power (MW)"] > 1e-3, "Action"] = "DISCHARGE ⚡"
                
                # Charts
                st.subheader("Price Forecast & SOC Profile")
                fig_tomorrow = plot_price_and_soc_plotly(
                    plot_df,
                    price_col="Predicted Price (€/MWh)",
                    p10_col="predicted_p10" if "predicted_p10" in plot_df.columns else None,
                    p90_col="predicted_p90" if "predicted_p90" in plot_df.columns else None,
                    soc_col="SOC (MWh)"
                )
                st.plotly_chart(fig_tomorrow, use_container_width=True)
                
                if use_multi_market:
                    st.subheader("BESS Multi-Market Scheduling Schedule")
                    st.bar_chart(plot_df[["DAM Charge (MW)", "DAM Discharge (MW)", "IDM Charge (MW)", "IDM Discharge (MW)", "FCR Capacity (MW)"]])
                else:
                    st.subheader("BESS Operation Schedule")
                    st.bar_chart(plot_df[["Total Charge Power (MW)", "Total Discharge Power (MW)"]])
                
                # Table
                st.subheader("Hourly Operation Guide")
                hourly_table_data = {
                    "Predicted Price (€/MWh)": plot_df["Predicted Price (€/MWh)"].round(2),
                    "SOC (MWh)": plot_df["SOC (MWh)"].round(3)
                }
                if use_multi_market:
                    hourly_table_data["DAM Charge (MW)"] = plot_df["DAM Charge (MW)"].round(3)
                    hourly_table_data["DAM Discharge (MW)"] = plot_df["DAM Discharge (MW)"].round(3)
                    hourly_table_data["IDM Charge (MW)"] = plot_df["IDM Charge (MW)"].round(3)
                    hourly_table_data["IDM Discharge (MW)"] = plot_df["IDM Discharge (MW)"].round(3)
                    hourly_table_data["FCR Capacity (MW)"] = plot_df["FCR Capacity (MW)"].round(3)
                    hourly_table_data["Action"] = plot_df["Action"]
                else:
                    hourly_table_data["Charge Rate (MW)"] = plot_df["Total Charge Power (MW)"].round(3)
                    hourly_table_data["Discharge Rate (MW)"] = plot_df["Total Discharge Power (MW)"].round(3)
                    hourly_table_data["Action"] = plot_df["Action"]
                    
                hourly_table = pd.DataFrame(hourly_table_data)
                hourly_table.index = hourly_table.index.strftime("%H:%M")
                st.dataframe(hourly_table, use_container_width=True)
                
                # Download tomorrow's schedule
                csv_tomorrow = plot_df.to_csv().encode('utf-8')
                st.download_button(
                    label="📥 Download Tomorrow's Schedule CSV",
                    data=csv_tomorrow,
                    file_name=f"tomorrow_bess_schedule_{tomorrow_date}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.error(f"Optimization solver failed for tomorrow with status: {res['status']}")

    # --- TAB 2: DAILY OPTIMIZATION ---
    with tab2:
        st.header("📊 Single Day Optimization (Perfect Foresight)")
        st.write("Select a single day to view the optimized BESS operation schedule.")
        
        # Get list of unique dates
        dates = sorted(list(df.index.date))
        selected_date = st.selectbox("Select Date", options=dates, index=min(len(dates)-1, 0))
        
        # Filter day data
        day_df = df[df.index.date == selected_date]
        
        if len(day_df) < 24:
            st.warning(f"Incomplete data for {selected_date} ({len(day_df)} hours found). Optimizing for available hours.")
            
        if len(day_df) > 0:
            prices = day_df["price_eur_mwh"].dropna().to_numpy()
            
            if len(prices) > 0:
                T = len(prices)
                day_idm = prices + idm_spread_val * np.sin(np.arange(T) * 2 * np.pi / 24)
                day_fcr = np.full(T, fcr_price_val)
                
                prices_std = np.std(prices) if len(prices) > 1 else 10.0
                prices_p10 = prices - 1.28 * prices_std
                prices_p90 = prices + 1.28 * prices_std
                
                res = bess_arbitrage.optimize_day(
                    prices, bat, 
                    use_degradation=use_degradation,
                    use_multi_market=use_multi_market,
                    idm_prices=day_idm,
                    fcr_prices=day_fcr,
                    prices_p10=prices_p10,
                    prices_p90=prices_p90,
                    gamma=gamma
                )
                
                if res["status"] in ("optimal", "optimal_inaccurate"):
                    # Calculate metrics
                    daily_profit = res["profit"]
                    total_charge_mwh = float(np.sum(res["charge"]))
                    total_discharge_mwh = float(np.sum(res["discharge"]))
                    
                    # Columns for metrics
                    if use_multi_market:
                        dam_prof = res["dam_profit"]
                        idm_prof = res["idm_profit"]
                        fcr_prof = res["fcr_profit"]
                        net_prof = res["profit"]
                        deg_cost = res.get("degradation_cost", 0.0)
                        
                        col1, col2, col3, col4, col5 = st.columns(5)
                        col1.metric("DAM Profit", f"€ {dam_prof:,.2f}", help="Day-Ahead market net arbitrage profit")
                        col2.metric("IDM Profit", f"€ {idm_prof:,.2f}", help="Intraday market net arbitrage profit")
                        col3.metric("FCR Revenue", f"€ {fcr_prof:,.2f}", help="FCR Reserve Capacity payments")
                        if use_degradation:
                            col4.metric("Est. Degradation Cost", f"€ {deg_cost:,.2f}")
                        else:
                            col4.metric("Cycle Wear Cost", f"€ {res['cycle_cost']:,.2f}")
                        col5.metric("Net Profit", f"€ {net_prof:,.2f}", help="Total Revenue minus wear and degradation cost")
                    elif use_degradation:
                        gross_profit = res["gross_profit"] - res["cycle_cost"]
                        deg_cost = res["degradation_cost"]
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("Gross Profit", f"€ {gross_profit:,.2f}", help="Arbitrage profit before degradation penalty")
                        m2.metric("Est. Degradation Cost", f"€ {deg_cost:,.2f}", help="Convex SOC and Power C-rate penalty")
                        m3.metric("Net Profit", f"€ {daily_profit:,.2f}", help="Arbitrage profit minus degradation penalty")
                        m4.metric("Total Discharged", f"{total_discharge_mwh:.2f} MWh")
                        m5.metric("Cycle Equivalent", f"{(total_discharge_mwh / bat.E):.2f} Cycles")
                    else:
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Daily Profit", f"€ {daily_profit:,.2f}")
                        m2.metric("Total Charged", f"{total_charge_mwh:.2f} MWh")
                        m3.metric("Total Discharged", f"{total_discharge_mwh:.2f} MWh")
                        m4.metric("Cycle Equivalent", f"{(total_discharge_mwh / bat.E):.2f} Cycles")
                    
                    # Create plotting data
                    plot_df_data = {
                        "Price (€/MWh)": prices,
                        "prices_p10": prices_p10,
                        "prices_p90": prices_p90,
                        "SOC (MWh)": res["soc"][1:],  # exclude initial soc at index 0
                        "Total Charge Power (MW)": res["charge"],
                        "Total Discharge Power (MW)": res["discharge"]
                    }
                    if use_multi_market:
                        plot_df_data["DAM Charge (MW)"] = res["c_dam"]
                        plot_df_data["DAM Discharge (MW)"] = res["d_dam"]
                        plot_df_data["IDM Charge (MW)"] = res["c_idm"]
                        plot_df_data["IDM Discharge (MW)"] = res["d_idm"]
                        plot_df_data["FCR Capacity (MW)"] = res["r_fcr"]
                        
                    plot_df = pd.DataFrame(plot_df_data, index=day_df.index[:len(prices)])
                    
                    plot_df["Action"] = "Idle"
                    plot_df.loc[plot_df["Total Charge Power (MW)"] > 1e-3, "Action"] = "Charge"
                    plot_df.loc[plot_df["Total Discharge Power (MW)"] > 1e-3, "Action"] = "Discharge"
                    
                    # Custom Streamlit Charts
                    st.subheader("Price Forecast & SOC Profile")
                    fig_day = plot_price_and_soc_plotly(
                        plot_df,
                        price_col="Price (€/MWh)",
                        p10_col="prices_p10" if "prices_p10" in plot_df.columns else None,
                        p90_col="prices_p90" if "prices_p90" in plot_df.columns else None,
                        soc_col="SOC (MWh)"
                    )
                    st.plotly_chart(fig_day, use_container_width=True)
                    
                    # Draw Charge/Discharge Schedule
                    if use_multi_market:
                        st.bar_chart(plot_df[["DAM Charge (MW)", "DAM Discharge (MW)", "IDM Charge (MW)", "IDM Discharge (MW)", "FCR Capacity (MW)"]])
                    else:
                        st.bar_chart(plot_df[["Total Charge Power (MW)", "Total Discharge Power (MW)"]])
                    
                    # Display hourly breakdown
                    st.subheader("Hourly Operations Table")
                    hourly_table_data = {
                        "Price (€/MWh)": plot_df["Price (€/MWh)"].round(2),
                        "SOC (MWh)": plot_df["SOC (MWh)"].round(3)
                    }
                    if use_multi_market:
                        hourly_table_data["DAM Charge (MW)"] = plot_df["DAM Charge (MW)"].round(3)
                        hourly_table_data["DAM Discharge (MW)"] = plot_df["DAM Discharge (MW)"].round(3)
                        hourly_table_data["IDM Charge (MW)"] = plot_df["IDM Charge (MW)"].round(3)
                        hourly_table_data["IDM Discharge (MW)"] = plot_df["IDM Discharge (MW)"].round(3)
                        hourly_table_data["FCR Capacity (MW)"] = plot_df["FCR Capacity (MW)"].round(3)
                        hourly_table_data["Action"] = plot_df["Action"]
                    else:
                        hourly_table_data["Charge Rate (MW)"] = plot_df["Total Charge Power (MW)"].round(3)
                        hourly_table_data["Discharge Rate (MW)"] = plot_df["Total Discharge Power (MW)"].round(3)
                        hourly_table_data["Action"] = plot_df["Action"]
                        
                    hourly_table = pd.DataFrame(hourly_table_data)
                    hourly_table.index = hourly_table.index.strftime("%H:%M")
                    st.dataframe(hourly_table, use_container_width=True)
                else:
                    st.error(f"Optimization solver failed with status: {res['status']}")
            else:
                st.error("No valid prices found for the selected date.")

    # --- TAB 3: MULTI-DAY BACKTEST & FORECASTING ---
    with tab3:
        st.header("📈 Backtest & Forecast Comparison")
        st.write("Compare the financial returns of different trading strategies:")
        st.markdown("""
        * **Perfect Foresight (Oracle)**: Optimal schedule solved with future prices known in advance. Gives the theoretical upper bound.
        * **Baseline Forecast**: Uses price from 1 week ago (lag 168h) as the forecast for the current day's optimization.
        * **LightGBM ML Forecast**: Predicts tomorrow's prices using historical lag features, weather, and load forecasts.
        """)
        
        # We need enough data to run LightGBM walk-forward (needs at least a week + test data)
        min_hours_req = 168 + 48
        if len(df) < min_hours_req:
            st.error(f"Not enough data for forecasting. Need at least {min_hours_req} hours, currently have {len(df)} hours.")
        else:
            # Let the user select the train/test split date
            default_test_idx = int(len(df) * 0.6)
            test_start_date = df.index[default_test_idx].date()
            
            eval_date = st.date_input("Evaluation Start Date (Test Set Start)", test_start_date)
            eval_dt = pd.Timestamp(eval_date, tz="Europe/Berlin")
            
            if eval_dt <= df.index[168]:
                st.error("Evaluation Start Date must be at least 1 week after the dataset start to allow lag feature calculation.")
            elif eval_dt >= df.index[-24]:
                st.error("Evaluation Start Date must be at least 1 day before the dataset end.")
            else:
                run_eval = st.button("📊 Evaluate Forecast & Arbitrage Models")
                
                if run_eval or "eval_results" not in st.session_state:
                    with st.spinner("Training Machine Learning models and running multi-day backtest..."):
                        real = df["price_eur_mwh"].dropna()
                        base_pred = forecast.baseline_forecast(df)
                        
                        # Compare on the test window only (fair comparison)
                        mask = real.index >= eval_dt
                        
                        # ML walkforward ensemble and quantiles
                        feat = forecast.make_features(df)
                        pred_ml = forecast.ensemble_walkforward(feat, str(eval_date))
                        pred_ml_p10 = forecast.ensemble_walkforward(feat, str(eval_date), quantile=0.1)
                        pred_ml_p90 = forecast.ensemble_walkforward(feat, str(eval_date), quantile=0.9)
                        
                        # Create cumulative profit curves
                        # We will build day-by-day profits
                        # Let's solve day-by-day and track running sum
                        test_idx = real[mask].index
                        test_dates = sorted(list(set(test_idx.date)))
                        
                        pf_cum = 0
                        bl_cum = 0
                        ml_cum = 0
                        
                        pf_profits = []
                        bl_profits = []
                        ml_profits = []
                        
                        cum_data = []
                        for day in test_dates:
                            day_idx = test_idx[test_idx.date == day]
                            if len(day_idx) < 24:
                                continue
                                
                            rp = real.loc[day_idx].to_numpy()
                            bp = base_pred.loc[day_idx].to_numpy()
                            mp = pred_ml.loc[day_idx].to_numpy() if day_idx[0] in pred_ml.index else np.array([])
                            
                            # Perfect Foresight scenarios
                            rp_std = np.std(rp) if len(rp) > 1 else 10.0
                            rp_p10 = rp - 1.28 * rp_std
                            rp_p90 = rp + 1.28 * rp_std
                            rp_idm = rp + idm_spread_val * np.sin(np.arange(24) * 2 * np.pi / 24)
                            rp_fcr = np.full(24, fcr_price_val)
                            
                            # Baseline scenarios
                            bp_std = np.std(bp) if (len(bp) > 1 and not np.isnan(bp).any()) else 10.0
                            bp_p10 = bp - 1.28 * bp_std
                            bp_p90 = bp + 1.28 * bp_std
                            bp_idm = bp + idm_spread_val * np.sin(np.arange(24) * 2 * np.pi / 24) if not np.isnan(bp).any() else None
                            
                            # ML scenarios
                            mp_p10 = pred_ml_p10.loc[day_idx].to_numpy() if (pred_ml_p10 is not None and day_idx[0] in pred_ml_p10.index) else (mp - 15.0)
                            mp_p90 = pred_ml_p90.loc[day_idx].to_numpy() if (pred_ml_p90 is not None and day_idx[0] in pred_ml_p90.index) else (mp + 15.0)
                            mp_idm = mp + idm_spread_val * np.sin(np.arange(24) * 2 * np.pi / 24) if len(mp) == 24 else None
                            
                            # 1. Perfect Foresight (Oracle)
                            p_res = bess_arbitrage.optimize_day(
                                rp, bat, 
                                use_degradation=use_degradation,
                                use_multi_market=use_multi_market,
                                idm_prices=rp_idm,
                                fcr_prices=rp_fcr,
                                prices_p10=rp_p10,
                                prices_p90=rp_p90,
                                gamma=gamma
                            ) if len(rp) == 24 else {"status": "failed"}
                            if p_res["status"] in ("optimal", "optimal_inaccurate"):
                                pf_cum += p_res["profit"]
                                pf_profits.append(p_res["profit"])
                            else:
                                pf_profits.append(0.0)
                                
                            # 2. Baseline
                            b_res = bess_arbitrage.optimize_day(
                                bp, bat, 
                                use_degradation=use_degradation,
                                use_multi_market=use_multi_market,
                                idm_prices=bp_idm,
                                fcr_prices=rp_fcr,
                                prices_p10=bp_p10,
                                prices_p90=bp_p90,
                                gamma=gamma
                            ) if len(bp) == 24 and not np.isnan(bp).any() else {"status": "failed"}
                            if b_res["status"] in ("optimal", "optimal_inaccurate"):
                                if use_multi_market:
                                    real_dam = rp @ b_res["d_dam"] - rp @ b_res["c_dam"]
                                    real_idm = rp_idm @ b_res["d_idm"] - rp_idm @ b_res["c_idm"]
                                    real_fcr = rp_fcr @ b_res["r_fcr"]
                                    gross_val = real_dam + real_idm + real_fcr
                                    cycle_wear = bat.cycle_cost * np.sum(b_res["d_dam"] + b_res["d_idm"])
                                else:
                                    gross_val = rp @ b_res["discharge"] - rp @ b_res["charge"]
                                    cycle_wear = bat.cycle_cost * b_res["discharge"].sum()
                                    
                                if use_degradation:
                                    soc_threshold = 0.8 * bat.E
                                    soc_penalty_val = bat.degradation_coef_soc * np.sum(np.maximum(0, b_res["soc"][1:] - soc_threshold))
                                    if use_multi_market:
                                        power_penalty_val = bat.degradation_coef_power * (np.sum((b_res["c_dam"]+b_res["c_idm"])**2) + np.sum((b_res["d_dam"]+b_res["d_idm"])**2))
                                    else:
                                        power_penalty_val = bat.degradation_coef_power * (np.sum(b_res["charge"]**2) + np.sum(b_res["discharge"]**2))
                                    deg_cost = soc_penalty_val + power_penalty_val
                                else:
                                    deg_cost = 0.0
                                real_profit = gross_val - cycle_wear - deg_cost
                                bl_cum += real_profit
                                bl_profits.append(real_profit)
                            else:
                                bl_profits.append(0.0)
                                
                            # 3. ML (Ensemble)
                            if len(mp) == 24 and not np.isnan(mp).any():
                                m_res = bess_arbitrage.optimize_day(
                                    mp, bat, 
                                    use_degradation=use_degradation,
                                    use_multi_market=use_multi_market,
                                    idm_prices=mp_idm,
                                    fcr_prices=rp_fcr,
                                    prices_p10=mp_p10,
                                    prices_p90=mp_p90,
                                    gamma=gamma
                                )
                                if m_res["status"] in ("optimal", "optimal_inaccurate"):
                                    if use_multi_market:
                                        real_dam = rp @ m_res["d_dam"] - rp @ m_res["c_dam"]
                                        real_idm = rp_idm @ m_res["d_idm"] - rp_idm @ m_res["c_idm"]
                                        real_fcr = rp_fcr @ m_res["r_fcr"]
                                        gross_val = real_dam + real_idm + real_fcr
                                        cycle_wear = bat.cycle_cost * np.sum(m_res["d_dam"] + m_res["d_idm"])
                                    else:
                                        gross_val = rp @ m_res["discharge"] - rp @ m_res["charge"]
                                        cycle_wear = bat.cycle_cost * m_res["discharge"].sum()
                                        
                                    if use_degradation:
                                        soc_threshold = 0.8 * bat.E
                                        soc_penalty_val = bat.degradation_coef_soc * np.sum(np.maximum(0, m_res["soc"][1:] - soc_threshold))
                                        if use_multi_market:
                                            power_penalty_val = bat.degradation_coef_power * (np.sum((m_res["c_dam"]+m_res["c_idm"])**2) + np.sum((m_res["d_dam"]+m_res["d_idm"])**2))
                                        else:
                                            power_penalty_val = bat.degradation_coef_power * (np.sum(m_res["charge"]**2) + np.sum(m_res["discharge"]**2))
                                        deg_cost = soc_penalty_val + power_penalty_val
                                    else:
                                        deg_cost = 0.0
                                    real_profit = gross_val - cycle_wear - deg_cost
                                    ml_cum += real_profit
                                    ml_profits.append(real_profit)
                                else:
                                    ml_profits.append(0.0)
                            else:
                                ml_profits.append(0.0)
                                
                            cum_data.append({
                                "Date": day,
                                "Perfect Foresight": pf_cum,
                                "Baseline Forecast": bl_cum,
                                "Ensemble ML Forecast": ml_cum
                            })
                            
                        cum_df = pd.DataFrame(cum_data).set_index("Date")
                        
                        pf_test = {
                            "total": float(np.sum(pf_profits)),
                            "daily_mean": float(np.mean(pf_profits)) if len(pf_profits) > 0 else 0.0,
                            "n_days": len(pf_profits)
                        }
                        bl_test = {
                            "total": float(np.sum(bl_profits)),
                            "daily_mean": float(np.mean(bl_profits)) if len(bl_profits) > 0 else 0.0,
                            "n_days": len(bl_profits)
                        }
                        ml_test = {
                            "total": float(np.sum(ml_profits)),
                            "daily_mean": float(np.mean(ml_profits)) if len(ml_profits) > 0 else 0.0,
                            "n_days": len(ml_profits)
                        }
                        
                        # Store in session state
                        st.session_state.eval_results = {
                            "pf_test": pf_test,
                            "bl_test": bl_test,
                            "ml_test": ml_test,
                            "cum_df": cum_df
                        }
                
                # Retrieve results
                results = st.session_state.eval_results
                pf_test = results["pf_test"]
                bl_test = results["bl_test"]
                ml_test = results["ml_test"]
                cum_df = results["cum_df"]
                
                # Print metrics table
                st.subheader("Performance Comparison (Test Window)")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Perfect Foresight Profit", f"€ {pf_test['total']:,.2f}", help="Maximum theoretical returns with 100% accurate forecast.")
                col2.metric("Baseline Profit (Lag-168)", f"€ {bl_test['total']:,.2f}", f"Capture: {100 * bl_test['total'] / pf_test['total']:.1f}%")
                col3.metric("Ensemble ML Profit", f"€ {ml_test['total']:,.2f}", f"Capture: {100 * ml_test['total'] / pf_test['total']:.1f}%")
                
                st.subheader("Cumulative Profits Over Time")
                st.line_chart(cum_df)
                
                # Show tabular report
                st.subheader("Strategy Metrics Table")
                metrics_table = pd.DataFrame({
                    "Strategy": ["Perfect Foresight (Oracle)", "Baseline (Lag-168h)", "Ensemble ML Forecast"],
                    "Total Profit (€)": [pf_test["total"], bl_test["total"], ml_test["total"]],
                    "Capture Rate (%)": ["100.0%", f"{100 * bl_test['total'] / pf_test['total']:.1f}%", f"{100 * ml_test['total'] / pf_test['total']:.1f}%"],
                    "Daily Average Profit (€)": [pf_test["daily_mean"], bl_test["daily_mean"], ml_test["daily_mean"]],
                    "Test Days": [pf_test["n_days"], bl_test["n_days"], ml_test["n_days"]]
                }).set_index("Strategy")
                st.dataframe(metrics_table.style.format({
                    "Total Profit (€)": "€{:,.2f}",
                    "Daily Average Profit (€)": "€{:,.2f}"
                }), use_container_width=True)

    # --- TAB 4: RAW DATA VIEWER ---
    with tab4:
        st.header("📁 Raw Hourly Dataset")
        st.write("Browse and download the underlying dataset containing prices, loads, and generation forecast data.")
        st.dataframe(df.style.format(precision=2), use_container_width=True)
        
        # Download button
        csv = df.to_csv().encode('utf-8')
        st.download_button(
            label="📥 Download Dataset as CSV",
            data=csv,
            file_name=f"bess_dataset_{start_date}_to_{end_date}.csv",
            mime="text/csv"
        )
else:
    st.info("Please click 'Run Analysis' in the sidebar to load prices and see results.")
