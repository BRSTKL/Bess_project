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
    cycle_cost_eur_mwh=cycle_cost
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
    tab1, tab2, tab3 = st.tabs(["📊 Daily Optimization Plan", "📈 Multi-Day Backtest & Forecasting", "📁 Raw Data Viewer"])
    
    # --- TAB 1: DAILY OPTIMIZATION ---
    with tab1:
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
                res = bess_arbitrage.optimize_day(prices, bat)
                
                if res["status"] in ("optimal", "optimal_inaccurate"):
                    # Calculate metrics
                    daily_profit = res["profit"]
                    total_charge_mwh = float(np.sum(res["charge"]))
                    total_discharge_mwh = float(np.sum(res["discharge"]))
                    
                    # Columns for metrics
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Daily Profit", f"€ {daily_profit:,.2f}")
                    m2.metric("Total Charged", f"{total_charge_mwh:.2f} MWh")
                    m3.metric("Total Discharged", f"{total_discharge_mwh:.2f} MWh")
                    m4.metric("Cycle Equivalent", f"{(total_discharge_mwh / bat.E):.2f} Cycles")
                    
                    # Create plotting data
                    plot_df = pd.DataFrame({
                        "Price (€/MWh)": prices,
                        "SOC (MWh)": res["soc"][1:],  # exclude initial soc at index 0
                        "Charge Power (MW)": res["charge"],
                        "Discharge Power (MW)": res["discharge"]
                    }, index=day_df.index[:len(prices)])
                    
                    plot_df["Action"] = "Idle"
                    plot_df.loc[plot_df["Charge Power (MW)"] > 1e-3, "Action"] = "Charge"
                    plot_df.loc[plot_df["Discharge Power (MW)"] > 1e-3, "Action"] = "Discharge"
                    
                    # Custom Streamlit Charts
                    st.subheader("Price vs Battery Action")
                    
                    # Draw Price and SOC
                    st.line_chart(plot_df[["Price (€/MWh)", "SOC (MWh)"]])
                    
                    # Draw Charge/Discharge Schedule
                    st.bar_chart(plot_df[["Charge Power (MW)", "Discharge Power (MW)"]])
                    
                    # Display hourly breakdown
                    st.subheader("Hourly Operations Table")
                    hourly_table = pd.DataFrame({
                        "Price (€/MWh)": plot_df["Price (€/MWh)"].round(2),
                        "Charge Rate (MW)": plot_df["Charge Power (MW)"].round(3),
                        "Discharge Rate (MW)": plot_df["Discharge Power (MW)"].round(3),
                        "SOC (MWh)": plot_df["SOC (MWh)"].round(3),
                        "Action": plot_df["Action"]
                    })
                    hourly_table.index = hourly_table.index.strftime("%H:%M")
                    st.dataframe(hourly_table, use_container_width=True)
                else:
                    st.error(f"Optimization solver failed with status: {res['status']}")
            else:
                st.error("No valid prices found for the selected date.")

    # --- TAB 2: MULTI-DAY BACKTEST & FORECASTING ---
    with tab2:
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
                    with st.spinner("Training LightGBM model and running multi-day backtest..."):
                        real = df["price_eur_mwh"].dropna()
                        
                        # Perfect foresight
                        pf = forecast.backtest_with_forecast(real, real, bat)
                        
                        # Baseline
                        base_pred = forecast.baseline_forecast(df)
                        bl = forecast.backtest_with_forecast(real, base_pred, bat)
                        
                        # LightGBM Walk-forward
                        feat = forecast.make_features(df)
                        pred_lgb = forecast.lgbm_walkforward(feat, str(eval_date))
                        
                        # Compare on the test window only (fair comparison)
                        mask = real.index >= eval_dt
                        pf_test = forecast.backtest_with_forecast(real[mask], real[mask], bat)
                        bl_test = forecast.backtest_with_forecast(real[mask], base_pred[mask], bat)
                        ml_test = forecast.backtest_with_forecast(real[mask], pred_lgb, bat)
                        
                        # Create cumulative profit curves
                        # We will build day-by-day profits
                        # Let's solve day-by-day and track running sum
                        test_idx = real[mask].index
                        test_dates = sorted(list(set(test_idx.date)))
                        
                        pf_cum = 0
                        bl_cum = 0
                        ml_cum = 0
                        
                        cum_data = []
                        for day in test_dates:
                            day_idx = test_idx[test_idx.date == day]
                            rp = real.loc[day_idx].to_numpy()
                            bp = base_pred.loc[day_idx].to_numpy()
                            mp = pred_lgb.loc[day_idx].to_numpy() if day_idx[0] in pred_lgb.index else np.array([])
                            
                            # Perfect
                            p_res = bess_arbitrage.optimize_day(rp, bat) if len(rp) == 24 else {"status": "failed"}
                            if p_res["status"] in ("optimal", "optimal_inaccurate"):
                                pf_cum += p_res["profit"]
                                
                            # Baseline
                            b_res = bess_arbitrage.optimize_day(bp, bat) if len(bp) == 24 and not np.isnan(bp).any() else {"status": "failed"}
                            if b_res["status"] in ("optimal", "optimal_inaccurate"):
                                bl_cum += (rp @ b_res["discharge"] - rp @ b_res["charge"] - bat.cycle_cost * b_res["discharge"].sum())
                                
                            # ML (LightGBM)
                            if len(mp) == 24 and not np.isnan(mp).any():
                                m_res = bess_arbitrage.optimize_day(mp, bat)
                                if m_res["status"] in ("optimal", "optimal_inaccurate"):
                                    ml_cum += (rp @ m_res["discharge"] - rp @ m_res["charge"] - bat.cycle_cost * m_res["discharge"].sum())
                                    
                            cum_data.append({
                                "Date": day,
                                "Perfect Foresight": pf_cum,
                                "Baseline Forecast": bl_cum,
                                "LightGBM Forecast": ml_cum
                            })
                            
                        cum_df = pd.DataFrame(cum_data).set_index("Date")
                        
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
                col3.metric("LightGBM ML Profit", f"€ {ml_test['total']:,.2f}", f"Capture: {100 * ml_test['total'] / pf_test['total']:.1f}%")
                
                st.subheader("Cumulative Profits Over Time")
                st.line_chart(cum_df)
                
                # Show tabular report
                st.subheader("Strategy Metrics Table")
                metrics_table = pd.DataFrame({
                    "Strategy": ["Perfect Foresight (Oracle)", "Baseline (Lag-168h)", "LightGBM ML Forecast"],
                    "Total Profit (€)": [pf_test["total"], bl_test["total"], ml_test["total"]],
                    "Capture Rate (%)": ["100.0%", f"{100 * bl_test['total'] / pf_test['total']:.1f}%", f"{100 * ml_test['total'] / pf_test['total']:.1f}%"],
                    "Daily Average Profit (€)": [pf_test["daily_mean"], bl_test["daily_mean"], ml_test["daily_mean"]],
                    "Test Days": [pf_test["n_days"], bl_test["n_days"], ml_test["n_days"]]
                }).set_index("Strategy")
                st.dataframe(metrics_table.style.format({
                    "Total Profit (€)": "€{:,.2f}",
                    "Daily Average Profit (€)": "€{:,.2f}"
                }), use_container_width=True)

    # --- TAB 3: RAW DATA VIEWER ---
    with tab3:
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
