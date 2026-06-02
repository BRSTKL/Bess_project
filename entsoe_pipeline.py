"""
ENTSO-E Data Pipeline - German Day-Ahead Prices + Generation/Load
=================================================================
Runs on YOUR machine (sandbox cannot reach ENTSO-E).

Setup:
    pip install entsoe-py pandas pyarrow
    set ENTSOE_API_KEY=your-key       (Windows PowerShell: $env:ENTSOE_API_KEY="your-key")

Usage:
    python entsoe_pipeline.py 2024-01-01 2025-01-01

Design:
  * API key read from environment variable, never hardcoded.
  * Every query cached to parquet (no duplicate fetches, avoids rate limit).
  * All timestamps tz-aware (Europe/Berlin). DST transitions preserved.
"""

import os
import sys
import pathlib
import cvxpy as cp
import pandas as pd

try:
    from entsoe import EntsoePandasClient
except ImportError:
    EntsoePandasClient = None

COUNTRY = "DE_LU"
TZ = "Europe/Berlin"
CACHE_DIR = pathlib.Path("./entsoe_cache")
CACHE_DIR.mkdir(exist_ok=True)


def _client():
    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise RuntimeError(
            "ENTSOE_API_KEY environment variable not set. "
            "Run:  $env:ENTSOE_API_KEY='your-key'"
        )
    if EntsoePandasClient is None:
        raise RuntimeError("entsoe-py not installed: pip install entsoe-py")
    return EntsoePandasClient(api_key=key)


def _ts(date_str):
    return pd.Timestamp(date_str, tz=TZ)


def _cached(name, start, end, fetch_fn):
    fpath = CACHE_DIR / ("%s_%s_%s.parquet" % (name, start, end))
    if fpath.exists():
        print("  [cache] %s" % fpath.name)
        obj = pd.read_parquet(fpath)
        return obj.iloc[:, 0] if obj.shape[1] == 1 else obj
    print("  [API ] %s %s->%s fetching..." % (name, start, end))
    data = fetch_fn()
    (data.to_frame() if isinstance(data, pd.Series) else data).to_parquet(fpath)
    return data


def get_day_ahead_prices(start, end):
    cli = _client()
    s = _cached("dayahead_price", start, end,
                lambda: cli.query_day_ahead_prices(COUNTRY, start=_ts(start), end=_ts(end)))
    return s.tz_convert(TZ).rename("price_eur_mwh")


def get_load_forecast(start, end):
    cli = _client()
    s = _cached("load_forecast", start, end,
                lambda: cli.query_load_forecast(COUNTRY, start=_ts(start), end=_ts(end)))
    s = s.iloc[:, 0] if isinstance(s, pd.DataFrame) else s
    return s.tz_convert(TZ).rename("load_fc_mw")


def get_renewables_forecast(start, end):
    cli = _client()
    df = _cached("wind_solar_fc", start, end,
                 lambda: cli.query_wind_and_solar_forecast(COUNTRY, start=_ts(start), end=_ts(end)))
    return df.tz_convert(TZ)


def build_dataset(start, end):
    price = get_day_ahead_prices(start, end)
    load = get_load_forecast(start, end)
    res = get_renewables_forecast(start, end)

    df = pd.concat([price, load, res], axis=1)
    df = df.resample("1h").mean()

    daily_counts = df.groupby(df.index.date).size()
    odd = daily_counts[daily_counts != 24]
    if len(odd):
        print("  [DST] %d days != 24 hours (clock change - normal): %s..."
              % (len(odd), list(odd.index[:4])))
    n_missing = df["price_eur_mwh"].isna().sum()
    if n_missing:
        print("  [WARN] %d hours missing price - inspect before ffill." % n_missing)
    return df


def run_perfect_foresight(df):
    from bess_arbitrage import Battery, backtest
    prices = df["price_eur_mwh"].dropna()
    bat = Battery(capacity_mwh=2.0, power_mw=1.0, rte=0.88, cycle_cost_eur_mwh=4.0)
    out = backtest(prices, bat)
    print("\nREAL DATA - PERFECT FORESIGHT RESULTS")
    for k, v in out["summary"].items():
        print("  %-26s %s" % (k, v))
    return out


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else "2025-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else "2025-02-01"

    print("ENTSO-E dataset: %s  %s -> %s" % (COUNTRY, start, end))
    df = build_dataset(start, end)
    print("\nRows: %d | Columns: %s" % (len(df), list(df.columns)))
    print(df.head(3).round(2).to_string())

    df.to_parquet(CACHE_DIR / ("dataset_%s_%s.parquet" % (start, end)))
    print("\nSaved: entsoe_cache/dataset_%s_%s.parquet" % (start, end))

    run_perfect_foresight(df)
