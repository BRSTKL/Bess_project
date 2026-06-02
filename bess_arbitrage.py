"""
BESS Day-Ahead Arbitrage - Perfect-Foresight LP + Backtest
==========================================================
Solver: HiGHS (free, bundled in cvxpy).
All comments ASCII to avoid Windows encoding issues.
"""

import cvxpy as cp
import numpy as np
import pandas as pd


class Battery:
    def __init__(
        self,
        capacity_mwh=2.0,
        power_mw=1.0,
        rte=0.88,
        soc_init_frac=0.5,
        soc_min_frac=0.0,
        soc_max_frac=1.0,
        cycle_cost_eur_mwh=4.0,
        degradation_coef_soc=0.5,
        degradation_coef_power=0.2,
    ):
        self.E = capacity_mwh
        self.P = power_mw
        self.eta_c = np.sqrt(rte)
        self.eta_d = np.sqrt(rte)
        self.soc_init = soc_init_frac * capacity_mwh
        self.soc_min = soc_min_frac * capacity_mwh
        self.soc_max = soc_max_frac * capacity_mwh
        self.cycle_cost = cycle_cost_eur_mwh
        self.degradation_coef_soc = degradation_coef_soc
        self.degradation_coef_power = degradation_coef_power


def optimize_day(prices, bat, dt_h=1.0, soc_start=None, soc_end=None, use_degradation=False):
    T = len(prices)
    if soc_start is None:
        soc_start = bat.soc_init
    if soc_end is None:
        soc_end = bat.soc_init

    c = cp.Variable(T, nonneg=True)
    d = cp.Variable(T, nonneg=True)
    soc = cp.Variable(T + 1)

    cons = [soc[0] == soc_start, soc[T] == soc_end]
    for t in range(T):
        cons += [soc[t + 1] == soc[t] + bat.eta_c * c[t] - d[t] / bat.eta_d]
        cons += [soc[t + 1] >= bat.soc_min, soc[t + 1] <= bat.soc_max]
        cons += [c[t] <= bat.P * dt_h, d[t] <= bat.P * dt_h]

    # Calculate gross arbitrage revenue and baseline cycle cost
    gross_revenue = prices @ d - prices @ c
    cycle_degradation = bat.cycle_cost * cp.sum(d)
    
    degradation_cost = 0.0
    soc_penalty_val = 0.0
    power_penalty_val = 0.0
    
    if use_degradation:
        # 1. High SOC stress: Penalty for staying above 80% SOC
        soc_threshold = 0.8 * bat.E
        soc_penalty = cp.sum(cp.pos(soc[1:] - soc_threshold))
        
        # 2. C-rate / Power stress: Penalty for rapid/high power charge/discharge
        power_penalty = cp.sum_squares(c) + cp.sum_squares(d)
        
        degradation_cost = bat.degradation_coef_soc * soc_penalty + bat.degradation_coef_power * power_penalty
        
    revenue = gross_revenue - cycle_degradation - degradation_cost
    prob = cp.Problem(cp.Maximize(revenue), cons)
    
    try:
        # Default solver selection by CVXPY (handles QP automatically)
        prob.solve()
    except Exception:
        # Fallback to ECOS solver if default solver has issues
        prob.solve(solver=cp.ECOS)

    # Evaluate individual cost terms post-solve
    if prob.status in ("optimal", "optimal_inaccurate"):
        gross_val = float(gross_revenue.value)
        cycle_val = float(cycle_degradation.value)
        if use_degradation:
            soc_threshold = 0.8 * bat.E
            soc_penalty_val = float((bat.degradation_coef_soc * cp.sum(cp.pos(soc[1:] - soc_threshold))).value)
            power_penalty_val = float((bat.degradation_coef_power * (cp.sum_squares(c) + cp.sum_squares(d))).value)
            deg_cost_val = soc_penalty_val + power_penalty_val
        else:
            deg_cost_val = 0.0
    else:
        gross_val = 0.0
        cycle_val = 0.0
        deg_cost_val = 0.0

    return {
        "status": prob.status,
        "profit": float(prob.value) if prob.status in ("optimal", "optimal_inaccurate") else 0.0,
        "gross_profit": gross_val,
        "cycle_cost": cycle_val,
        "degradation_cost": deg_cost_val,
        "soc_penalty": soc_penalty_val,
        "power_penalty": power_penalty_val,
        "charge": c.value,
        "discharge": d.value,
        "soc": soc.value,
    }



def synthetic_prices(n_days=30, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_days * 24, freq="h",
                        tz="Europe/Berlin")
    hours = idx.hour.to_numpy()
    shape = np.array([
        45, 40, 38, 36, 35, 40, 55, 75,
        70, 55, 35, 15, -5, -10, 5, 30,
        55, 80, 110, 120, 95, 75, 60, 50
    ])
    base = shape[hours].astype(float)
    day_level = np.repeat(rng.normal(0, 12, n_days), 24)
    noise = rng.normal(0, 6, len(idx))
    return pd.Series(base + day_level + noise, index=idx, name="price")


def backtest(prices, bat, dt_h=1.0):
    daily_profit, daily_throughput = [], []
    all_charge, all_discharge, all_soc = [], [], []

    for day, grp in prices.groupby(prices.index.date):
        res = optimize_day(grp.to_numpy(), bat, dt_h)
        if res["status"] not in ("optimal", "optimal_inaccurate"):
            print("  WARN %s: solver status %s" % (day, res["status"]))
            continue
        daily_profit.append(res["profit"])
        daily_throughput.append(float(np.sum(res["discharge"])))
        all_charge.append(res["charge"])
        all_discharge.append(res["discharge"])
        all_soc.append(res["soc"][1:])

    profit = np.array(daily_profit)
    thr = np.array(daily_throughput)
    cycles = thr / bat.E

    schedule = pd.DataFrame({
        "price": prices.values,
        "charge_mwh": np.concatenate(all_charge),
        "discharge_mwh": np.concatenate(all_discharge),
        "soc_mwh": np.concatenate(all_soc),
    }, index=prices.index)

    n_days = len(profit)
    total = profit.sum()
    return {
        "schedule": schedule,
        "daily_profit": pd.Series(profit, name="profit"),
        "summary": {
            "n_days": n_days,
            "total_profit_eur": round(total, 2),
            "avg_daily_profit_eur": round(profit.mean(), 2),
            "avg_daily_cycles": round(cycles.mean(), 2),
            "annual_eur_per_MW": round(total / n_days * 365 / bat.P, 2),
        },
    }


if __name__ == "__main__":
    bat = Battery(capacity_mwh=2.0, power_mw=1.0, rte=0.88,
                  cycle_cost_eur_mwh=4.0)
    prices = synthetic_prices(n_days=30, seed=42)
    print("=" * 60)
    print("PERFECT-FORESIGHT BESS ARBITRAGE - BACKTEST")
    print("=" * 60)
    out = backtest(prices, bat)
    for k, v in out["summary"].items():
        print("  %-26s %s" % (k, v))
    print("\nExample day plan (first 24 hours):")
    day0 = out["schedule"].iloc[:24].copy()
    day0["action"] = np.where(day0.charge_mwh > 1e-4, "CHARGE",
                      np.where(day0.discharge_mwh > 1e-4, "DISCHARGE", "-"))
    show = day0[["price", "charge_mwh", "discharge_mwh", "soc_mwh", "action"]]
    show.index = show.index.strftime("%H:%M")
    print(show.round(3).to_string())
