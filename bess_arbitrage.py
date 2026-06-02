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


def optimize_day(
    prices,
    bat,
    dt_h=1.0,
    soc_start=None,
    soc_end=None,
    use_degradation=False,
    use_multi_market=False,
    idm_prices=None,
    fcr_prices=None,
):
    T = len(prices)
    if soc_start is None:
        soc_start = bat.soc_init
    if soc_end is None:
        soc_end = bat.soc_init

    # Decision variables for Day-Ahead (DAM), Intraday (IDM) and FCR Reserve capacity
    c_dam = cp.Variable(T, nonneg=True)
    d_dam = cp.Variable(T, nonneg=True)
    c_idm = cp.Variable(T, nonneg=True)
    d_idm = cp.Variable(T, nonneg=True)
    r_fcr = cp.Variable(T, nonneg=True)
    soc = cp.Variable(T + 1)

    # Use input prices or apply defaults if None
    if idm_prices is None:
        idm_prices = prices + 10.0 * np.sin(np.arange(T) * 2 * np.pi / 24)
    if fcr_prices is None:
        fcr_prices = np.full(T, 18.0)

    cons = [soc[0] == soc_start, soc[T] == soc_end]
    for t in range(T):
        # SOC dynamics: FCR reserve activation is energy-neutral on average
        cons += [soc[t + 1] == soc[t] + bat.eta_c * (c_dam[t] + c_idm[t]) - (d_dam[t] + d_idm[t]) / bat.eta_d]
        cons += [soc[t + 1] >= bat.soc_min, soc[t + 1] <= bat.soc_max]
        
        # Combined power limits (DAM + IDM + FCR capacity cannot exceed power rating P)
        cons += [c_dam[t] + c_idm[t] + r_fcr[t] <= bat.P * dt_h]
        cons += [d_dam[t] + d_idm[t] + r_fcr[t] <= bat.P * dt_h]
        
        # FCR SOC buffer constraints: 15 mins (0.25h) full activation must not violate SOC bounds
        cons += [soc[t] + (0.25 * r_fcr[t]) / bat.eta_d <= bat.soc_max]
        cons += [soc[t] - 0.25 * bat.eta_c * r_fcr[t] >= bat.soc_min]

    if not use_multi_market:
        # Disable IDM and FCR markets if multi-market optimization is turned off
        cons += [c_idm == 0, d_idm == 0, r_fcr == 0]

    # Revenue streams
    dam_revenue = prices @ d_dam - prices @ c_dam
    idm_revenue = idm_prices @ d_idm - idm_prices @ c_idm
    fcr_revenue = fcr_prices @ r_fcr
    
    gross_revenue = dam_revenue + idm_revenue + fcr_revenue
    cycle_degradation = bat.cycle_cost * cp.sum(d_dam + d_idm)
    
    degradation_cost = 0.0
    soc_penalty_val = 0.0
    power_penalty_val = 0.0
    
    if use_degradation:
        # 1. High SOC stress
        soc_threshold = 0.8 * bat.E
        soc_penalty = cp.sum(cp.pos(soc[1:] - soc_threshold))
        
        # 2. Power stress
        total_c = c_dam + c_idm
        total_d = d_dam + d_idm
        power_penalty = cp.sum_squares(total_c) + cp.sum_squares(total_d)
        
        degradation_cost = bat.degradation_coef_soc * soc_penalty + bat.degradation_coef_power * power_penalty
        
    revenue = gross_revenue - cycle_degradation - degradation_cost
    prob = cp.Problem(cp.Maximize(revenue), cons)
    
    try:
        prob.solve()
    except Exception:
        prob.solve(solver=cp.ECOS)

    # Evaluate individual cost terms post-solve
    if prob.status in ("optimal", "optimal_inaccurate"):
        dam_val = float(dam_revenue.value)
        idm_val = float(idm_revenue.value)
        fcr_val = float(fcr_revenue.value)
        gross_val = dam_val + idm_val + fcr_val
        cycle_val = float(cycle_degradation.value)
        
        if use_degradation:
            soc_threshold = 0.8 * bat.E
            soc_penalty_val = float((bat.degradation_coef_soc * cp.sum(cp.pos(soc[1:] - soc_threshold))).value)
            power_penalty_val = float((bat.degradation_coef_power * (cp.sum_squares(c_dam + c_idm) + cp.sum_squares(d_dam + d_idm))).value)
            deg_cost_val = soc_penalty_val + power_penalty_val
        else:
            deg_cost_val = 0.0
            
        c_val = c_dam.value + c_idm.value
        d_val = d_dam.value + d_idm.value
        c_dam_val = c_dam.value
        d_dam_val = d_dam.value
        c_idm_val = c_idm.value
        d_idm_val = d_idm.value
        r_fcr_val = r_fcr.value
    else:
        dam_val = 0.0
        idm_val = 0.0
        fcr_val = 0.0
        gross_val = 0.0
        cycle_val = 0.0
        deg_cost_val = 0.0
        c_val = np.zeros(T)
        d_val = np.zeros(T)
        c_dam_val = np.zeros(T)
        d_dam_val = np.zeros(T)
        c_idm_val = np.zeros(T)
        d_idm_val = np.zeros(T)
        r_fcr_val = np.zeros(T)

    return {
        "status": prob.status,
        "profit": float(prob.value) if prob.status in ("optimal", "optimal_inaccurate") else 0.0,
        "gross_profit": gross_val,
        "dam_profit": dam_val,
        "idm_profit": idm_val,
        "fcr_profit": fcr_val,
        "cycle_cost": cycle_val,
        "degradation_cost": deg_cost_val,
        "soc_penalty": soc_penalty_val,
        "power_penalty": power_penalty_val,
        "charge": c_val,
        "discharge": d_val,
        "c_dam": c_dam_val,
        "d_dam": d_dam_val,
        "c_idm": c_idm_val,
        "d_idm": d_idm_val,
        "r_fcr": r_fcr_val,
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
