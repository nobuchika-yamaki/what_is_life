#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
What Is Life? Temporal Causal Closure Against Dissipation

Independent validation code.

Stage 0  Regression against the accepted Temporal Membranes result.
Stage 1  Delay -> effective temporal retention map (chi).
Stage 2  Minimal causal-closure model with pre-specified interventions.
Stage 3  Paired causal contrasts and retention-window analysis.
Stage 4  Fine causal-delay scaling: does the optimal retention timescale track loop delay?
Stage 5  Consolidated report, yoke audit, and figures.

No numba/scipy dependency. The temporal-membrane DDE uses a true circular buffer.
The causal-closure model is a reduced two-state internal model (h, b) driven by an
external OU process e(t):

    tau_R(b; chi) dh/dt = e(t) - h(t)
    m(t) = e(t-delta_C) - a(t)
    a(t) = k_a b(t) h_use(t)
    V(t) = exp[-m(t)^2 / (2 w^2)]
    db/dt = r_b V(t) [1-b(t)] - mu_b [1-V(t)] b(t)

where

    tau_R(b; chi) = tau_diss [1 + b (chi - 1)].

Thus b=0 returns the intrinsic dissipative retention timescale tau_diss, while
b=1 expresses the effective retention timescale measured by the independent
Temporal Membranes assay, T_corr = chi * tau_diss.

Primary interventions are applied only after a common intact warm-up:
    intact              full retained-history causal closure
    history_shuffle     temporally misalign h at readout while retaining its values
    reuse_knockout      retain h but remove h -> action
    retention_knockout  prevent b from extending retention beyond tau_diss
    closure_cut         sever viability -> b maintenance; established b then decays
    closure_yoked       sever focal viability -> b coupling but replay partner intact b(t) as an exogenous rescue
    external_maintenance clamp b=1 (non-autonomous maintenance control)

All causal-model conditions for a given seed receive the same external trajectory and
are identical until the intervention onset at the end of warm-up.
"""

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EPS = 1e-12
DEFAULT_OUTPUT = Path.home() / "Desktop" / "WHAT_IS_LIFE_TEMPORAL_CAUSAL_CLOSURE"


def now():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg, logfile=None):
    s = f"[{now()}] {msg}"
    print(s, flush=True)
    if logfile is not None:
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(s + "\n")


def finite(a):
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


def mean(a):
    a = finite(a)
    return float(np.mean(a)) if len(a) else np.nan


def sd(a):
    a = finite(a)
    return float(np.std(a, ddof=1)) if len(a) > 1 else np.nan


def sem(a):
    a = finite(a)
    return float(np.std(a, ddof=1) / math.sqrt(len(a))) if len(a) > 1 else np.nan


def interp_cross(x0, y0, x1, y1, target=1.0):
    if not all(np.isfinite([x0, y0, x1, y1])):
        return np.nan
    if abs(y1 - y0) < EPS:
        return float(x1)
    a = (target - y0) / (y1 - y0)
    a = min(1.0, max(0.0, a))
    return float(x0 + a * (x1 - x0))


# -----------------------------------------------------------------------------
# Temporal Membranes regression / retention assay
# -----------------------------------------------------------------------------

@dataclass
class TMConfig:
    gamma: float = 1.0
    beta: float = 2.0
    tau: float = 1.0
    sigma: float = 0.1
    dt_factor: float = 0.01
    warmup_time_factor: float = 100.0
    n_steps: int = 80000
    max_lag_fraction: float = 0.30
    n_reps: int = 20
    base_seed: int = 100000


def tm_dt(gamma, tau, dt_factor=0.01):
    return float(dt_factor * min(tau, 1.0 / gamma))


def tm_simulate_ensemble(cfg: TMConfig):
    """Vectorized across replicates; O(1) circular-buffer update per time step."""
    dt = tm_dt(cfg.gamma, cfg.tau, cfg.dt_factor)
    d = max(1, int(round(cfg.tau / dt)))
    buf_len = d + 5
    warm = int(round((cfg.warmup_time_factor / cfg.gamma) / dt))
    total = warm + cfg.n_steps

    rng = np.random.default_rng(cfg.base_seed)
    init_sd = cfg.sigma / math.sqrt(max(2.0 * cfg.gamma, EPS)) if cfg.sigma > 0 else 1e-6
    buf = rng.normal(0.0, init_sd, size=(buf_len, cfg.n_reps))
    pos = 0
    x = buf[(pos - 1) % buf_len].copy()
    rec = np.empty((cfg.n_steps, cfg.n_reps), dtype=np.float64)
    sqrt_dt = math.sqrt(dt)
    ri = 0

    for n in range(total):
        x_del = buf[(pos - d) % buf_len]
        drift = -cfg.gamma * x - cfg.beta * np.tanh(x_del)
        x = x + drift * dt + cfg.sigma * sqrt_dt * rng.normal(size=cfg.n_reps)
        buf[pos] = x
        pos = (pos + 1) % buf_len
        if n >= warm:
            rec[ri] = x
            ri += 1

    return rec, dt, d


def acf_fft_1d(x, max_lag):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.full(max_lag + 1, np.nan)
    x = x - np.mean(x)
    v = np.var(x)
    if v <= EPS:
        return np.full(max_lag + 1, np.nan)
    max_lag = min(max_lag, n - 2)
    nfft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(fx * np.conjugate(fx), n=nfft)[:max_lag + 1]
    ac = ac / np.arange(n, n - max_lag - 1, -1) / v
    return ac


def tcorr_from_acf(acf, dt):
    thr = 1.0 / math.e
    if acf is None or len(acf) < 3 or not np.isfinite(acf[0]):
        return np.nan, "invalid"
    if acf[0] < thr:
        return 0.0, "initial_below"
    for k in range(1, len(acf)):
        y0, y1 = acf[k - 1], acf[k]
        if np.isfinite(y0) and np.isfinite(y1) and y0 > thr and y1 <= thr:
            t = interp_cross((k - 1) * dt, y0, k * dt, y1, thr)
            return t, "downward_crossing"
    return np.nan, "censored"


def tm_condition(cfg: TMConfig):
    rec, dt, d = tm_simulate_ensemble(cfg)
    max_lag = max(10, min(int(round(cfg.max_lag_fraction * cfg.n_steps)), cfg.n_steps - 2))
    max_lag_time = max_lag * dt
    rows = []
    for r in range(cfg.n_reps):
        acf = acf_fft_1d(rec[:, r], max_lag)
        t_corr, status = tcorr_from_acf(acf, dt)
        chi = cfg.gamma * t_corr if np.isfinite(t_corr) else np.nan
        rows.append({
            "gamma": cfg.gamma,
            "beta": cfg.beta,
            "tau": cfg.tau,
            "sigma": cfg.sigma,
            "rep": r,
            "ensemble_seed": cfg.base_seed,
            "dt": dt,
            "delay_steps": d,
            "n_steps": cfg.n_steps,
            "max_lag_time": max_lag_time,
            "t_corr": t_corr,
            "chi": chi,
            "acf_status": status,
            "x_mean": float(np.mean(rec[:, r])),
            "x_sd": float(np.std(rec[:, r], ddof=1)),
        })
    return pd.DataFrame(rows)


def summarize_tm(raw):
    rows = []
    for (gamma, beta, tau, sigma), g in raw.groupby(["gamma", "beta", "tau", "sigma"], sort=True):
        c = finite(g["chi"])
        rows.append({
            "gamma": gamma,
            "beta": beta,
            "tau": tau,
            "sigma": sigma,
            "n_total": len(g),
            "n_valid": len(c),
            "valid_fraction": len(c) / len(g),
            "chi_mean": mean(c),
            "chi_sd": sd(c),
            "chi_sem": sem(c),
            "t_corr_mean": mean(g["t_corr"]),
            "x_sd_mean": mean(g["x_sd"]),
        })
    return pd.DataFrame(rows)


def first_chi_crossing(summary):
    s = summary.sort_values("tau")
    x = s["tau"].to_numpy(float)
    y = s["chi_mean"].to_numpy(float)
    for i in range(1, len(x)):
        if np.isfinite(y[i - 1]) and np.isfinite(y[i]) and y[i - 1] < 1.0 <= y[i]:
            return interp_cross(x[i - 1], y[i - 1], x[i], y[i], 1.0)
    return np.nan


# -----------------------------------------------------------------------------
# Minimal causal-closure model
# -----------------------------------------------------------------------------

@dataclass
class ClosureConfig:
    tau_diss: float = 1.0
    causal_delay: float = 3.0
    tau_env: float = 5.0
    env_sd: float = 1.0
    action_gain: float = 1.0
    viability_width: float = 0.5
    repair_rate: float = 0.2
    decay_rate: float = 0.2
    b_critical: float = 0.5
    dt: float = 0.01
    warmup_time: float = 100.0
    record_time: float = 200.0
    shuffle_min_lag_time: float = 15.0
    shuffle_max_lag_time: float = 50.0


INTERVENTIONS = [
    "intact",
    "history_shuffle",
    "reuse_knockout",
    "retention_knockout",
    "closure_cut",
    "closure_yoked",
    "external_maintenance",
]


def generate_environment(cfg: ClosureConfig, n_reps, seed):
    n_total = int(round((cfg.warmup_time + cfg.record_time) / cfg.dt))
    rng = np.random.default_rng(seed)
    e = np.empty((n_total, n_reps), dtype=np.float64)
    state = rng.normal(0.0, cfg.env_sd, size=n_reps)
    e[0] = state
    sqrt_term = math.sqrt(2.0 * cfg.env_sd * cfg.env_sd / cfg.tau_env) * math.sqrt(cfg.dt)
    for t in range(1, n_total):
        state = state + (-state / cfg.tau_env) * cfg.dt + sqrt_term * rng.normal(size=n_reps)
        e[t] = state
    return e


def simulate_closure_condition(chi, mode, cfg: ClosureConfig, environment, shuffle_seed=0, b_replay=None, return_b_trace=False):
    if mode not in INTERVENTIONS:
        raise ValueError(f"Unknown intervention: {mode}")

    e = np.asarray(environment, dtype=float)
    n_total, n_reps = e.shape
    warm_steps = int(round(cfg.warmup_time / cfg.dt))
    delay_steps = max(1, int(round(cfg.causal_delay / cfg.dt)))
    if delay_steps >= warm_steps:
        raise ValueError("causal_delay must be shorter than warmup_time")

    h = np.zeros(n_reps, dtype=float)
    b = np.ones(n_reps, dtype=float)
    if mode == "closure_yoked":
        if b_replay is None:
            raise ValueError("closure_yoked requires b_replay")
        b_replay = np.asarray(b_replay, dtype=float)
        if b_replay.shape != (n_total, n_reps):
            raise ValueError(f"b_replay shape {b_replay.shape} != {(n_total, n_reps)}")
    b_trace = np.empty((n_total, n_reps), dtype=np.float64) if return_b_trace else None

    min_lag = max(1, int(round(cfg.shuffle_min_lag_time / cfg.dt)))
    max_lag = max(min_lag + 1, int(round(cfg.shuffle_max_lag_time / cfg.dt)))
    hbuf_len = max_lag + 3
    hbuf = np.zeros((hbuf_len, n_reps), dtype=np.float64)
    hpos = 0
    rng_shuffle = np.random.default_rng(shuffle_seed)

    sum_v = np.zeros(n_reps)
    sum_b = np.zeros(n_reps)
    sum_b_ok = np.zeros(n_reps)
    sum_m2 = np.zeros(n_reps)
    sum_a2 = np.zeros(n_reps)
    sum_tau_r = np.zeros(n_reps)
    sum_h = np.zeros(n_reps)
    sum_h2 = np.zeros(n_reps)
    sum_ed = np.zeros(n_reps)
    sum_ed2 = np.zeros(n_reps)
    sum_hed = np.zeros(n_reps)
    count = 0

    for t in range(n_total):
        active = (t >= warm_steps)
        if active and mode == "closure_yoked":
            b = np.clip(b_replay[t].copy(), 0.0, 1.0)
        elif active and mode == "external_maintenance":
            b = np.ones(n_reps)

        b_used = b.copy()
        if return_b_trace:
            b_trace[t] = b_used

        if active and mode == "retention_knockout":
            tau_r = np.full(n_reps, cfg.tau_diss)
        else:
            tau_r = cfg.tau_diss * (1.0 + b_used * (chi - 1.0))
            tau_r = np.maximum(0.05 * cfg.tau_diss, tau_r)

        h = h + ((e[t] - h) / tau_r) * cfg.dt
        hbuf[hpos] = h

        if active and mode == "history_shuffle" and t > max_lag:
            offsets = rng_shuffle.integers(min_lag, max_lag + 1, size=n_reps)
            idx = (hpos - offsets) % hbuf_len
            h_use = hbuf[idx, np.arange(n_reps)]
        elif active and mode == "reuse_knockout":
            h_use = np.zeros(n_reps)
        else:
            h_use = h

        if active and mode == "reuse_knockout":
            action = np.zeros(n_reps)
        else:
            action = cfg.action_gain * b_used * h_use

        if t >= delay_steps:
            e_delayed = e[t - delay_steps]
        else:
            e_delayed = np.zeros(n_reps)

        mismatch = e_delayed - action
        viability = np.exp(-0.5 * (mismatch / cfg.viability_width) ** 2)

        if active and mode == "external_maintenance":
            b = np.ones(n_reps)
        elif active and mode == "closure_yoked":
            # Focal viability cannot update b; b is supplied exogenously by a partner intact trajectory.
            b = b_used
        elif active and mode == "closure_cut":
            # Total-effect intervention: remove viability-dependent maintenance after organization exists.
            b = np.clip(b_used + (-cfg.decay_rate * b_used) * cfg.dt, 0.0, 1.0)
        else:
            db = (
                cfg.repair_rate * viability * (1.0 - b_used)
                - cfg.decay_rate * (1.0 - viability) * b_used
            )
            b = np.clip(b_used + db * cfg.dt, 0.0, 1.0)

        if t >= warm_steps:
            sum_v += viability
            sum_b += b_used
            sum_b_ok += (b_used >= cfg.b_critical)
            sum_m2 += mismatch * mismatch
            sum_a2 += action * action
            sum_tau_r += tau_r
            sum_h += h_use
            sum_h2 += h_use * h_use
            sum_ed += e_delayed
            sum_ed2 += e_delayed * e_delayed
            sum_hed += h_use * e_delayed
            count += 1

        hpos = (hpos + 1) % hbuf_len

    mean_h = sum_h / count
    mean_ed = sum_ed / count
    cov = sum_hed / count - mean_h * mean_ed
    var_h = np.maximum(sum_h2 / count - mean_h * mean_h, 0.0)
    var_ed = np.maximum(sum_ed2 / count - mean_ed * mean_ed, 0.0)
    corr = cov / np.sqrt(np.maximum(var_h * var_ed, EPS))

    rows = []
    for r in range(n_reps):
        rows.append({
            "rep": r,
            "chi_mechanism": float(chi),
            "mode": mode,
            "causal_delay": cfg.causal_delay,
            "tau_env": cfg.tau_env,
            "mean_viability": float(sum_v[r] / count),
            "mean_integrity_b": float(sum_b[r] / count),
            "final_integrity_b": float(b[r]),
            "integrity_fraction_above_critical": float(sum_b_ok[r] / count),
            "mismatch_rms": float(math.sqrt(sum_m2[r] / count)),
            "action_rms": float(math.sqrt(sum_a2[r] / count)),
            "mean_effective_retention_time": float(sum_tau_r[r] / count),
            "history_impact_corr": float(corr[r]) if np.isfinite(corr[r]) else np.nan,
        })
    df = pd.DataFrame(rows)
    if return_b_trace:
        return df, b_trace
    return df


def summarize_closure(raw):
    keys = ["tm_tau", "chi_mechanism", "mode", "causal_delay", "tau_env"]
    rows = []
    for vals, g in raw.groupby(keys, sort=True, dropna=False):
        r = dict(zip(keys, vals))
        r.update({
            "n": len(g),
            "viability_mean": mean(g["mean_viability"]),
            "viability_sd": sd(g["mean_viability"]),
            "viability_sem": sem(g["mean_viability"]),
            "integrity_mean": mean(g["mean_integrity_b"]),
            "integrity_sd": sd(g["mean_integrity_b"]),
            "integrity_final_mean": mean(g["final_integrity_b"]),
            "integrity_fraction_mean": mean(g["integrity_fraction_above_critical"]),
            "mismatch_rms_mean": mean(g["mismatch_rms"]),
            "action_rms_mean": mean(g["action_rms"]),
            "effective_retention_mean": mean(g["mean_effective_retention_time"]),
            "history_impact_corr_mean": mean(g["history_impact_corr"]),
        })
        rows.append(r)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Paired inference
# -----------------------------------------------------------------------------


def paired_bootstrap_and_signflip(diffs, rng, n_boot=5000, n_perm=5000):
    d = finite(diffs)
    n = len(d)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    obs = float(np.mean(d))
    if n == 1:
        return obs, np.nan, np.nan, np.nan, np.nan

    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    perm_means = (signs * d[None, :]).mean(axis=1)
    p = (1.0 + np.sum(np.abs(perm_means) >= abs(obs))) / (n_perm + 1.0)
    dz = obs / np.std(d, ddof=1) if np.std(d, ddof=1) > EPS else np.nan
    return obs, float(lo), float(hi), float(p), float(dz)


def paired_contrast_table(raw, n_boot, n_perm, seed=909090):
    # (base mode, comparator mode, label). Positive values support the named causal contribution.
    comparisons = [
        ("intact", "history_shuffle", "history_alignment"),
        ("intact", "reuse_knockout", "causal_reuse"),
        ("intact", "retention_knockout", "extended_retention"),
        ("intact", "closure_cut", "self_maintenance_closure_total_effect"),
        ("closure_yoked", "closure_cut", "closure_yoked_rescue"),
        ("intact", "closure_yoked", "intact_vs_yoked_rescue"),
        ("intact", "external_maintenance", "external_vs_autonomous"),
    ]
    rng = np.random.default_rng(seed)
    rows = []
    for (tm_tau, chi, delta), g in raw.groupby(["tm_tau", "chi_mechanism", "causal_delay"], sort=True):
        for base_mode, other_mode, label in comparisons:
            base = g[g["mode"] == base_mode].set_index("rep")
            other = g[g["mode"] == other_mode].set_index("rep")
            common = base.index.intersection(other.index)
            if len(common) == 0:
                continue
            d = base.loc[common, "mean_viability"].to_numpy() - other.loc[common, "mean_viability"].to_numpy()
            est, lo, hi, p, dz = paired_bootstrap_and_signflip(d, rng, n_boot, n_perm)
            rows.append({
                "tm_tau": tm_tau,
                "chi_mechanism": chi,
                "causal_delay": delta,
                "base_mode": base_mode,
                "other_mode": other_mode,
                "contrast": label,
                "base_minus_other_mean": est,
                "intact_minus_other_mean": est if base_mode == "intact" else np.nan,
                "ci95_low": lo,
                "ci95_high": hi,
                "signflip_p_two_sided": p,
                "paired_dz": dz,
                "n_pairs": len(common),
            })
    return pd.DataFrame(rows)


def retention_window_table(contrasts):
    g = contrasts[contrasts["contrast"] == "extended_retention"].sort_values("chi_mechanism")
    rows = []
    for delta, d in g.groupby("causal_delay"):
        pos = d[d["ci95_low"] > 0]
        neg = d[d["ci95_high"] < 0]
        rows.append({
            "causal_delay": delta,
            "first_chi_with_positive_ci": float(pos["chi_mechanism"].iloc[0]) if len(pos) else np.nan,
            "last_chi_with_positive_ci": float(pos["chi_mechanism"].iloc[-1]) if len(pos) else np.nan,
            "n_positive_ci_points": len(pos),
            "n_negative_ci_points": len(neg),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Run plan
# -----------------------------------------------------------------------------


def plan(mode):
    if mode == "smoke":
        return {
            "tm_tau_grid": np.array([1.0, 2.0, 5.0, 10.0]),
            "tm_reps": 4,
            "tm_steps": 10000,
            "causal_reps": 6,
            "closure_record_time": 40.0,
            "closure_warmup_time": 60.0,
            "delta_grid": np.array([1.0, 2.0, 3.0, 4.0]),
            "n_boot": 500,
            "n_perm": 500,
        }
    return {
        "tm_tau_grid": np.array([0.5, 1.0, 1.5, 1.9, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]),
        "tm_reps": 20,
        "tm_steps": 80000,
        "causal_reps": 32,
        "closure_record_time": 200.0,
        "closure_warmup_time": 100.0,
        "delta_grid": np.arange(0.5, 4.5 + 0.001, 0.5),
        "n_boot": 5000,
        "n_perm": 5000,
    }


def ensure_dirs(out):
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)


def save_json(path, obj):
    def conv(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return x
    with open(path, "w", encoding="utf-8") as f:
        json.dump({k: conv(v) for k, v in obj.items()}, f, indent=2)


def stage0_and_1_tm(p, out, lf):
    raw_parts = []
    log("[Stage 0/5] Temporal Membranes regression and retention map", lf)
    for i, tau in enumerate(p["tm_tau_grid"]):
        log(f"  TM condition {i+1}/{len(p['tm_tau_grid'])}: tau={tau:g}", lf)
        cfg = TMConfig(
            tau=float(tau),
            n_steps=int(p["tm_steps"]),
            n_reps=int(p["tm_reps"]),
            base_seed=100000 + i * 10000,
        )
        raw_parts.append(tm_condition(cfg))
    raw = pd.concat(raw_parts, ignore_index=True)
    summ = summarize_tm(raw)
    raw.to_csv(out / "01_temporal_membrane_retention_raw.csv", index=False)
    summ.to_csv(out / "02_temporal_membrane_retention_summary.csv", index=False)
    tau_c = first_chi_crossing(summ)

    # Regression audit against published regime; broad tolerances avoid false failures in smoke mode.
    audit = []
    def get_chi(t):
        z = summ[np.isclose(summ["tau"], t)]
        return float(z["chi_mean"].iloc[0]) if len(z) else np.nan
    chi1 = get_chi(1.0)
    chi2 = get_chi(2.0)
    maxchi = float(summ.sort_values("tau")["chi_mean"].iloc[-1])
    audit.append(("chi_tau1_below_1", bool(np.isfinite(chi1) and chi1 < 1.0), chi1))
    audit.append(("chi_tau2_near_or_above_1", bool(np.isfinite(chi2) and chi2 > 0.90), chi2))
    audit.append(("large_delay_retention_above_1", bool(np.isfinite(maxchi) and maxchi > 1.0), maxchi))
    if np.isfinite(tau_c):
        audit.append(("crossing_in_expected_region", bool(1.5 <= tau_c <= 2.5), tau_c))
    else:
        audit.append(("crossing_in_expected_region", False, tau_c))
    adf = pd.DataFrame(audit, columns=["check", "pass", "value"])
    adf.to_csv(out / "00_regression_audit.csv", index=False)
    log(f"  Estimated chi=1 crossing tau_c={tau_c}", lf)
    log(f"  Regression audit: {int(adf['pass'].sum())}/{len(adf)} checks passed", lf)
    return raw, summ, adf


def make_environment_for_cfg(cfg, n_reps, seed):
    return generate_environment(cfg, n_reps=n_reps, seed=seed)


def run_primary_closure(p, tm_summary, out, lf):
    log("[Stage 2/5] Primary causal-closure interventions", lf)
    cfg = ClosureConfig(
        causal_delay=3.0,
        warmup_time=float(p["closure_warmup_time"]),
        record_time=float(p["closure_record_time"]),
    )
    env = make_environment_for_cfg(cfg, p["causal_reps"], seed=700001)
    raw_parts = []
    yoke_audit_rows = []
    ordered = tm_summary.sort_values("tau").reset_index(drop=True)
    total = len(ordered) * len(INTERVENTIONS)
    c = 0
    for _, rr in ordered.iterrows():
        tm_tau = float(rr["tau"])
        chi = float(rr["chi_mean"])
        if not np.isfinite(chi):
            continue

        c += 1
        log(f"  closure {c}/{total}: tm_tau={tm_tau:g}, chi={chi:.3f}, mode=intact", lf)
        intact, b_trace = simulate_closure_condition(
            chi=chi, mode="intact", cfg=cfg, environment=env,
            shuffle_seed=800000 + int(round(tm_tau * 1000)), return_b_trace=True,
        )
        intact["tm_tau"] = tm_tau
        raw_parts.append(intact)

        # Derangement across matched replicate columns. At every time point this preserves
        # the complete distribution of b(t), but the organizational state no longer comes
        # from the focal replicate's own viability history.
        if b_trace.shape[1] < 2:
            raise ValueError("closure_yoked requires at least two causal replicates")
        yoked_trace = np.roll(b_trace, shift=1, axis=1)

        for mi, mode in enumerate([m for m in INTERVENTIONS if m != "intact"]):
            c += 1
            log(f"  closure {c}/{total}: tm_tau={tm_tau:g}, chi={chi:.3f}, mode={mode}", lf)
            d = simulate_closure_condition(
                chi=chi, mode=mode, cfg=cfg, environment=env,
                shuffle_seed=810000 + int(round(tm_tau * 1000)) + mi * 10000,
                b_replay=yoked_trace if mode == "closure_yoked" else None,
            )
            d["tm_tau"] = tm_tau
            raw_parts.append(d)

            if mode == "closure_yoked":
                yoke_audit_rows.append({
                    "tm_tau": tm_tau,
                    "chi_mechanism": chi,
                    "intact_integrity_grand_mean": float(intact["mean_integrity_b"].mean()),
                    "yoked_integrity_grand_mean": float(d["mean_integrity_b"].mean()),
                    "integrity_mean_difference": float(intact["mean_integrity_b"].mean() - d["mean_integrity_b"].mean()),
                    "intact_effective_retention_grand_mean": float(intact["mean_effective_retention_time"].mean()),
                    "yoked_effective_retention_grand_mean": float(d["mean_effective_retention_time"].mean()),
                    "retention_mean_difference": float(intact["mean_effective_retention_time"].mean() - d["mean_effective_retention_time"].mean()),
                })

    raw = pd.concat(raw_parts, ignore_index=True)
    summ = summarize_closure(raw)
    raw.to_csv(out / "03_primary_causal_closure_raw.csv", index=False)
    summ.to_csv(out / "04_primary_causal_closure_summary.csv", index=False)
    yoke_audit = pd.DataFrame(yoke_audit_rows)
    yoke_audit.to_csv(out / "04B_closure_yoke_audit.csv", index=False)
    return raw, summ, yoke_audit


def run_inference(p, primary_raw, out, lf):
    log("[Stage 3/5] Paired causal contrasts", lf)
    contrasts = paired_contrast_table(primary_raw, p["n_boot"], p["n_perm"])
    contrasts.to_csv(out / "05_paired_causal_contrasts.csv", index=False)
    window = retention_window_table(contrasts)
    window.to_csv(out / "06_retention_window_summary.csv", index=False)
    return contrasts, window


def run_delay_scaling(p, tm_summary, out, lf):
    log("[Stage 4/5] Fine causal-delay scaling", lf)
    raw_parts = []
    ordered = tm_summary.sort_values("tau").reset_index(drop=True)
    total = len(p["delta_grid"]) * len(ordered) * 2
    c = 0

    # Identical environmental trajectories are reused for every causal delay.
    # Therefore a shift in optimal retention cannot be attributed to a different noise realization.
    base_cfg = ClosureConfig(
        causal_delay=float(p["delta_grid"][0]),
        warmup_time=float(p["closure_warmup_time"]),
        record_time=float(p["closure_record_time"]),
    )
    shared_env = make_environment_for_cfg(base_cfg, p["causal_reps"], seed=710000)

    for di, delta in enumerate(p["delta_grid"]):
        cfg = ClosureConfig(
            causal_delay=float(delta),
            warmup_time=float(p["closure_warmup_time"]),
            record_time=float(p["closure_record_time"]),
            shuffle_min_lag_time=max(15.0, 2.0 * float(delta)),
            shuffle_max_lag_time=50.0,
        )
        for _, rr in ordered.iterrows():
            tm_tau = float(rr["tau"])
            chi = float(rr["chi_mean"])
            if not np.isfinite(chi):
                continue
            for mode in ["intact", "retention_knockout"]:
                c += 1
                log(f"  delay-scaling {c}/{total}: delta={delta:g}, chi={chi:.3f}, mode={mode}", lf)
                d = simulate_closure_condition(
                    chi=chi, mode=mode, cfg=cfg, environment=shared_env,
                    shuffle_seed=820000 + int(round(tm_tau * 100)),
                )
                d["tm_tau"] = tm_tau
                raw_parts.append(d)
    raw = pd.concat(raw_parts, ignore_index=True)
    summ = summarize_closure(raw)
    raw.to_csv(out / "07_causal_delay_scaling_raw.csv", index=False)
    summ.to_csv(out / "08_causal_delay_scaling_summary.csv", index=False)

    opt_rows = []
    for delta, g in summ[summ["mode"] == "intact"].groupby("causal_delay"):
        g = g.sort_values("chi_mechanism")
        best = g.loc[g["viability_mean"].idxmax()]
        ko = summ[(summ["causal_delay"] == delta) & (summ["mode"] == "retention_knockout")]
        merged = g.merge(ko[["chi_mechanism", "viability_mean"]], on="chi_mechanism", suffixes=("_intact", "_ko"))
        merged["retention_benefit"] = merged["viability_mean_intact"] - merged["viability_mean_ko"]
        best_benefit = merged.loc[merged["retention_benefit"].idxmax()]
        opt_rows.append({
            "causal_delay": float(delta),
            "optimal_chi_for_viability": float(best["chi_mechanism"]),
            "optimal_tm_tau_for_viability": float(best["tm_tau"]),
            "optimal_effective_retention_time": float(best["effective_retention_mean"]),
            "optimal_retention_to_causal_delay": float(best["effective_retention_mean"] / delta),
            "max_viability": float(best["viability_mean"]),
            "chi_of_max_retention_benefit": float(best_benefit["chi_mechanism"]),
            "max_retention_benefit": float(best_benefit["retention_benefit"]),
        })
    opt = pd.DataFrame(opt_rows).sort_values("causal_delay")
    if len(opt) >= 2:
        x = opt["causal_delay"].to_numpy(float)
        y = opt["optimal_effective_retention_time"].to_numpy(float)
        slope, intercept = np.polyfit(x, y, 1)
        pred = intercept + slope * x
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > EPS else np.nan
        opt["optimal_retention_vs_delay_slope"] = slope
        opt["optimal_retention_vs_delay_intercept"] = intercept
        opt["optimal_retention_vs_delay_r2"] = r2
        opt["median_optimal_retention_to_delay"] = float(np.median(opt["optimal_retention_to_causal_delay"]))
    opt.to_csv(out / "09_causal_delay_optimal_retention.csv", index=False)
    return raw, summ, opt


# -----------------------------------------------------------------------------
# Figures and report
# -----------------------------------------------------------------------------


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def make_figures(tm_summary, primary_summary, contrasts, delay_summary, opt, out, lf):
    fd = out / "figures"

    plt.figure(figsize=(7, 5))
    s = tm_summary.sort_values("tau")
    plt.errorbar(s["tau"], s["chi_mean"], yerr=s["chi_sem"], marker="o", capsize=2)
    plt.axhline(1.0, linestyle="--")
    plt.xlabel("Temporal-membrane delay τ")
    plt.ylabel("Self-retention index χ")
    plt.title("Fig. 1. Delay generates retention beyond dissipation")
    savefig(fd / "fig1_temporal_retention_map.png")

    plt.figure(figsize=(7, 5))
    for mode, g in primary_summary.groupby("mode"):
        g = g.sort_values("chi_mechanism")
        plt.plot(g["chi_mechanism"], g["viability_mean"], marker="o", label=mode)
    plt.axvline(1.0, linestyle="--")
    plt.xlabel("Mechanistic retention χ")
    plt.ylabel("Mean viability")
    plt.title("Fig. 2. Causal closure across retention regimes")
    plt.legend(fontsize=8)
    savefig(fd / "fig2_primary_viability_by_chi.png")

    plt.figure(figsize=(7, 5))
    for contrast_name in ["history_alignment", "causal_reuse", "extended_retention", "self_maintenance_closure_total_effect", "closure_yoked_rescue"]:
        g = contrasts[contrasts["contrast"] == contrast_name].sort_values("chi_mechanism")
        if len(g):
            plt.plot(g["chi_mechanism"], g["base_minus_other_mean"], marker="o", label=contrast_name)
    plt.axhline(0.0, linestyle="--")
    plt.axvline(1.0, linestyle=":")
    plt.xlabel("Mechanistic retention χ")
    plt.ylabel("Paired viability effect: intact − intervention")
    plt.title("Fig. 3. Causal intervention effects")
    plt.legend(fontsize=8)
    savefig(fd / "fig3_causal_intervention_effects.png")

    plt.figure(figsize=(7, 5))
    for delta, g in delay_summary[delay_summary["mode"] == "intact"].groupby("causal_delay"):
        g = g.sort_values("chi_mechanism")
        plt.plot(g["chi_mechanism"], g["viability_mean"], marker="o", label=f"δ={delta:g}")
    plt.axvline(1.0, linestyle="--")
    plt.xlabel("Mechanistic retention χ")
    plt.ylabel("Mean viability")
    plt.title("Fig. 4. The useful retention window shifts with causal delay")
    plt.legend()
    savefig(fd / "fig4_causal_delay_scaling.png")

    if len(opt):
        plt.figure(figsize=(7, 5))
        plt.plot(opt["causal_delay"], opt["optimal_effective_retention_time"], marker="o")
        lo = min(float(opt["causal_delay"].min()), float(opt["optimal_effective_retention_time"].min()))
        hi = max(float(opt["causal_delay"].max()), float(opt["optimal_effective_retention_time"].max()))
        plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
        plt.xlabel("Causal delay δ")
        plt.ylabel("Optimal effective retention time")
        plt.title("Fig. 5. Retention timescale tracks loop delay")
        savefig(fd / "fig5_optimal_retention_vs_causal_delay.png")

    log("[Figures] saved", lf)


def write_report(p, tm_summary, audit, primary_summary, contrasts, window, yoke_audit, opt, out, lf):
    path = out / "10_analysis_report.md"
    tau_c = first_chi_crossing(tm_summary)
    ext = contrasts[contrasts["contrast"] == "extended_retention"]
    hist = contrasts[contrasts["contrast"] == "history_alignment"]
    reuse = contrasts[contrasts["contrast"] == "causal_reuse"]
    close = contrasts[contrasts["contrast"] == "self_maintenance_closure_total_effect"]
    rescue = contrasts[contrasts["contrast"] == "closure_yoked_rescue"]
    yokeeq = contrasts[contrasts["contrast"] == "intact_vs_yoked_rescue"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("# What Is Life? Causal Persistence Against Dissipation — Validation Report\n\n")
        f.write("## Stage 0: numerical regression\n\n")
        f.write(f"Estimated Temporal Membranes χ=1 crossing: {tau_c:.6g}\n\n" if np.isfinite(tau_c) else "Estimated χ=1 crossing: not resolved.\n\n")
        f.write(audit.to_string(index=False))
        f.write("\n\n## Stage 1: retention map\n\n")
        f.write(tm_summary[["tau", "chi_mean", "chi_sem", "valid_fraction"]].to_string(index=False))
        f.write("\n\n## Stage 2–3: causal-closure tests\n\n")
        f.write("Primary model conditions were evaluated under identical external trajectories for matched replicates.\n\n")
        f.write("### Extended-retention contrast (intact − retention knockout)\n\n")
        f.write(ext[["tm_tau", "chi_mechanism", "intact_minus_other_mean", "ci95_low", "ci95_high", "signflip_p_two_sided"]].to_string(index=False))
        f.write("\n\n### History-alignment contrast\n\n")
        f.write(hist[["tm_tau", "chi_mechanism", "intact_minus_other_mean", "ci95_low", "ci95_high"]].to_string(index=False))
        f.write("\n\n### Causal-reuse contrast\n\n")
        f.write(reuse[["tm_tau", "chi_mechanism", "intact_minus_other_mean", "ci95_low", "ci95_high"]].to_string(index=False))
        f.write("\n\n### Closure total-effect contrast (intact − closure cut)\n\n")
        f.write(close[["tm_tau", "chi_mechanism", "base_minus_other_mean", "ci95_low", "ci95_high"]].to_string(index=False))
        f.write("\n\n### Yoked organizational rescue (closure-yoked − closure cut)\n\n")
        f.write(rescue[["tm_tau", "chi_mechanism", "base_minus_other_mean", "ci95_low", "ci95_high"]].to_string(index=False))
        f.write("\n\n### Intact versus yoked rescue\n\n")
        f.write(yokeeq[["tm_tau", "chi_mechanism", "base_minus_other_mean", "ci95_low", "ci95_high"]].to_string(index=False))
        f.write("\n\n### Retention-window summary\n\n")
        f.write(window.to_string(index=False))
        f.write("\n\n### Closure-yoke integrity audit\n\n")
        f.write(yoke_audit.to_string(index=False))
        f.write("\n\n## Stage 4: causal-delay scaling\n\n")
        f.write(opt.to_string(index=False))
        f.write("\n\n## Pre-specified interpretation rule\n\n")
        f.write(
            "Support for the temporal causal-closure hypothesis requires all of the following: "
            "(i) the independent delay-dissipative assay resolves χ<1 and χ>1 regimes; "
            "(ii) intact viability exceeds temporally shuffled-history and causal-reuse knockout controls over a nontrivial retention range; "
            "(iii) extended-retention benefit is positive for at least one χ>1 condition relative to the retention knockout; "
            "(iv) yoking the organizational state to a partner trajectory while preserving its marginal statistics reduces focal viability; and "
            "(v) the retention value maximizing viability shifts upward as causal delay increases. "
            "Failure of any item is reported as a failure of that component rather than repaired by post-hoc parameter selection.\n"
        )
    log(f"[Report] saved {path}", lf)



# =============================================================================
# Comprehensive validation extension
# =============================================================================

COMPREHENSIVE_OUTPUT = Path.home() / "Desktop" / "WHAT_IS_LIFE_COMPREHENSIVE_VALIDATION"


def generate_environment_kind(cfg: ClosureConfig, n_reps, seed, kind="ou"):
    kind = str(kind).lower()
    if kind == "ou":
        return generate_environment(cfg, n_reps=n_reps, seed=seed)
    if kind != "telegraph":
        raise ValueError(f"Unknown environment kind: {kind}")
    n_total = int(round((cfg.warmup_time + cfg.record_time) / cfg.dt))
    rng = np.random.default_rng(seed)
    e = np.empty((n_total, n_reps), dtype=np.float64)
    state = rng.choice(np.array([-cfg.env_sd, cfg.env_sd]), size=n_reps)
    e[0] = state
    # For a symmetric two-state telegraph process, C(t) ~ exp(-2 lambda t).
    # lambda = 1/(2 tau_env) makes tau_env the correlation timescale.
    p_switch = min(1.0, cfg.dt / max(2.0 * cfg.tau_env, EPS))
    for t in range(1, n_total):
        sw = rng.random(n_reps) < p_switch
        state = np.where(sw, -state, state)
        e[t] = state
    return e


def simulate_intact_grid(chi_grid, cfg: ClosureConfig, environment, b_initial=1.0, retention_knockout=False):
    """Simulate all retention capacities simultaneously under one shared environment.

    Returns replicate-level summaries for every chi. This is the workhorse for dense
    scaling and robustness stages and avoids a Python time loop per chi value.
    """
    chis = np.asarray(chi_grid, dtype=float)
    e = np.asarray(environment, dtype=float)
    n_total, n_reps = e.shape
    n_chi = len(chis)
    warm_steps = int(round(cfg.warmup_time / cfg.dt))
    delay_steps = max(1, int(round(cfg.causal_delay / cfg.dt)))
    if delay_steps >= warm_steps:
        raise ValueError("causal_delay must be shorter than warmup_time")

    h = np.zeros((n_chi, n_reps), dtype=float)
    b = np.full((n_chi, n_reps), float(b_initial), dtype=float)
    sum_v = np.zeros_like(h)
    sum_b = np.zeros_like(h)
    sum_tau = np.zeros_like(h)
    count = 0

    chi2 = chis[:, None]
    for t in range(n_total):
        active = t >= warm_steps
        if active and retention_knockout:
            tau_r = np.full_like(h, cfg.tau_diss)
        else:
            tau_r = cfg.tau_diss * (1.0 + b * (chi2 - 1.0))
            tau_r = np.maximum(0.05 * cfg.tau_diss, tau_r)

        h += ((e[t][None, :] - h) / tau_r) * cfg.dt
        action = cfg.action_gain * b * h
        ed = e[t - delay_steps] if t >= delay_steps else np.zeros(n_reps)
        mismatch = ed[None, :] - action
        viability = np.exp(-0.5 * (mismatch / cfg.viability_width) ** 2)
        db = (
            cfg.repair_rate * viability * (1.0 - b)
            - cfg.decay_rate * (1.0 - viability) * b
        )
        b = np.clip(b + db * cfg.dt, 0.0, 1.0)

        if active:
            sum_v += viability
            sum_b += b
            sum_tau += tau_r
            count += 1

    rows = []
    for i, chi in enumerate(chis):
        for r in range(n_reps):
            rows.append({
                "chi_mechanism": float(chi),
                "rep": int(r),
                "mean_viability": float(sum_v[i, r] / count),
                "mean_integrity_b": float(sum_b[i, r] / count),
                "mean_effective_retention_time": float(sum_tau[i, r] / count),
                "causal_delay": float(cfg.causal_delay),
                "tau_env": float(cfg.tau_env),
                "tau_diss": float(cfg.tau_diss),
                "retention_knockout": bool(retention_knockout),
            })
    return pd.DataFrame(rows)


def local_quadratic_optimum(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) == 0:
        return np.nan, np.nan, "none"
    order = np.argsort(x)
    x, y = x[order], y[order]
    imax = int(np.argmax(y))
    lo = max(0, imax - 2)
    hi = min(len(x), imax + 3)
    xx, yy = x[lo:hi], y[lo:hi]
    if len(xx) >= 3 and np.ptp(xx) > EPS:
        try:
            a, b, c = np.polyfit(xx, yy, 2)
            if a < 0:
                xv = -b / (2.0 * a)
                if float(xx.min()) <= xv <= float(xx.max()):
                    yv = a * xv * xv + b * xv + c
                    return float(xv), float(yv), "quadratic"
        except Exception:
            pass
    return float(x[imax]), float(y[imax]), "grid"


def optimum_from_grid_raw(raw):
    g = raw.groupby("chi_mechanism", sort=True).agg(
        viability=("mean_viability", "mean"),
        retention=("mean_effective_retention_time", "mean"),
    ).reset_index()
    r_opt, v_opt, method = local_quadratic_optimum(g["retention"], g["viability"])
    # Interpolate the chi associated with the estimated optimum retention time.
    gg = g.sort_values("retention")
    chi_opt = float(np.interp(r_opt, gg["retention"], gg["chi_mechanism"])) if np.isfinite(r_opt) else np.nan
    return r_opt, v_opt, chi_opt, method


def bootstrap_scaling(raw, n_boot=1000, seed=555001):
    """Shared-replicate bootstrap for tau_R* ~ tau_C scaling."""
    deltas = np.array(sorted(raw["causal_delay"].unique()), dtype=float)
    reps = np.array(sorted(raw["rep"].unique()), dtype=int)
    if len(reps) < 2 or len(deltas) < 2:
        return pd.DataFrame(), {}
    rng = np.random.default_rng(seed)
    slopes, intercepts, r2s = [], [], []
    opt_store = {float(d): [] for d in deltas}
    for _ in range(int(n_boot)):
        samp = rng.choice(reps, size=len(reps), replace=True)
        ys = []
        good_d = []
        for d in deltas:
            gd = raw[raw["causal_delay"] == d]
            parts = [gd[gd["rep"] == int(r)] for r in samp]
            gb = pd.concat(parts, ignore_index=True)
            r_opt, _, _, _ = optimum_from_grid_raw(gb)
            opt_store[float(d)].append(r_opt)
            if np.isfinite(r_opt):
                good_d.append(float(d)); ys.append(float(r_opt))
        if len(ys) >= 2:
            x = np.asarray(good_d); y = np.asarray(ys)
            slope, intercept = np.polyfit(x, y, 1)
            pred = intercept + slope * x
            ss_res = float(np.sum((y - pred) ** 2)); ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            slopes.append(float(slope)); intercepts.append(float(intercept)); r2s.append(float(1.0 - ss_res / ss_tot) if ss_tot > EPS else np.nan)
    bdf = pd.DataFrame({"slope": slopes, "intercept": intercepts, "r2": r2s})
    summary = {}
    for name, vals in [("slope", slopes), ("intercept", intercepts), ("r2", r2s)]:
        a = finite(vals)
        if len(a):
            summary[name+"_mean"] = float(np.mean(a))
            summary[name+"_ci95_low"] = float(np.percentile(a, 2.5))
            summary[name+"_ci95_high"] = float(np.percentile(a, 97.5))
    summary["p_slope_le_zero"] = float((1 + np.sum(np.asarray(slopes) <= 0)) / (1 + len(slopes))) if slopes else np.nan
    for d in deltas:
        a = finite(opt_store[float(d)])
        if len(a):
            summary[f"delta_{d:g}_opt_ci_low"] = float(np.percentile(a, 2.5))
            summary[f"delta_{d:g}_opt_ci_high"] = float(np.percentile(a, 97.5))
    return bdf, summary


def simulate_trace_assay(chi, cfg: ClosureConfig, environment, mode="intact", replay=None, b_initial=1.0, return_traces=False):
    """Single-chi intervention model with mediator/yoked replay support."""
    allowed = {
        "intact", "history_yoked", "action_yoked", "retention_cut", "reuse_cut",
        "closure_cut", "closure_yoked", "retention_cut_h_rescue", "reuse_cut_a_rescue"
    }
    if mode not in allowed:
        raise ValueError(mode)
    e = np.asarray(environment, dtype=float)
    n_total, n_reps = e.shape
    warm_steps = int(round(cfg.warmup_time / cfg.dt))
    delay_steps = max(1, int(round(cfg.causal_delay / cfg.dt)))
    h = np.zeros(n_reps, dtype=float)
    b = np.full(n_reps, float(b_initial), dtype=float)
    sum_v = np.zeros(n_reps); sum_b = np.zeros(n_reps); sum_tau = np.zeros(n_reps); count = 0
    traces = None
    if return_traces:
        traces = {
            "h": np.empty((n_total, n_reps), dtype=np.float64),
            "a": np.empty((n_total, n_reps), dtype=np.float64),
            "b": np.empty((n_total, n_reps), dtype=np.float64),
        }
    for t in range(n_total):
        active = t >= warm_steps
        if active and mode == "closure_yoked":
            b = np.clip(np.asarray(replay["b"])[t], 0.0, 1.0).copy()
        b_used = b.copy()
        if active and mode in {"retention_cut", "retention_cut_h_rescue"}:
            tau_r = np.full(n_reps, cfg.tau_diss)
        else:
            tau_r = cfg.tau_diss * (1.0 + b_used * (chi - 1.0))
            tau_r = np.maximum(0.05 * cfg.tau_diss, tau_r)
        h += ((e[t] - h) / tau_r) * cfg.dt
        h_use = h
        if active and mode in {"history_yoked", "retention_cut_h_rescue"}:
            h_use = np.asarray(replay["h"])[t]
        if active and mode == "reuse_cut":
            action = np.zeros(n_reps)
        elif active and mode in {"action_yoked", "reuse_cut_a_rescue"}:
            action = np.asarray(replay["a"])[t]
        else:
            action = cfg.action_gain * b_used * h_use
        ed = e[t - delay_steps] if t >= delay_steps else np.zeros(n_reps)
        mismatch = ed - action
        viability = np.exp(-0.5 * (mismatch / cfg.viability_width) ** 2)
        if active and mode == "closure_yoked":
            b = b_used
        elif active and mode == "closure_cut":
            b = np.clip(b_used + (-cfg.decay_rate * b_used) * cfg.dt, 0.0, 1.0)
        else:
            db = cfg.repair_rate * viability * (1.0 - b_used) - cfg.decay_rate * (1.0 - viability) * b_used
            b = np.clip(b_used + db * cfg.dt, 0.0, 1.0)
        if return_traces:
            traces["h"][t] = h
            traces["a"][t] = action
            traces["b"][t] = b_used
        if active:
            sum_v += viability; sum_b += b_used; sum_tau += tau_r; count += 1
    rows = []
    for r in range(n_reps):
        rows.append({
            "rep": r, "mode": mode, "chi_mechanism": float(chi),
            "causal_delay": float(cfg.causal_delay), "tau_env": float(cfg.tau_env), "tau_diss": float(cfg.tau_diss),
            "mean_viability": float(sum_v[r]/count), "mean_integrity_b": float(sum_b[r]/count),
            "mean_effective_retention_time": float(sum_tau[r]/count),
        })
    df = pd.DataFrame(rows)
    return (df, traces) if return_traces else df


def simple_paired_effect(raw, base_mode, other_mode, label, n_boot, n_perm, seed):
    a = raw[raw["mode"] == base_mode].set_index("rep")
    b = raw[raw["mode"] == other_mode].set_index("rep")
    common = a.index.intersection(b.index)
    d = a.loc[common, "mean_viability"].to_numpy() - b.loc[common, "mean_viability"].to_numpy()
    est, lo, hi, p, dz = paired_bootstrap_and_signflip(d, np.random.default_rng(seed), n_boot, n_perm)
    return {
        "contrast": label, "base_mode": base_mode, "other_mode": other_mode,
        "effect": est, "ci95_low": lo, "ci95_high": hi, "p_two_sided": p,
        "paired_dz": dz, "n_pairs": len(common),
    }


def comprehensive_plan(mode):
    if mode == "smoke":
        return {
            "dense_chi_grid": np.array([0.6, 0.9, 1.0, 1.3, 2.0, 3.0, 5.0]),
            "dense_delta_grid": np.array([1.0, 2.0, 3.0, 4.0]),
            "dense_reps": 6, "dense_boot": 100,
            "sentinel_chi": np.array([1.0, 3.0]),
            "tau_env_grid": np.array([2.5, 5.0]),
            "env_ratio_grid": np.array([0.5, 1.0, 1.25]),
            "tau_diss_grid": np.array([0.5, 1.0]),
            "diss_delta_ratio_grid": np.array([1.0, 3.0]),
            "robust_sets": 2, "robust_reps": 4,
            "robust_delta_grid": np.array([1.0, 3.0]),
            "robust_chi_grid": np.array([0.8, 1.5, 3.0, 5.0]),
            "numeric_dt_factors": np.array([0.5, 1.0, 2.0]),
            "numeric_record_times": np.array([40.0, 80.0]),
            "numeric_b0": np.array([0.1, 1.0]),
            "extra_record_time": 50.0, "extra_warmup_time": 60.0,
            "extra_n_boot": 300, "extra_n_perm": 300,
        }
    return {
        "dense_chi_grid": np.unique(np.concatenate([np.linspace(0.4, 1.4, 11), np.linspace(1.6, 6.8, 27)])),
        "dense_delta_grid": np.arange(0.5, 4.5 + 0.001, 0.25),
        "dense_reps": 32, "dense_boot": 1000,
        "sentinel_chi": np.array([0.8, 1.0, 2.0, 3.0, 5.0]),
        "tau_env_grid": np.array([2.5, 5.0, 10.0]),
        "env_ratio_grid": np.array([0.10, 0.25, 0.50, 0.75, 1.00, 1.25]),
        "tau_diss_grid": np.array([0.5, 1.0, 2.0]),
        "diss_delta_ratio_grid": np.array([0.5, 1.0, 2.0, 3.0, 4.0]),
        "robust_sets": 12, "robust_reps": 8,
        "robust_delta_grid": np.array([1.0, 2.0, 3.0, 4.0]),
        "robust_chi_grid": np.array([0.6, 0.8, 1.0, 1.4, 2.0, 3.0, 4.5, 6.0]),
        "numeric_dt_factors": np.array([0.5, 1.0, 2.0]),
        "numeric_record_times": np.array([100.0, 200.0, 400.0]),
        "numeric_b0": np.array([0.1, 0.5, 1.0]),
        "extra_record_time": 200.0, "extra_warmup_time": 100.0,
        "extra_n_boot": 2000, "extra_n_perm": 5000,
    }


def stage_done(out, names):
    return all((out / n).exists() and (out / n).stat().st_size > 0 for n in names)


def stage_dense_scaling(cp, out, lf):
    log("[Stage 5/10] Dense tau_R* ~ tau_C scaling with bootstrap", lf)
    cfg0 = ClosureConfig(causal_delay=float(cp["dense_delta_grid"][0]), warmup_time=cp["extra_warmup_time"], record_time=cp["extra_record_time"])
    env = generate_environment_kind(cfg0, cp["dense_reps"], seed=900001, kind="ou")
    parts = []
    for i, delta in enumerate(cp["dense_delta_grid"]):
        log(f"  dense scaling {i+1}/{len(cp['dense_delta_grid'])}: delta={delta:g}", lf)
        cfg = ClosureConfig(causal_delay=float(delta), warmup_time=cp["extra_warmup_time"], record_time=cp["extra_record_time"])
        g = simulate_intact_grid(cp["dense_chi_grid"], cfg, env)
        parts.append(g)
    raw = pd.concat(parts, ignore_index=True)
    raw.to_csv(out / "11_dense_causal_scaling_raw.csv", index=False)
    opts = []
    for d, g in raw.groupby("causal_delay", sort=True):
        r_opt, v_opt, chi_opt, method = optimum_from_grid_raw(g)
        opts.append({"causal_delay": d, "optimal_retention_time": r_opt, "optimal_viability": v_opt, "optimal_chi": chi_opt, "method": method, "ratio_retention_to_delay": r_opt/d})
    opt = pd.DataFrame(opts)
    x=opt["causal_delay"].to_numpy(float); y=opt["optimal_retention_time"].to_numpy(float)
    slope, intercept=np.polyfit(x,y,1); pred=intercept+slope*x
    ss_res=float(np.sum((y-pred)**2)); ss_tot=float(np.sum((y-np.mean(y))**2)); r2=1-ss_res/ss_tot if ss_tot>EPS else np.nan
    opt["global_slope"] = slope; opt["global_intercept"] = intercept; opt["global_r2"] = r2
    opt.to_csv(out / "12_dense_causal_scaling_optima.csv", index=False)
    bdf, bsum = bootstrap_scaling(raw, n_boot=cp["dense_boot"], seed=900002)
    bdf.to_csv(out / "13_dense_causal_scaling_bootstrap.csv", index=False)
    pd.DataFrame([bsum]).to_csv(out / "14_dense_causal_scaling_bootstrap_summary.csv", index=False)
    return raw, opt, bdf, bsum


def stage_yoked_and_rescue(cp, out, lf):
    log("[Stage 6/10] Yoked-history/action controls and edge-cut rescues", lf)
    cfg = ClosureConfig(causal_delay=3.0, warmup_time=cp["extra_warmup_time"], record_time=cp["extra_record_time"])
    env = generate_environment_kind(cfg, cp["dense_reps"], seed=910001, kind="ou")
    parts=[]; effects=[]
    for ci, chi in enumerate(cp["sentinel_chi"]):
        log(f"  causal trace assay {ci+1}/{len(cp['sentinel_chi'])}: chi={chi:g}", lf)
        intact, tr = simulate_trace_assay(float(chi), cfg, env, "intact", return_traces=True)
        intact["chi_sentinel"] = chi; parts.append(intact)
        partner = {k: np.roll(v, 1, axis=1) for k,v in tr.items()}
        modes = [
            ("history_yoked", partner), ("action_yoked", partner),
            ("retention_cut", None), ("retention_cut_h_rescue", tr),
            ("reuse_cut", None), ("reuse_cut_a_rescue", tr),
            ("closure_cut", None), ("closure_yoked", partner),
        ]
        local=[intact]
        for mode, replay in modes:
            d=simulate_trace_assay(float(chi), cfg, env, mode, replay=replay)
            d["chi_sentinel"] = chi; parts.append(d); local.append(d)
        lr=pd.concat(local, ignore_index=True)
        comps=[
            ("intact","history_yoked","history_yoked_effect"),
            ("intact","action_yoked","action_yoked_effect"),
            ("intact","retention_cut","retention_cut_effect"),
            ("retention_cut_h_rescue","retention_cut","history_mediator_rescue"),
            ("intact","reuse_cut","reuse_cut_effect"),
            ("reuse_cut_a_rescue","reuse_cut","action_mediator_rescue"),
            ("intact","closure_cut","closure_cut_effect"),
            ("closure_yoked","closure_cut","closure_yoked_rescue"),
        ]
        for j,(a,b,label) in enumerate(comps):
            rr=simple_paired_effect(lr,a,b,label,cp["extra_n_boot"],cp["extra_n_perm"],920000+ci*100+j)
            rr["chi_sentinel"]=chi; effects.append(rr)
    raw=pd.concat(parts,ignore_index=True); eff=pd.DataFrame(effects)
    raw.to_csv(out/"15_yoked_and_rescue_raw.csv",index=False)
    eff.to_csv(out/"16_yoked_and_rescue_contrasts.csv",index=False)
    return raw, eff


def stage_temporal_window(cp, out, lf):
    log("[Stage 7/10] Environmental temporal-window test", lf)
    rows=[]
    for ei, tau_env in enumerate(cp["tau_env_grid"]):
        base=ClosureConfig(tau_env=float(tau_env), causal_delay=1.0, warmup_time=cp["extra_warmup_time"], record_time=cp["extra_record_time"])
        env=generate_environment_kind(base, cp["dense_reps"], seed=930000+ei, kind="ou")
        for ratio in cp["env_ratio_grid"]:
            delta=float(ratio*tau_env)
            cfg=ClosureConfig(tau_env=float(tau_env), causal_delay=delta, warmup_time=cp["extra_warmup_time"], record_time=cp["extra_record_time"])
            g=simulate_intact_grid(cp["dense_chi_grid"],cfg,env)
            ko=simulate_intact_grid(np.array([1.0]),cfg,env,retention_knockout=True)
            r_opt,v_opt,chi_opt,method=optimum_from_grid_raw(g)
            ko_mean=float(ko["mean_viability"].mean())
            rows.append({"tau_env":tau_env,"causal_delay":delta,"causal_to_env_ratio":ratio,"optimal_retention_time":r_opt,"optimal_chi":chi_opt,"optimal_viability":v_opt,"retention_benefit_at_optimum":v_opt-ko_mean,"retention_to_causal_ratio":r_opt/delta,"method":method})
            log(f"  tau_env={tau_env:g}, delta/tau_env={ratio:g}, benefit={v_opt-ko_mean:.4g}",lf)
    df=pd.DataFrame(rows); df.to_csv(out/"17_environment_temporal_window.csv",index=False)
    return df


def stage_dissipation_scaling(cp, out, lf):
    log("[Stage 8/10] Dissipation-timescale nondimensional scaling", lf)
    rows=[]
    for ti, td in enumerate(cp["tau_diss_grid"]):
        # Scale every dimensional time/rate so only nondimensional ratios change.
        for ratio in cp["diss_delta_ratio_grid"]:
            delta=float(ratio*td)
            cfg=ClosureConfig(
                tau_diss=float(td), causal_delay=delta, tau_env=5.0*td,
                dt=0.01*td, warmup_time=cp["extra_warmup_time"]*td,
                record_time=cp["extra_record_time"]*td,
                repair_rate=0.2/td, decay_rate=0.2/td,
                shuffle_min_lag_time=15.0*td, shuffle_max_lag_time=50.0*td,
            )
            env=generate_environment_kind(cfg,cp["dense_reps"],seed=940000+ti*100+int(round(ratio*10)),kind="ou")
            g=simulate_intact_grid(cp["dense_chi_grid"],cfg,env)
            r_opt,v_opt,chi_opt,method=optimum_from_grid_raw(g)
            rows.append({"tau_diss":td,"causal_delay":delta,"causal_over_diss":ratio,"optimal_retention_time":r_opt,"optimal_retention_over_diss":r_opt/td,"optimal_chi":chi_opt,"optimal_viability":v_opt,"method":method})
            log(f"  tau_diss={td:g}, delta/tau_diss={ratio:g}, optR/tau_diss={r_opt/td:.3g}",lf)
    df=pd.DataFrame(rows)
    # Matched-ratio collapse dispersion.
    disp=[]
    for ratio,g in df.groupby("causal_over_diss"):
        vals=g["optimal_retention_over_diss"].to_numpy(float)
        disp.append({"causal_over_diss":ratio,"mean_opt_retention_over_diss":mean(vals),"sd_across_tau_diss":sd(vals),"cv_across_tau_diss":sd(vals)/mean(vals) if mean(vals)>EPS else np.nan})
    pd.DataFrame(disp).to_csv(out/"19_dissipation_scaling_collapse.csv",index=False)
    df.to_csv(out/"18_dissipation_scaling.csv",index=False)
    return df, pd.DataFrame(disp)


def latin_hypercube(n, d, seed):
    rng=np.random.default_rng(seed)
    u=(np.arange(n)[:,None]+rng.random((n,d)))/n
    for j in range(d): rng.shuffle(u[:,j])
    return u


def stage_parameter_environment_robustness(cp, out, lf):
    log("[Stage 9/10] Parameter and environment-form robustness", lf)
    # Ranges are fixed before outcome inspection.
    names=["action_gain","viability_width","repair_rate","decay_rate","env_sd"]
    lows=np.array([0.5,0.7,0.5,0.5,0.75]); highs=np.array([2.0,1.5,2.0,2.0,1.5])
    lhs=latin_hypercube(int(cp["robust_sets"]),len(names),seed=950001)
    vals=lows+(highs-lows)*lhs
    rows=[]
    for si,v in enumerate(vals):
        pars=dict(zip(names,v))
        for kind in ["ou","telegraph"]:
            optima=[]
            for delta in cp["robust_delta_grid"]:
                cfg=ClosureConfig(causal_delay=float(delta),warmup_time=cp["extra_warmup_time"],record_time=min(cp["extra_record_time"],100.0),**pars)
                env=generate_environment_kind(cfg,cp["robust_reps"],seed=951000+si*100+int(delta*10)+(0 if kind=="ou" else 50000),kind=kind)
                g=simulate_intact_grid(cp["robust_chi_grid"],cfg,env)
                r_opt,_,_,_=optimum_from_grid_raw(g); optima.append(r_opt)
            x=np.asarray(cp["robust_delta_grid"],float); y=np.asarray(optima,float)
            slope=float(np.polyfit(x,y,1)[0]) if len(x)>=2 else np.nan
            # Causal yoke effects at a fixed representative operating point.
            cfg=ClosureConfig(causal_delay=3.0,warmup_time=cp["extra_warmup_time"],record_time=min(cp["extra_record_time"],100.0),**pars)
            env=generate_environment_kind(cfg,cp["robust_reps"],seed=952000+si+(0 if kind=="ou" else 50000),kind=kind)
            intact,tr=simulate_trace_assay(3.0,cfg,env,"intact",return_traces=True)
            partner={k:np.roll(z,1,axis=1) for k,z in tr.items()}
            hy=simulate_trace_assay(3.0,cfg,env,"history_yoked",replay=partner)
            ay=simulate_trace_assay(3.0,cfg,env,"action_yoked",replay=partner)
            cc=simulate_trace_assay(3.0,cfg,env,"closure_cut")
            cy=simulate_trace_assay(3.0,cfg,env,"closure_yoked",replay=partner)
            rec={"set":si,"environment_kind":kind,"slope_opt_retention_vs_delay":slope}
            rec.update(pars)
            rec["history_yoked_effect"]=float((intact["mean_viability"]-hy["mean_viability"]).mean())
            rec["action_yoked_effect"]=float((intact["mean_viability"]-ay["mean_viability"]).mean())
            rec["closure_cut_effect"]=float((intact["mean_viability"]-cc["mean_viability"]).mean())
            rec["closure_rescue_effect"]=float((cy["mean_viability"]-cc["mean_viability"]).mean())
            rows.append(rec)
            log(f"  robustness set={si+1}/{len(vals)} env={kind} slope={slope:.3g}",lf)
    df=pd.DataFrame(rows)
    summary=pd.DataFrame([{
        "n_cases":len(df),
        "fraction_positive_scaling_slope":float(np.mean(df["slope_opt_retention_vs_delay"]>0)),
        "fraction_positive_history_yoked_effect":float(np.mean(df["history_yoked_effect"]>0)),
        "fraction_positive_action_yoked_effect":float(np.mean(df["action_yoked_effect"]>0)),
        "fraction_positive_closure_cut_effect":float(np.mean(df["closure_cut_effect"]>0)),
        "fraction_positive_closure_rescue_effect":float(np.mean(df["closure_rescue_effect"]>0)),
    }])
    df.to_csv(out/"20_parameter_environment_robustness.csv",index=False)
    summary.to_csv(out/"21_parameter_environment_robustness_summary.csv",index=False)
    return df,summary


def stage_numerical_robustness(cp, out, lf):
    log("[Stage 10/10] Numerical and initial-state robustness", lf)
    rows=[]
    tests=[]
    for fac in cp["numeric_dt_factors"]: tests.append(("dt_factor",float(fac)))
    for rt in cp["numeric_record_times"]: tests.append(("record_time",float(rt)))
    for b0 in cp["numeric_b0"]: tests.append(("b_initial",float(b0)))
    for ti,(kind,val) in enumerate(tests):
        dt=0.01*(val if kind=="dt_factor" else 1.0)
        rt=val if kind=="record_time" else cp["extra_record_time"]
        b0=val if kind=="b_initial" else 1.0
        cfg=ClosureConfig(causal_delay=3.0,dt=dt,warmup_time=cp["extra_warmup_time"],record_time=rt)
        env=generate_environment_kind(cfg,cp["dense_reps"],seed=960000+ti,kind="ou")
        g=simulate_intact_grid(cp["dense_chi_grid"],cfg,env,b_initial=b0)
        r_opt,v_opt,chi_opt,method=optimum_from_grid_raw(g)
        rows.append({"test":kind,"value":val,"dt":dt,"record_time":rt,"b_initial":b0,"optimal_retention_time":r_opt,"optimal_chi":chi_opt,"optimal_viability":v_opt,"method":method})
        log(f"  numerical {kind}={val:g}: optR={r_opt:.3g}",lf)
    df=pd.DataFrame(rows); df.to_csv(out/"22_numerical_robustness.csv",index=False)
    return df


def comprehensive_figures(dense_opt, window, diss, robust, numerical, out, lf):
    fd=out/"figures"; fd.mkdir(exist_ok=True)
    plt.figure(figsize=(7,5)); plt.plot(dense_opt["causal_delay"],dense_opt["optimal_retention_time"],marker="o")
    x=dense_opt["causal_delay"].to_numpy(float); sl=float(dense_opt["global_slope"].iloc[0]); ic=float(dense_opt["global_intercept"].iloc[0]); plt.plot(x,ic+sl*x,linestyle="--")
    plt.xlabel("Causal closure time"); plt.ylabel("Optimal retention time"); plt.title("Dense temporal-closure scaling"); savefig(fd/"fig6_dense_temporal_closure_scaling.png")
    plt.figure(figsize=(7,5))
    for te,g in window.groupby("tau_env"):
        g=g.sort_values("causal_to_env_ratio"); plt.plot(g["causal_to_env_ratio"],g["retention_benefit_at_optimum"],marker="o",label=f"tau_E={te:g}")
    plt.axhline(0,linestyle="--"); plt.xlabel("tau_C / tau_E"); plt.ylabel("Retention benefit at optimum"); plt.legend(); plt.title("Environmental temporal window"); savefig(fd/"fig7_environment_temporal_window.png")
    plt.figure(figsize=(7,5))
    for td,g in diss.groupby("tau_diss"):
        g=g.sort_values("causal_over_diss"); plt.plot(g["causal_over_diss"],g["optimal_retention_over_diss"],marker="o",label=f"tau_diss={td:g}")
    plt.xlabel("tau_C / tau_diss"); plt.ylabel("tau_R* / tau_diss"); plt.legend(); plt.title("Nondimensional dissipation scaling"); savefig(fd/"fig8_dissipation_scaling_collapse.png")
    plt.figure(figsize=(7,5)); plt.hist(robust["slope_opt_retention_vs_delay"].dropna().to_numpy(),bins=min(12,max(4,len(robust)//2))); plt.axvline(0,linestyle="--"); plt.xlabel("Scaling slope"); plt.ylabel("Count"); plt.title("Parameter/environment robustness"); savefig(fd/"fig9_robustness_slope_distribution.png")
    log("[Comprehensive figures] saved",lf)


def write_comprehensive_report(out, audit, dense_opt, bsum, yoke_eff, window, diss_collapse, robust_summary, numerical, lf):
    path=out/"23_comprehensive_analysis_report.md"
    with open(path,"w",encoding="utf-8") as f:
        f.write("# What Is Life? Comprehensive Temporal Causal-Closure Validation\n\n")
        f.write("## Numerical regression\n\n"+audit.to_string(index=False)+"\n\n")
        f.write("## Dense causal-delay scaling\n\n"+dense_opt.to_string(index=False)+"\n\n")
        f.write("### Shared-replicate bootstrap\n\n"+pd.DataFrame([bsum]).to_string(index=False)+"\n\n")
        f.write("## Yoked and mediator-rescue causal assays\n\n"+yoke_eff.to_string(index=False)+"\n\n")
        f.write("## Environmental temporal window\n\n"+window.to_string(index=False)+"\n\n")
        f.write("## Dissipation scaling collapse\n\n"+diss_collapse.to_string(index=False)+"\n\n")
        f.write("## Parameter and environment-form robustness\n\n"+robust_summary.to_string(index=False)+"\n\n")
        f.write("## Numerical robustness\n\n"+numerical.to_string(index=False)+"\n\n")
        f.write("## Interpretation criteria\n\n")
        f.write(
            "The temporal causal-closure hypothesis is supported only if: (1) the independent Temporal Membranes regression succeeds; "
            "(2) the bootstrap distribution of the optimal-retention versus causal-delay slope is predominantly positive; "
            "(3) temporally structured but non-focal history and action yokes reduce viability relative to intact dynamics; "
            "(4) cutting retention, reuse, or closure reduces viability and supplying the corresponding downstream mediator rescues it; "
            "(5) useful retention is constrained by the environmental timescale; (6) nondimensional retention-versus-causal-delay relations remain stable when the dissipative timescale is changed; and "
            "(7) the principal signs persist across parameter sets, OU and telegraph environments, numerical timestep, record duration, and initial integrity.\n"
        )
    log(f"[Report] saved {path}",lf)


def load_csv(out,name):
    return pd.read_csv(out/name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["smoke","full"],default="smoke")
    ap.add_argument("--output-dir",default=str(COMPREHENSIVE_OUTPUT))
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--start-stage",type=int,default=0,help="0-10; earlier outputs are loaded if present")
    args=ap.parse_args()
    out=Path(args.output_dir).expanduser().resolve(); ensure_dirs(out)
    lf=out/"run.log"; open(lf,"a",encoding="utf-8").write("\n"+"="*88+f"\nNew comprehensive run {now()} mode={args.mode}\n")
    p=plan(args.mode); cp=comprehensive_plan(args.mode); save_json(out/"run_parameters_core.json",p); save_json(out/"run_parameters_comprehensive.json",cp)
    start=time.time(); log(f"Starting comprehensive mode={args.mode}; output={out}",lf)
    try:
        # Core stages 0-4. Resume uses their existing csv products.
        if args.resume and stage_done(out,["00_regression_audit.csv","02_temporal_membrane_retention_summary.csv"]):
            audit=load_csv(out,"00_regression_audit.csv"); tm_summary=load_csv(out,"02_temporal_membrane_retention_summary.csv")
            log("[Resume] loaded Temporal Membranes regression",lf)
        else:
            _,tm_summary,audit=stage0_and_1_tm(p,out,lf)
        if args.mode=="full" and not bool(audit["pass"].all()):
            raise RuntimeError("Temporal Membranes regression failed")
        if not (args.resume and stage_done(out,["03_primary_causal_closure_raw.csv","04_primary_causal_closure_summary.csv","04B_closure_yoke_audit.csv"])):
            primary_raw,primary_summary,yoke_audit=run_primary_closure(p,tm_summary,out,lf)
            contrasts,window0=run_inference(p,primary_raw,out,lf)
            _,delay_summary,opt0=run_delay_scaling(p,tm_summary,out,lf)
            make_figures(tm_summary,primary_summary,contrasts,delay_summary,opt0,out,lf)
        else:
            log("[Resume] core causal-closure outputs already exist",lf)

        # Stage 5
        if args.resume and stage_done(out,["11_dense_causal_scaling_raw.csv","12_dense_causal_scaling_optima.csv","13_dense_causal_scaling_bootstrap.csv","14_dense_causal_scaling_bootstrap_summary.csv"]):
            dense_raw=load_csv(out,"11_dense_causal_scaling_raw.csv"); dense_opt=load_csv(out,"12_dense_causal_scaling_optima.csv"); bdf=load_csv(out,"13_dense_causal_scaling_bootstrap.csv"); bsum=load_csv(out,"14_dense_causal_scaling_bootstrap_summary.csv").iloc[0].to_dict(); log("[Resume] Stage 5",lf)
        else:
            dense_raw,dense_opt,bdf,bsum=stage_dense_scaling(cp,out,lf)
        # Stage 6
        if args.resume and stage_done(out,["15_yoked_and_rescue_raw.csv","16_yoked_and_rescue_contrasts.csv"]):
            yoke_raw=load_csv(out,"15_yoked_and_rescue_raw.csv"); yoke_eff=load_csv(out,"16_yoked_and_rescue_contrasts.csv"); log("[Resume] Stage 6",lf)
        else:
            yoke_raw,yoke_eff=stage_yoked_and_rescue(cp,out,lf)
        # Stage 7
        if args.resume and stage_done(out,["17_environment_temporal_window.csv"]): window=load_csv(out,"17_environment_temporal_window.csv"); log("[Resume] Stage 7",lf)
        else: window=stage_temporal_window(cp,out,lf)
        # Stage 8
        if args.resume and stage_done(out,["18_dissipation_scaling.csv","19_dissipation_scaling_collapse.csv"]): diss=load_csv(out,"18_dissipation_scaling.csv"); diss_collapse=load_csv(out,"19_dissipation_scaling_collapse.csv"); log("[Resume] Stage 8",lf)
        else: diss,diss_collapse=stage_dissipation_scaling(cp,out,lf)
        # Stage 9
        if args.resume and stage_done(out,["20_parameter_environment_robustness.csv","21_parameter_environment_robustness_summary.csv"]): robust=load_csv(out,"20_parameter_environment_robustness.csv"); robust_summary=load_csv(out,"21_parameter_environment_robustness_summary.csv"); log("[Resume] Stage 9",lf)
        else: robust,robust_summary=stage_parameter_environment_robustness(cp,out,lf)
        # Stage 10
        if args.resume and stage_done(out,["22_numerical_robustness.csv"]): numerical=load_csv(out,"22_numerical_robustness.csv"); log("[Resume] Stage 10",lf)
        else: numerical=stage_numerical_robustness(cp,out,lf)
        comprehensive_figures(dense_opt,window,diss,robust,numerical,out,lf)
        write_comprehensive_report(out,audit,dense_opt,bsum,yoke_eff,window,diss_collapse,robust_summary,numerical,lf)
        log(f"Completed comprehensive validation successfully in {time.time()-start:.1f} s",lf)
    except Exception as e:
        log(f"FATAL: {e}",lf); traceback.print_exc(); sys.exit(1)


if __name__ == "__main__":
    main()
