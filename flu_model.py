"""Weekly national influenza positivity forecast models.

Adapted from the EpiETL modeling suite (prepare_data.py, backtest.py,
forecast_final.py, ensemble_conformal.py), which originally targeted
national *monthly flu case counts*. This module retargets the same
methodological family — lag / rolling / seasonal features, seasonal
naive, a seasonal-climatology member, LightGBM quantile regression,
quantile ensemble, and conformalized quantile regression (CQR) — to
the China Infectious Disease Forecast Hub target:

    wk inc flu prop ili
    CNIC weekly national ILI specimen laboratory positivity rate
    unit: percentage points in [0, 100]
    location: CN
    horizons: recommended set -1 through 6 weeks
    23 hubverse quantile levels

Do not treat this as the original monthly case-count model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable
from urllib.request import urlopen

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HUB_TIME_SERIES_URL = (
    "https://raw.githubusercontent.com/dailypartita/"
    "China-Infectious-Disease-Forecast-Hub/main/target-data/time-series.csv"
)
TARGET = "wk inc flu prop ili"
LOCATION = "CN"
QUANTILES = [
    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35,
    0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,
    0.85, 0.9, 0.95, 0.975, 0.99,
]
SUBMIT_HORIZONS = list(range(-1, 7))  # -1 .. 6
FEATURE_COLS = [
    "lag1", "lag2", "lag3", "lag4", "lag6", "lag8", "lag12",
    "r4w", "r8w", "d1",
    "cal_sin1", "cal_cos1", "cal_sin2", "cal_cos2",
]
# Symmetric PI pairs for CQR: (tau_lo, tau_hi) with alpha = 2 * tau_lo.
CQR_PAIRS = [(q, 1.0 - q) for q in QUANTILES if q < 0.5]
LGBM_PARAMS = dict(
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=10,
    subsample=0.9,
    colsample_bytree=0.9,
    verbose=-1,
    n_jobs=1,
)
LGBM_WORKERS = 8
CALIB_W = 26          # sliding CQR window in origins (weeks)
CQR_BURN_IN = 16      # minimum calibration points before CQR output
MIN_TRAIN = 80        # minimum LightGBM training rows


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_flu_series(path_or_url: str | None = None) -> pd.DataFrame:
    """Load hub time-series and return the national weekly flu positivity series.

    Columns: target_end_date (Monday, datetime64), y (percentage points).
    Sorted ascending, unique weekly index. Values are already on the 0-100
    percentage-point scale used by the hub (not a 0-1 proportion).
    """
    src = path_or_url or HUB_TIME_SERIES_URL
    if src.startswith("http://") or src.startswith("https://"):
        with urlopen(src, timeout=60) as resp:
            df = pd.read_csv(resp)
    else:
        df = pd.read_csv(src)
    flu = df[df["target"] == TARGET].copy()
    if "location" in flu.columns:
        flu = flu[flu["location"] == LOCATION]
    flu["target_end_date"] = pd.to_datetime(flu["target_end_date"])
    flu = flu.rename(columns={"observation": "y"})
    flu = flu[["target_end_date", "y"]].dropna()
    flu = flu.sort_values("target_end_date").drop_duplicates("target_end_date")
    flu = flu.reset_index(drop=True)
    if flu.empty:
        raise ValueError(f"No rows for target={TARGET} in {src}")
    return flu


def monday_on_or_before(d: date) -> date:
    return d - timedelta(days=d.weekday())


def default_reference_date(last_obs: date, today: date | None = None) -> date:
    """Latest Monday that is still a plausible open weekly round.

    Uses the most recent Monday on or before *today*. The two-week
    reporting lag between last_obs and that Monday is expected.
    """
    today = today or date.today()
    return monday_on_or_before(today)


# ---------------------------------------------------------------------------
# Features (weekly analogue of the EpiETL lag / rolling / Fourier block)
# ---------------------------------------------------------------------------

def _fourier(week_of_year) -> dict[str, np.ndarray]:
    w = np.asarray(week_of_year, dtype=float)
    out = {
        "cal_sin1": np.sin(2 * np.pi * w / 52.0),
        "cal_cos1": np.cos(2 * np.pi * w / 52.0),
        "cal_sin2": np.sin(4 * np.pi * w / 52.0),
        "cal_cos2": np.cos(4 * np.pi * w / 52.0),
    }
    # Keep scalars as 0-d arrays so float() works for a single week.
    if out["cal_sin1"].ndim == 0 or out["cal_sin1"].shape == ():
        return out
    if out["cal_sin1"].size == 1:
        return {k: np.asarray(v).reshape(()) for k, v in out.items()}
    return out


def iso_week(ts) -> int:
    if hasattr(ts, "isocalendar"):
        return int(ts.isocalendar().week)
    return int(pd.Timestamp(ts).isocalendar().week)


def features_from_history(y: np.ndarray, dates: np.ndarray, origin: int,
                          target_date) -> dict[str, float] | None:
    """Build features at last-observed index `origin` for a given target date.

    lag1 is the most recent observation y[origin] (weekly analogue of a
    standard lag from the last observed week, slightly more natural than
    the original monthly code which shifted lags by one extra month).
    """
    if origin < 11:
        return None
    hist = y[: origin + 1]
    if np.isnan(hist[-1]):
        return None

    def lag(k: int) -> float:
        v = hist[-k] if len(hist) >= k else np.nan
        return float(v)

    row = {
        "lag1": lag(1),
        "lag2": lag(2),
        "lag3": lag(3),
        "lag4": lag(4),
        "lag6": lag(6),
        "lag8": lag(8),
        "lag12": lag(12),
        "r4w": float(np.nanmean(hist[-4:])),
        "r8w": float(np.nanmean(hist[-8:])),
        "d1": lag(1) - lag(2),
    }
    row.update({k: float(np.asarray(v).reshape(-1)[0]) for k, v in _fourier(iso_week(target_date)).items()})
    if any(not np.isfinite(row[c]) for c in FEATURE_COLS):
        return None
    return row


# ---------------------------------------------------------------------------
# Quantile helpers
# ---------------------------------------------------------------------------

def enforce_quantiles(qs: Iterable[float]) -> np.ndarray:
    """Clip to [0, 100] and restore monotonicity."""
    x = np.clip(np.asarray(list(qs), dtype=float), 0.0, 100.0)
    x = np.maximum.accumulate(x)
    x = np.minimum.accumulate(x[::-1])[::-1]
    return np.clip(x, 0.0, 100.0)


def empirical_error_quantiles(errors: np.ndarray, point: float) -> np.ndarray:
    """Add historical residual quantiles to a point forecast."""
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    if len(errors) < 8:
        # fallback: wide but finite residual scale from the errors we do have
        sigma = float(np.nanstd(errors)) if len(errors) else 5.0
        sigma = max(sigma, 1.0)
        from scipy.stats import norm
        qs = point + sigma * norm.ppf(QUANTILES)
        return enforce_quantiles(qs)
    add = np.quantile(errors, QUANTILES, method="linear")
    return enforce_quantiles(point + add)


# ---------------------------------------------------------------------------
# Member models
# ---------------------------------------------------------------------------

def predict_seasonal_naive(y: np.ndarray, dates: np.ndarray, origin: int,
                           target_date, k_steps: int) -> np.ndarray:
    """Same-week-last-year naive, with empirical same-horizon residual quantiles.

    Intervals are built on the *raw* percentage-point scale (the original
    monthly model used log-normal intervals, which were too wide for counts
    and would be inappropriate for a 0-100 positivity rate).
    """
    dates = pd.to_datetime(dates)
    target_date = pd.Timestamp(target_date)
    seasonal_lag = pd.Timestamp(target_date) - pd.Timedelta(weeks=52)
    # point: value 52 weeks before the target, if observed; else 52-week lag of last obs
    point = None
    match = np.where(dates[: origin + 1] == seasonal_lag)[0]
    if len(match):
        point = float(y[match[0]])
    elif origin >= 52:
        point = float(y[origin - 51])  # approx 52 weeks before last obs, shifted
    if point is None or not np.isfinite(point):
        point = float(y[origin])

    # residuals of 52-week seasonal naive at this step length, using only
    # history through origin (no look-ahead)
    errors = []
    for t in range(52 + k_steps, origin + 1):
        pred_idx = t - k_steps
        seas_idx = t - 52
        if seas_idx >= 0 and np.isfinite(y[t]) and np.isfinite(y[seas_idx]):
            errors.append(y[t] - y[seas_idx])
    return empirical_error_quantiles(np.asarray(errors, dtype=float), point)


def _climatology_map(y: np.ndarray, dates: np.ndarray, origin: int,
                     min_years: int = 5) -> dict[int, float]:
    dates = pd.to_datetime(dates[: origin + 1])
    weeks = np.array([iso_week(d) for d in dates])
    vals = y[: origin + 1]
    clim = {}
    for w in range(1, 54):
        hist = vals[weeks == w]
        hist = hist[np.isfinite(hist)]
        # drop the most recent point of this week-of-year if we want a stable clim;
        # keep all history through origin — it is known.
        if len(hist) >= max(3, min_years // 2):
            clim[w] = float(np.median(hist))
    if not clim:
        clim = {w: float(np.nanmedian(vals)) for w in range(1, 54)}
    return clim


def predict_clim_persist(y: np.ndarray, dates: np.ndarray, origin: int,
                         target_date, k_steps: int) -> np.ndarray:
    """Week-of-year climatology plus decaying anomaly persistence.

    This is the robust seasonal/statistical member that stands in for
    Prophet (Prophet is not required at runtime).
    """
    dates = pd.to_datetime(dates)
    clim = _climatology_map(y, dates, origin)
    w_orig = iso_week(dates[origin])
    w_tgt = iso_week(target_date)
    c_orig = clim.get(w_orig, float(y[origin]))
    c_tgt = clim.get(w_tgt, c_orig)
    anomaly = float(y[origin]) - c_orig
    # weekly persistence; half-life ~ 4 weeks
    rho = 0.5 ** (1.0 / 4.0)
    point = c_tgt + (rho ** k_steps) * anomaly

    errors = []
    # walk through history and collect same-step errors of this predictor
    for t in range(80, origin + 1):
        tgt = t  # we will pretend last obs is t - k_steps
        orig_t = t - k_steps
        if orig_t < 80:
            continue
        # Use climatology computed at the forecast origin (all of it is
        # already known by then). Evaluating only targets <= origin avoids
        # look-ahead in the residual itself.
        wo = iso_week(dates[orig_t])
        wt = iso_week(dates[t])
        co = clim.get(wo, float(y[orig_t]))
        ct = clim.get(wt, co)
        p = ct + (rho ** k_steps) * (float(y[orig_t]) - co)
        if np.isfinite(y[t]) and np.isfinite(p):
            errors.append(float(y[t]) - p)
    return empirical_error_quantiles(np.asarray(errors, dtype=float), point)


def _design_matrix(y: np.ndarray, dates: np.ndarray, last_inclusive: int,
                   k_steps: int) -> tuple[pd.DataFrame, np.ndarray] | tuple[None, None]:
    """Training design matrix for a direct k-step LightGBM quantile model.

    Rows use features at origin t and target y[t + k_steps], with
    t + k_steps <= last_inclusive. No values after last_inclusive are used.
    """
    dates = pd.to_datetime(dates)
    rows = []
    targets = []
    max_t = last_inclusive - k_steps
    for t in range(11, max_t + 1):
        tgt_idx = t + k_steps
        if not np.isfinite(y[tgt_idx]) or not np.isfinite(y[t]):
            continue
        feat = features_from_history(y, dates, t, dates[tgt_idx])
        if feat is None:
            continue
        rows.append(feat)
        targets.append(float(y[tgt_idx]))
    if len(rows) < MIN_TRAIN:
        return None, None
    return pd.DataFrame(rows)[FEATURE_COLS], np.asarray(targets, dtype=float)


def _fit_one_quantile(args):
    q, X, yv, x_pred = args
    model = lgb.LGBMRegressor(objective="quantile", alpha=q, **LGBM_PARAMS)
    model.fit(X, yv)
    return float(model.predict(x_pred)[0])


def predict_lightgbm(y: np.ndarray, dates: np.ndarray, origin: int,
                     target_date, k_steps: int) -> np.ndarray:
    """Direct k-step LightGBM quantile regression (one model per quantile)."""
    X, yv = _design_matrix(y, dates, origin, k_steps)
    feat = features_from_history(y, dates, origin, target_date)
    if X is None or feat is None:
        # not enough history — fall back to climatology
        return predict_clim_persist(y, dates, origin, target_date, k_steps)
    x_pred = pd.DataFrame([feat])[FEATURE_COLS]
    payload = [(q, X, yv, x_pred) for q in QUANTILES]
    with ThreadPoolExecutor(max_workers=LGBM_WORKERS) as pool:
        qs = list(pool.map(_fit_one_quantile, payload))
    return enforce_quantiles(qs)


MEMBER_FUNCS = {
    "seasonal_naive": predict_seasonal_naive,
    "clim_persist": predict_clim_persist,
    "lightgbm": predict_lightgbm,
}
ENSEMBLE_MEMBERS = ["lightgbm", "clim_persist"]


def predict_ensemble(member_qs: dict[str, np.ndarray],
                     members: list[str] | None = None) -> np.ndarray:
    members = members or ENSEMBLE_MEMBERS
    stacked = np.vstack([member_qs[m] for m in members if m in member_qs])
    return enforce_quantiles(stacked.mean(axis=0))


# ---------------------------------------------------------------------------
# Conformalized quantile regression (online sliding-window CQR)
# ---------------------------------------------------------------------------

def cqr_adjust_one(row_qs: np.ndarray, calib_qs: np.ndarray, calib_y: np.ndarray) -> np.ndarray:
    """Adjust a single 23-quantile vector using CQR scores from a calibration set.

    For each symmetric pair (q_tau, q_{1-tau}), the conformity score is
    S = max(q_lo - y, y - q_hi) and the interval is expanded by the
    (1-alpha)(1+1/n) quantile of S, with alpha = 2*tau. The median is
    left unchanged. Same idea as ensemble_conformal.py, extended from
    two intervals (50%/90%) to all hub interval levels.
    """
    if len(calib_y) < 5:
        return enforce_quantiles(row_qs)
    out = row_qs.copy()
    idx = {q: i for i, q in enumerate(QUANTILES)}
    n = len(calib_y)
    for lo_q, hi_q in CQR_PAIRS:
        i, j = idx[lo_q], idx[hi_q]
        s = np.maximum(calib_qs[:, i] - calib_y, calib_y - calib_qs[:, j])
        alpha = 2.0 * lo_q
        k = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
        # Only expand (original EpiETL write-up: 外扩区间). A negative Q
        # would shrink intervals and, on this series, hurt tail coverage.
        Q = max(0.0, float(np.quantile(s, k, method="higher")))
        out[i] = out[i] - Q
        out[j] = out[j] + Q
    return enforce_quantiles(out)


# ---------------------------------------------------------------------------
# Rolling-origin backtest + final forecast
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    metrics_fair: pd.DataFrame  # origins where CQR is available


def _wis(y: float, qs: np.ndarray) -> float:
    """Mean of quantile scores (Bracher et al. style; median term equals AE)."""
    y = float(y)
    scores = []
    for tau, qhat in zip(QUANTILES, qs):
        err = y - qhat
        scores.append(2.0 * ((tau - (err < 0)) * err))  # 2 * pinball
        # pinball = (tau - 1{err<0}) * err; times 2 so tau=0.5 gives |err|
    return float(np.mean(scores))


def coverage_flags(y: float, qs: np.ndarray) -> dict[str, bool]:
    idx = {q: i for i, q in enumerate(QUANTILES)}
    return {
        "cover50": qs[idx[0.25]] <= y <= qs[idx[0.75]],
        "cover80": qs[idx[0.1]] <= y <= qs[idx[0.9]],
        "cover95": qs[idx[0.025]] <= y <= qs[idx[0.975]],
    }


def run_backtest(
    flu: pd.DataFrame,
    origin_start: str = "2024-01-01",
    origin_end: str | None = None,
    reporting_lag_weeks: int = 2,
    members: list[str] | None = None,
) -> BacktestResult:
    """Weekly rolling-origin backtest.

    At each reference Monday R, only observations with
    target_end_date <= R - reporting_lag_weeks * 7 days are available
    (default lag of 2 weeks matches the current hub time-series delay).
    Forecasts are produced for horizons -1..6 relative to R.
    """
    members = members or list(MEMBER_FUNCS)
    flu = flu.sort_values("target_end_date").reset_index(drop=True)
    y = flu["y"].to_numpy(dtype=float)
    dates = flu["target_end_date"].to_numpy()
    date_index = {pd.Timestamp(d).date(): i for i, d in enumerate(pd.to_datetime(dates))}

    last_date = pd.Timestamp(flu["target_end_date"].max()).date()
    start = pd.Timestamp(origin_start).date()
    # last origin that can be evaluated at horizon 6
    max_end = last_date - timedelta(weeks=6)
    end = pd.Timestamp(origin_end).date() if origin_end else max_end
    end = min(end, max_end)

    origins: list[date] = []
    r = monday_on_or_before(start)
    if r < start:
        r = r + timedelta(weeks=1)
    while r <= end:
        origins.append(r)
        r = r + timedelta(weeks=1)

    records = []
    print(f"backtest origins: {origins[0]} -> {origins[-1]}  n={len(origins)}  lag={reporting_lag_weeks}w")
    for i_orig, R in enumerate(origins):
        if i_orig % 10 == 0 or i_orig == len(origins) - 1:
            print(f"  origin {i_orig+1}/{len(origins)}  {R}", flush=True)
        last_obs_day = R - timedelta(weeks=reporting_lag_weeks)
        origin = date_index.get(last_obs_day)
        if origin is None:
            # if that Monday is a missing week, walk back to the latest available
            earlier = [d for d in date_index if d <= last_obs_day]
            if not earlier:
                continue
            origin = date_index[max(earlier)]
            last_obs_day = max(earlier)
        k0 = (R - last_obs_day).days // 7  # steps from last obs to horizon 0
        member_cache: dict[int, dict[str, np.ndarray]] = {}
        for h in SUBMIT_HORIZONS:
            k_steps = k0 + h
            if k_steps <= 0:
                continue
            target_day = R + timedelta(weeks=h)
            tgt_idx = date_index.get(target_day)
            if tgt_idx is None or tgt_idx >= len(y):
                continue
            actual = float(y[tgt_idx])
            if not np.isfinite(actual):
                continue
            qs_map: dict[str, np.ndarray] = {}
            for name in members:
                qs_map[name] = MEMBER_FUNCS[name](y, dates, origin, target_day, k_steps)
            qs_map["ensemble"] = predict_ensemble(qs_map)
            member_cache[h] = qs_map
            for name, qs in qs_map.items():
                rec = {
                    "reference_date": R.isoformat(),
                    "horizon": h,
                    "target_end_date": target_day.isoformat(),
                    "model": name,
                    "actual": actual,
                    "mae": abs(qs[QUANTILES.index(0.5)] - actual),
                    "wis": _wis(actual, qs),
                    **coverage_flags(actual, qs),
                }
                for q, v in zip(QUANTILES, qs):
                    rec[f"q_{q}"] = float(v)
                records.append(rec)

    pred = pd.DataFrame.from_records(records)
    if pred.empty:
        raise RuntimeError("Backtest produced no rows; check dates and lag.")

    # CQR variants for lightgbm / clim_persist / ensemble
    cqr_rows = []
    qcols = [f"q_{q}" for q in QUANTILES]
    for (model, h), g in pred[pred["model"].isin(ENSEMBLE_MEMBERS + ["ensemble"])].groupby(
            ["model", "horizon"], sort=False):
        g = g.sort_values("reference_date").reset_index(drop=True)
        calib_mat = g[qcols].to_numpy()
        actuals = g["actual"].to_numpy()
        for j in range(CQR_BURN_IN, len(g)):
            sl = slice(max(0, j - CALIB_W), j)
            adj = cqr_adjust_one(calib_mat[j], calib_mat[sl], actuals[sl])
            row = g.iloc[j].copy()
            row["model"] = f"{model}_cqr"
            for q, v in zip(QUANTILES, adj):
                row[f"q_{q}"] = float(v)
            row["mae"] = abs(adj[QUANTILES.index(0.5)] - row["actual"])
            row["wis"] = _wis(row["actual"], adj)
            row.update(coverage_flags(row["actual"], adj))
            cqr_rows.append(row)
    cqr = pd.DataFrame(cqr_rows)
    all_pred = pd.concat([pred, cqr], ignore_index=True)

    def _metric_table(df: pd.DataFrame) -> pd.DataFrame:
        return (
            df.groupby(["model", "horizon"], as_index=False)
            .agg(
                MAE=("mae", "mean"),
                WIS=("wis", "mean"),
                cover50=("cover50", "mean"),
                cover80=("cover80", "mean"),
                cover95=("cover95", "mean"),
                n=("mae", "size"),
            )
        )

    metrics = _metric_table(all_pred)
    if not cqr.empty:
        keys = set(zip(cqr["horizon"], cqr["reference_date"]))
        mask = [(h, r) in keys for h, r in zip(all_pred["horizon"], all_pred["reference_date"])]
        metrics_fair = _metric_table(all_pred[pd.Series(mask, index=all_pred.index)])
    else:
        metrics_fair = metrics.copy()

    return BacktestResult(predictions=all_pred, metrics=metrics, metrics_fair=metrics_fair)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Average metrics across horizons -1..6 for model selection."""
    return (
        metrics.groupby("model", as_index=False)
        .agg(
            MAE=("MAE", "mean"),
            WIS=("WIS", "mean"),
            cover50=("cover50", "mean"),
            cover80=("cover80", "mean"),
            cover95=("cover95", "mean"),
            n=("n", "sum"),
        )
        .sort_values("WIS")
        .reset_index(drop=True)
    )


def pick_best_model(avg: pd.DataFrame, candidates: list[str] | None = None) -> str:
    """Select the single best registered method by mean WIS."""
    candidates = candidates or [
        "lightgbm_cqr", "ensemble_cqr", "lightgbm", "ensemble",
        "clim_persist_cqr", "clim_persist", "seasonal_naive",
    ]
    sub = avg[avg["model"].isin(candidates)].copy()
    if sub.empty:
        return str(avg.iloc[0]["model"])
    sub = sub.sort_values(["WIS", "MAE"]).reset_index(drop=True)
    return str(sub.iloc[0]["model"])


def final_forecast(
    flu: pd.DataFrame,
    reference_date: date,
    backtest_pred: pd.DataFrame,
    model: str,
    reporting_lag_weeks: int | None = None,
) -> pd.DataFrame:
    """Fit on all data through last observation and emit a hubverse forecast frame."""
    flu = flu.sort_values("target_end_date").reset_index(drop=True)
    y = flu["y"].to_numpy(dtype=float)
    dates = flu["target_end_date"].to_numpy()
    last_obs_day = pd.Timestamp(flu["target_end_date"].max()).date()
    origin = len(flu) - 1
    if reporting_lag_weeks is None:
        reporting_lag_weeks = (reference_date - last_obs_day).days // 7
    k0 = reporting_lag_weeks

    # raw member forecasts
    rows = []
    for h in SUBMIT_HORIZONS:
        k_steps = k0 + h
        if k_steps <= 0:
            # last obs is at or after this horizon's target — should not happen
            # with a positive reporting lag; skip rather than leak.
            continue
        target_day = reference_date + timedelta(weeks=h)
        qs_map = {}
        for name in MEMBER_FUNCS:
            qs_map[name] = MEMBER_FUNCS[name](y, dates, origin, target_day, k_steps)
        qs_map["ensemble"] = predict_ensemble(qs_map)
        # CQR using backtest scores for the same horizon / base model
        for base in ENSEMBLE_MEMBERS + ["ensemble"]:
            calib = backtest_pred[
                (backtest_pred["model"] == base) & (backtest_pred["horizon"] == h)
            ].sort_values("reference_date").tail(CALIB_W)
            qcols = [f"q_{q}" for q in QUANTILES]
            if len(calib) >= 8:
                qs_map[f"{base}_cqr"] = cqr_adjust_one(
                    qs_map[base],
                    calib[qcols].to_numpy(),
                    calib["actual"].to_numpy(),
                )
        if model not in qs_map:
            raise KeyError(f"Requested model {model} was not produced for horizon {h}")
        qs = qs_map[model]
        for q, v in zip(QUANTILES, qs):
            rows.append({
                "reference_date": reference_date.isoformat(),
                "target": TARGET,
                "horizon": int(h),
                "target_end_date": target_day.isoformat(),
                "location": LOCATION,
                "output_type": "quantile",
                "output_type_id": q,
                "value": round(float(v), 2),
            })
    out = pd.DataFrame(rows)
    # re-enforce monotonicity after rounding, per horizon
    fixed = []
    for h, g in out.groupby("horizon", sort=False):
        g = g.sort_values("output_type_id").copy()
        g["value"] = enforce_quantiles(g["value"].to_numpy())
        g["value"] = np.round(g["value"].to_numpy(), 2)
        # rounding can nick monotonicity by 0.01; repair without changing much
        vals = g["value"].to_numpy()
        for i in range(1, len(vals)):
            if vals[i] < vals[i - 1]:
                vals[i] = vals[i - 1]
        g["value"] = vals
        fixed.append(g)
    out = pd.concat(fixed, ignore_index=True)
    out = out.sort_values(["horizon", "output_type_id"]).reset_index(drop=True)
    # output_type_id should be exact quantile labels, not noisy floats
    out["output_type_id"] = out["output_type_id"].map(
        lambda x: QUANTILES[min(range(len(QUANTILES)), key=lambda i: abs(QUANTILES[i] - float(x)))]
    )
    return out


def write_hubverse_csv(df: pd.DataFrame, path: str) -> None:
    cols = [
        "reference_date", "target", "horizon", "target_end_date",
        "location", "output_type", "output_type_id", "value",
    ]
    out = df[cols].copy()
    out.to_csv(path, index=False, encoding="utf-8")


def validate_forecast_frame(df: pd.DataFrame, reference_date: date) -> list[str]:
    """Schema / format sanity checks (not a substitute for hubValidations)."""
    issues = []
    cols = [
        "reference_date", "target", "horizon", "target_end_date",
        "location", "output_type", "output_type_id", "value",
    ]
    if list(df.columns) != cols and set(df.columns) != set(cols):
        issues.append(f"unexpected columns: {list(df.columns)}")
    if df[cols].shape[1] != 8:
        issues.append("need exactly 8 columns")
    if df.empty:
        issues.append("no rows")
        return issues
    if (df["reference_date"] != reference_date.isoformat()).any():
        issues.append("reference_date mismatch")
    if (df["target"] != TARGET).any():
        issues.append("unexpected target")
    if (df["location"] != LOCATION).any():
        issues.append("unexpected location")
    if (df["output_type"] != "quantile").any():
        issues.append("output_type must be quantile")
    if sorted(df["horizon"].unique()) != SUBMIT_HORIZONS:
        issues.append(f"horizons {sorted(df['horizon'].unique())} != {SUBMIT_HORIZONS}")
    for h, g in df.groupby("horizon"):
        qids = [float(x) for x in g["output_type_id"]]
        if sorted(qids) != QUANTILES:
            issues.append(f"horizon {h} quantile set mismatch: {sorted(qids)}")
        vals = g.sort_values("output_type_id")["value"].to_numpy(dtype=float)
        if (vals < 0).any() or (vals > 100).any():
            issues.append(f"horizon {h} value outside [0, 100]")
        if np.any(np.diff(vals) < -1e-9):
            issues.append(f"horizon {h} quantiles not monotone")
        ted = pd.Timestamp(g["target_end_date"].iloc[0]).date()
        expect = reference_date + timedelta(weeks=int(h))
        if ted != expect:
            issues.append(f"horizon {h} target_end_date {ted} != {expect}")
    expect_n = len(SUBMIT_HORIZONS) * len(QUANTILES)
    if len(df) != expect_n:
        issues.append(f"row count {len(df)} != {expect_n}")
    return issues
