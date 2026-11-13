#!/usr/bin/env python3
"""Train the weekly flu-positivity model and write a hubverse forecast CSV.

Example:
    python run_forecast.py
    python run_forecast.py --data data/time-series.csv --reference-date 2026-11-09

By default the official hub file target-data/time-series.csv is downloaded
from the China Infectious Disease Forecast Hub. The script runs a rolling-
origin backtest, picks the single best method by mean WIS (preferring a
CQR-calibrated variant when WIS is essentially tied), then writes:

    output/<model_id>/<reference_date>-<model_id>.csv
    metadata/<model_id>.yaml
    results/backtest_metrics.csv
    results/backtest_metrics_by_horizon.csv
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

import flu_model as fm

ROOT = Path(__file__).resolve().parent
TEAM_ABBR = "GZNL"
TEAM_NAME = "GZNL Group"
CONTRIBUTOR = {
    "name": "Yang Kaixin",
    "affiliation": "Guangzhou National Laboratory",
    "email": "yang_kaixin@gzlab.ac.cn",
}

# Short names used in metadata / paths (must be alphanumeric/underscore, <=16).
MODEL_META = {
    "lightgbm_cqr": {
        "model_abbr": "FluLGBMCQR",
        "model_name": "Flu LightGBM Conformal Quantile Regression",
        "ensemble_of_models": False,
        "methods": (
            "Weekly LightGBM quantile regression on lag/rolling/Fourier features, "
            "with sliding-window conformal (CQR) interval calibration."
        ),
        "methods_long": (
            "Adapted from the EpiETL modeling suite (originally a national monthly "
            "flu case-count model) to the hub target wk inc flu prop ili: the CNIC "
            "weekly national influenza surveillance network ILI specimen laboratory "
            "positivity rate, in percentage points on a 0-100 scale, location CN. "
            "At each weekly origin the model sees only observations available after "
            "the hub's typical two-week reporting lag. Features are weekly lags 1-4, "
            "6, 8 and 12, 4- and 8-week rolling means, the first difference, and "
            "yearly Fourier terms for the known target week. For each horizon "
            "relative to the last observed week a separate LightGBM quantile "
            "regressor is trained at each of the 23 hub quantile levels "
            "(direct multi-horizon quantile regression). Prediction intervals are "
            "then calibrated with online sliding-window conformalized quantile "
            "regression (CQR): for every symmetric quantile pair the conformal "
            "score S = max(q_lo - y, y - q_hi) is computed on the previous 26 "
            "origins and the interval is expanded (never shrunk) by the non-negative (1-alpha)(1+1/n) quantile "
            "of S; the median is left unchanged. Quantiles are clipped to [0, 100] "
            "and isotonic-sorted to prevent crossing. Seasonal naive and a "
            "climatology-plus-anomaly-persistence member are used only for "
            "backtest comparison and are not part of the submitted forecast. "
            "Prophet from the original suite is not used; the climatology member "
            "is the robust seasonal/statistical stand-in."
        ),
    },
    "ensemble_cqr": {
        "model_abbr": "EnsembleCQR",
        "model_name": "Conformal Quantile Ensemble",
        "ensemble_of_models": True,
        "methods": (
            "Equal-weight quantile ensemble of LightGBM and climatology-persistence, "
            "with sliding-window conformal (CQR) interval calibration."
        ),
        "methods_long": (
            "Adapted from the EpiETL modeling suite (originally a national monthly "
            "flu case-count model) to the hub target wk inc flu prop ili: the CNIC "
            "weekly national influenza surveillance network ILI specimen laboratory "
            "positivity rate, in percentage points on a 0-100 scale, location CN. "
            "Two members are combined by equal-weight averaging of each of the 23 "
            "hub quantiles (Vincent / linear pool on the quantile scale), then "
            "calibrated with online sliding-window conformalized quantile regression "
            "(CQR). Member 1 is direct multi-horizon LightGBM quantile regression "
            "with weekly lags 1-4, 6, 8 and 12, 4- and 8-week rolling means, the "
            "first difference, and yearly Fourier terms for the known target week. "
            "Member 2 is a week-of-year climatology with decaying anomaly "
            "persistence (half-life of about four weeks) and empirical residual "
            "quantiles; it replaces Prophet from the original suite, which is not "
            "used here. Seasonal naive is retained only as a backtest baseline and "
            "is excluded from the ensemble, as in the original EpiETL protocol, "
            "because its intervals are systematically weaker. CQR uses the previous "
            "26 origins; the median is not shifted. Quantiles are clipped to "
            "[0, 100] and isotonic-sorted. Training uses only the hub "
            "target-data/time-series.csv series for this target, with a two-week "
            "reporting lag at each origin."
        ),
    },
    "lightgbm": {
        "model_abbr": "LGBMQuantile",
        "model_name": "LightGBM Quantile Regression",
        "ensemble_of_models": False,
        "methods": (
            "Weekly LightGBM quantile regression on lag, rolling-mean, first-"
            "difference and yearly Fourier features; 23 hub quantiles, clipped and isotonic."
        ),
        "methods_long": (
            "Adapted from the EpiETL modeling suite to the hub target wk inc flu "
            "prop ili (CNIC weekly national ILI specimen laboratory positivity rate, "
            "percentage points, location CN). Direct multi-horizon LightGBM quantile "
            "regression at 23 hub levels, with weekly lags 1-4, 6, 8 and 12, rolling "
            "means, first difference, and yearly Fourier terms. Quantiles are clipped "
            "to [0, 100] and isotonic-sorted. Trained on hub target-data/time-series.csv."
        ),
    },
    "ensemble": {
        "model_abbr": "EnsembleQ",
        "model_name": "Quantile Ensemble",
        "ensemble_of_models": True,
        "methods": (
            "Equal-weight quantile ensemble of LightGBM quantile regression and "
            "climatology-persistence members."
        ),
        "methods_long": (
            "Adapted from the EpiETL modeling suite to wk inc flu prop ili. Equal-"
            "weight average of LightGBM quantile regression and a climatology-plus-"
            "anomaly-persistence member at each of the 23 hub quantile levels. "
            "Quantiles are clipped to [0, 100] and isotonic-sorted. Trained on hub "
            "target-data/time-series.csv with a two-week reporting lag."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weekly flu positivity hubverse forecast")
    p.add_argument("--data", default=None,
                   help="Path to hub time-series.csv (downloaded if omitted)")
    p.add_argument("--reference-date", default=None,
                   help="Monday reference date YYYY-MM-DD (default: latest Monday)")
    p.add_argument("--origin-start", default="2024-01-01",
                   help="First backtest reference date")
    p.add_argument("--model", default=None,
                   help="Force a method name (e.g. lightgbm_cqr); default is best by WIS")
    p.add_argument("--skip-backtest", action="store_true",
                   help="Reuse results/backtest_predictions.csv if present")
    return p.parse_args()


def write_metadata(path: Path, method: str, reference_date: date, last_obs: date) -> str:
    meta = MODEL_META.get(method)
    if meta is None:
        # generic fallback for unexpected winners
        abbr = "".join(ch for ch in method.title().replace("_", "") if ch.isalnum())[:16]
        meta = {
            "model_abbr": abbr or "FluCQR",
            "model_name": method,
            "ensemble_of_models": "ensemble" in method,
            "methods": (
                "Weekly influenza positivity forecast adapted from the EpiETL suite "
                "to wk inc flu prop ili, with conformal calibration where applicable."
            ),
            "methods_long": (
                "Adapted from the EpiETL modeling suite to the hub target wk inc flu "
                "prop ili (CNIC weekly national ILI specimen laboratory positivity "
                f"rate, percentage points, location CN). Selected method: {method}. "
                "Trained on hub target-data/time-series.csv."
            ),
        }
    # methods must be <= 200 chars
    methods = meta["methods"]
    if len(methods) > 200:
        raise ValueError(f"methods is {len(methods)} chars")
    model_id = f"{TEAM_ABBR}-{meta['model_abbr']}"
    # YAML written by hand so the file matches existing hub style closely.
    data_inputs = (
        "CNIC weekly national influenza surveillance network ILI specimen "
        "laboratory positivity rate, as published in the China National Influenza "
        "Center weekly reports and provided in this hub's target-data/time-series.csv "
        f"(target wk inc flu prop ili, location CN; last observation {last_obs.isoformat()})."
    )
    lines = [
        f'team_name: "{TEAM_NAME}"',
        f'team_abbr: "{TEAM_ABBR}"',
        f'model_name: "{meta["model_name"]}"',
        f'model_abbr: "{meta["model_abbr"]}"',
        "model_contributors:",
        f'  - name: "{CONTRIBUTOR["name"]}"',
        f'    affiliation: "{CONTRIBUTOR["affiliation"]}"',
        f'    email: "{CONTRIBUTOR["email"]}"',
        'license: "CC-BY-4.0"',
        "designated_model: false",
        f'data_inputs: "{data_inputs}"',
        f'methods: "{methods}"',
        f'methods_long: "{meta["methods_long"]}"',
        f"ensemble_of_models: {str(bool(meta['ensemble_of_models'])).lower()}",
        "ensemble_of_hub_models: false",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return model_id


def main() -> None:
    args = parse_args()
    data_path = Path(args.data) if args.data else ROOT / "data" / "time-series.csv"
    if args.data:
        flu = fm.load_flu_series(args.data)
    elif data_path.exists():
        # still refresh from the hub unless the user passed --data explicitly
        try:
            flu = fm.load_flu_series(fm.HUB_TIME_SERIES_URL)
            data_path.parent.mkdir(parents=True, exist_ok=True)
            # save a snapshot of the full official file, not just the flu subset
            import urllib.request
            data_path.write_bytes(urllib.request.urlopen(fm.HUB_TIME_SERIES_URL, timeout=60).read())
        except Exception as exc:
            print(f"download failed ({exc}); using local {data_path}")
            flu = fm.load_flu_series(str(data_path))
    else:
        flu = fm.load_flu_series(fm.HUB_TIME_SERIES_URL)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        data_path.write_bytes(urllib.request.urlopen(fm.HUB_TIME_SERIES_URL, timeout=60).read())

    last_obs = pd.Timestamp(flu["target_end_date"].max()).date()
    if args.reference_date:
        reference_date = date.fromisoformat(args.reference_date)
        if reference_date.weekday() != 0:
            raise SystemExit(f"reference_date {reference_date} is not a Monday")
    else:
        reference_date = fm.default_reference_date(last_obs)
    lag = (reference_date - last_obs).days // 7
    print(f"flu series: {flu['target_end_date'].min().date()} -> {last_obs}, n={len(flu)}")
    print(f"reference_date={reference_date}  reporting_lag_weeks={lag}")

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    pred_path = results_dir / "backtest_predictions.csv"

    if args.skip_backtest and pred_path.exists():
        pred = pd.read_csv(pred_path)
        bt_metrics = pd.read_csv(results_dir / "backtest_metrics_by_horizon.csv")
        # rebuild a Backtest-like average from the fair metrics file
        avg = fm.summarize_metrics(bt_metrics)
        print("reused cached backtest")
    else:
        print(f"running rolling-origin backtest from {args.origin_start} ...")
        bt = fm.run_backtest(
            flu,
            origin_start=args.origin_start,
            reporting_lag_weeks=max(lag, 1),
        )
        bt.predictions.to_csv(pred_path, index=False, encoding="utf-8")
        bt.metrics_fair.to_csv(results_dir / "backtest_metrics_by_horizon.csv",
                               index=False, encoding="utf-8")
        avg = fm.summarize_metrics(bt.metrics_fair)
        avg.to_csv(results_dir / "backtest_metrics.csv", index=False, encoding="utf-8")
        pred = bt.predictions
        print("\nBacktest mean over horizons -1..6 (fair CQR window):")
        print(avg.round(3).to_string(index=False))

    method = args.model or fm.pick_best_model(avg)
    print(f"\nselected method: {method}")

    fc = fm.final_forecast(flu, reference_date, pred, method)
    issues = fm.validate_forecast_frame(fc, reference_date)
    if issues:
        raise SystemExit("forecast failed validation:\n- " + "\n- ".join(issues))

    meta_path = ROOT / "metadata" / "_tmp.yaml"
    model_id = write_metadata(meta_path, method, reference_date, last_obs)
    final_meta = ROOT / "metadata" / f"{model_id}.yaml"
    meta_path.replace(final_meta)
    meta_path = final_meta

    out_dir = ROOT / "output" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{reference_date.isoformat()}-{model_id}.csv"
    fm.write_hubverse_csv(fc, str(out_path))

    # median trajectory vs last observation
    mid = fc[fc["output_type_id"] == 0.5][
        ["horizon", "target_end_date", "value"]
    ].sort_values("horizon")
    print(f"\nlast observed positivity on {last_obs}: {float(flu['y'].iloc[-1]):.1f}")
    print("median forecast:")
    print(mid.to_string(index=False))
    print(f"\nwrote {out_path}  ({len(fc)} rows)")
    print(f"wrote {meta_path}")

    summary = {
        "model_id": model_id,
        "method": method,
        "reference_date": reference_date.isoformat(),
        "last_obs": last_obs.isoformat(),
        "last_obs_value": float(flu["y"].iloc[-1]),
        "reporting_lag_weeks": lag,
        "n_rows": int(len(fc)),
        "median_forecast": mid.to_dict(orient="records"),
        "backtest_average": avg.round(4).to_dict(orient="records"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    (results_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
