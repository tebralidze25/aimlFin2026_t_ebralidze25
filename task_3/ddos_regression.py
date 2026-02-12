"""
DDoS time-window detection from web-server access logs using regression analysis.

Author: t_ebralidze25
Python: 3.10+
Dependencies: pandas, numpy, scikit-learn, matplotlib

Run:
    python ddos_regression.py --log t_ebralidze25_36714_server.log

This script:
1) parses timestamps
2) aggregates requests per 10 seconds
3) fits a robust polynomial regression baseline
4) flags anomalies using residual threshold (median + 6*MAD)
5) outputs detected DDoS time intervals and saves plots
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline


TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+\d{2}:\d{2})\]")


def parse_timestamps(log_path: Path) -> pd.Series:
    ts = []
    with log_path.open("r", errors="ignore") as f:
        for line in f:
            m = TS_RE.search(line)
            if m:
                ts.append(m.group(1))
    if not ts:
        raise ValueError("No timestamps found. Check log format.")
    return pd.to_datetime(pd.Series(ts), format="%Y-%m-%d %H:%M:%S%z")


def aggregate_requests(dt: pd.Series, bucket: str = "10s") -> pd.DataFrame:
    df = pd.DataFrame({"ts": dt})
    df["bucket"] = df["ts"].dt.floor(bucket)
    return df.groupby("bucket").size().rename("req").to_frame()


def robust_regression_baseline(counts: pd.DataFrame, degree: int = 3) -> tuple[np.ndarray, np.ndarray]:
    # Time in hours since start keeps numbers well-scaled
    t = ((counts.index - counts.index.min()).total_seconds().values.reshape(-1, 1)) / 3600.0
    y = counts["req"].values
    model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), HuberRegressor())
    model.fit(t, y)
    y_pred = model.predict(t)
    residuals = y - y_pred
    return y_pred, residuals


def detect_segments(index: pd.DatetimeIndex, mask: np.ndarray, bucket_seconds: int = 10, gap_seconds: int = 20):
    anom = index[mask]
    if len(anom) == 0:
        return []
    anom = anom.sort_values()
    segments = []
    start = anom[0]
    prev = anom[0]
    for cur in anom[1:]:
        if (cur - prev).total_seconds() > gap_seconds:
            segments.append((start, prev + pd.Timedelta(seconds=bucket_seconds)))
            start = cur
        prev = cur
    segments.append((start, prev + pd.Timedelta(seconds=bucket_seconds)))
    return segments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to server log file")
    ap.add_argument("--outdir", default="figures", help="Directory to store plots")
    args = ap.parse_args()

    log_path = Path(args.log)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    dt = parse_timestamps(log_path)
    counts = aggregate_requests(dt, "10s")

    y_pred, residuals = robust_regression_baseline(counts, degree=3)

    # Robust threshold using MAD (median absolute deviation)
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    threshold = med + 6 * mad  # conservative
    mask = residuals > threshold

    segments = detect_segments(counts.index, mask, bucket_seconds=10, gap_seconds=20)

    print("Detected DDoS interval(s):")
    for s, e in segments:
        print(f" - {s}  ->  {e}")

    # Plot: traffic + regression + highlighted segments
    times = counts.index
    y = counts["req"].values

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, y, label="Observed req/10s")
    ax.plot(times, y_pred, label="Robust poly regression (deg=3)")
    for s, e in segments:
        ax.axvspan(s, e, alpha=0.2)
    ax.set_title("Web traffic with regression baseline and detected DDoS windows")
    ax.set_ylabel("Requests per 10 seconds")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "traffic_regression.png", dpi=200)
    plt.close(fig)

    # Plot: residuals + threshold
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, residuals, label="Residual (obs - pred)")
    ax.axhline(threshold, linestyle="--", label="Threshold (median + 6*MAD)")
    for s, e in segments:
        ax.axvspan(s, e, alpha=0.2)
    ax.set_title("Residuals and anomaly threshold")
    ax.set_ylabel("Residual")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "residuals.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
