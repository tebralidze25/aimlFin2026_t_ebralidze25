"""
DDoS detection using regression on web server logs.

Reads an Apache/Nginx-like combined log with lines formatted as:
IP - - [YYYY-MM-DD HH:MM:SS+TZ] "METHOD /path HTTP/1.0" STATUS BYTES "REF" "UA" RESP_TIME

Outputs:
- Detected DDoS time intervals (UTC+04:00)
- Plots in ./figures/

Usage:
  python ddos_regression.py t_ebralidze25_36714_server.log
"""

import re
import sys
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG_RE = re.compile(
    r'^(?P<ip>\S+) \S+ \S+ \[(?P<ts>[^\]]+)\] "(?P<method>[A-Z]+) (?P<path>\S+) (?P<proto>[^"]+)" '
    r'(?P<status>\d{3}) (?P<size>\d+) "(?P<ref>[^"]*)" "(?P<ua>[^"]*)" (?P<rt>\d+)\s*$'
)

def parse_log(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            m = LOG_RE.match(line)
            if not m:
                continue
            gd = m.groupdict()
            ts = datetime.fromisoformat(gd["ts"])  # keeps timezone
            rows.append((ts, gd["ip"], gd["method"], gd["path"], int(gd["status"]), int(gd["size"]), int(gd["rt"])))
    df = pd.DataFrame(rows, columns=["ts","ip","method","path","status","size","rt"])
    return df

def detect_intervals(ts_index: pd.DatetimeIndex, is_attack: np.ndarray, bin_seconds: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    attack_times = ts_index[is_attack]
    if len(attack_times) == 0:
        return []
    intervals = []
    start = attack_times[0]
    prev = attack_times[0]
    for cur in attack_times[1:]:
        if (cur - prev).total_seconds() <= bin_seconds:
            prev = cur
        else:
            intervals.append((start, prev + pd.Timedelta(seconds=bin_seconds)))
            start = cur
            prev = cur
    intervals.append((start, prev + pd.Timedelta(seconds=bin_seconds)))
    return intervals

def main():
    if len(sys.argv) < 2:
        print("Usage: python ddos_regression.py <log_file>")
        sys.exit(1)

    log_path = sys.argv[1]
    df = parse_log(log_path)

    # Aggregate traffic per 10 seconds
    bin_seconds = 10
    df["bin"] = df["ts"].dt.floor(f"{bin_seconds}s")
    s = df.groupby("bin").size().rename("req").sort_index()

    ts = s.index
    y = s.values.astype(float)
    t = (ts - ts[0]).total_seconds().astype(float)

    # Regression baseline (poly degree 3) trained on "normal" bins (<= 90th percentile)
    p90 = np.percentile(y, 90)
    train_mask = y <= p90

    coef = np.polyfit(t[train_mask], y[train_mask], 3)
    yhat = np.polyval(coef, t)
    res = y - yhat

    # Residual threshold from training data
    thr_res = np.quantile(res[train_mask], 0.995)

    # Additional robust volume threshold (median + 10*MAD) to avoid mild spikes
    median = np.median(y)
    mad = np.median(np.abs(y - median))
    thr_vol = median + 10 * mad

    is_attack = (res > thr_res) & (y > thr_vol)

    intervals = detect_intervals(ts, is_attack, bin_seconds)

    print("Detected DDoS intervals (UTC offset preserved):")
    for a, b in intervals:
        print(f" - {a.isoformat()}  to  {b.isoformat()}")

    # ---- Visualizations ----
    import os
    os.makedirs("figures", exist_ok=True)

    plt.figure(figsize=(12,4))
    plt.plot(ts, y, label=f"Requests / {bin_seconds}s")
    plt.plot(ts, yhat, label="Regression baseline (poly3)")
    plt.scatter(ts[is_attack], y[is_attack], s=18, label="Attack bins")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Time")
    plt.ylabel("Requests")
    plt.title("Traffic with regression baseline and detected attack bins")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/traffic_regression.png", dpi=160)
    plt.close()

    plt.figure(figsize=(12,3.5))
    plt.plot(ts, res, label="Residual (actual - predicted)")
    plt.axhline(thr_res, linestyle="--", label="Residual threshold (99.5% train)")
    plt.scatter(ts[is_attack], res[is_attack], s=18, label="Attack bins")
    plt.xticks(rotation=45, ha="right")
    plt.xlabel("Time")
    plt.ylabel("Residual")
    plt.title("Residuals and detection threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/residuals.png", dpi=160)
    plt.close()

if __name__ == "__main__":
    main()
