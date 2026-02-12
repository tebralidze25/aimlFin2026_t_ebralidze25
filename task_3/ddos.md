# Task 3 — Web server log analysis (DDoS detection via regression)

## Input data
- Provided log source (as in the assignment): `http://max.ge/aiml_final/t_ebralidze25_36714_server.log`
- This repository also includes the same log file for reproducibility:
  - `t_ebralidze25_36714_server.log`

## Goal
Identify the time interval(s) of a **DDoS attack** in the provided web-server access log using **regression analysis**.

## Method overview
1. **Parse timestamps** from each log line (timezone-aware).
2. **Aggregate** requests into a time series: **requests per 10 seconds**.
3. Fit a **robust polynomial regression** baseline (degree 3) to model normal traffic trend:
   - Robustness is achieved using **Huber regression**, which reduces the influence of extreme spikes.
4. Compute **residuals**: \( r_t = y_t - \hat{y}_t \).
5. Flag anomalies using a robust threshold based on **MAD** (median absolute deviation):
   - \( \text{MAD} = \text{median}(|r_t - \text{median}(r)|) \)
   - **Threshold:** \( \tau = \text{median}(r) + 6\cdot \text{MAD} \)
6. Convert consecutive anomalous buckets into **attack intervals**, allowing small gaps (\<= 20s).

## Results — detected DDoS intervals (UTC+04:00)
The regression + residual analysis detected **two** DDoS windows:

- **2024-03-22 18:06:00+04:00 → 2024-03-22 18:08:00+04:00**
- **2024-03-22 18:09:00+04:00 → 2024-03-22 18:11:00+04:00**

During these windows, the traffic jumps to **~2000–2600 requests per 10 seconds**, far above the modeled baseline.

## Key code fragments

### 1) Timestamp parsing
```python
TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\+\d{2}:\d{2})\]")
...
dt = pd.to_datetime(pd.Series(ts), format="%Y-%m-%d %H:%M:%S%z")
```

### 2) Aggregation to requests/10s
```python
df["bucket"] = df["ts"].dt.floor("10s")
counts = df.groupby("bucket").size().rename("req").to_frame()
```

### 3) Robust regression baseline + residual thresholding
```python
model = make_pipeline(PolynomialFeatures(degree=3, include_bias=False), HuberRegressor())
model.fit(t_hours, y)
y_pred = model.predict(t_hours)
residuals = y - y_pred

med = np.median(residuals)
mad = np.median(np.abs(residuals - med))
threshold = med + 6 * mad
mask = residuals > threshold
```

## Visualizations
### Traffic + regression baseline (highlighted DDoS windows)
![Traffic regression](figures/traffic_regression.png)

### Residuals + anomaly threshold
![Residuals](figures/residuals.png)

## How to reproduce
From the `task_3/` directory:

```bash
python ddos_regression.py --log t_ebralidze25_36714_server.log --outdir figures
```

This prints the detected intervals and regenerates the figures in `figures/`.
