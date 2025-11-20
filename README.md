# Anomaly Detection — LSTM Autoencoder 

This repository contains a Streamlit web app that demonstrates an LSTM Autoencoder-based anomaly detection pipeline designed for time-series telemetry (e.g.spacecraft telemetry). The project uses synthetic data for simulation, label-based anomaly injection for benchmarking and reconstruction-error-based scoring.

## Features
- Synthetic multivariate time-series generator (configurable channel count and length)
- Label-based anomaly injection (configurable number & magnitude of anomaly windows)
- Sliding-window preprocessing (configurable window size & step)
- LSTM Autoencoder training on 'normal' windows
- Reconstruction-error scoring and dynamic thresholding (std-based or percentile-based)
- Interactive visualization of errors, prediction timeline and channel overlays
- Model & results download

## How to run
1. Create a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate  # windows
```

2. Install dependencies:

```bash
pip install streamlit tensorflow scikit-learn pandas matplotlib
```

3. Run the app:

```bash
streamlit run app.py
```

## Files
- `app.py` — main Streamlit app
- `README.md` — info file

## Design choices and options
- The app trains on windows labeled as 'normal' and evaluates reconstruction error on all windows.
- Thresholding: default is mean + k * std, but percentile-based thresholding is also available.
- Synthetic anomalies include spikes, drift and increased noise to mimic realistic telemetry faults.
