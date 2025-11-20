# Demo Steps — LSTM Autoencoder Anomaly Detection App

This document explains **how to run**, **how to explore** and **how to demonstrate every feature** of the Streamlit-based LSTM Autoencoder anomaly detection system. 
---

## 1. Overview

This application demonstrates a complete anomaly detection pipeline using an **LSTM Autoencoder** trained on synthetic telemetry. It supports:

* Synthetic time-series generation
* Label-based anomaly injection
* Sliding‑window preprocessing
* LSTM Autoencoder training
* Reconstruction error–based anomaly scoring
* Dynamic thresholding
* Visualizations (Error plot, binary timeline, channel overlays)
* Metrics (Precision, Recall, F1)
* Download of results, model

---

## 2. How to Run the App

Follow these steps:

### **Step 1 — Install Dependencies**

```bash
pip install streamlit tensorflow scikit-learn pandas matplotlib
```

### **Step 2 — Run Streamlit Application**

Navigate to the folder containing `app.py` and run:

```bash
streamlit run app.py
```

### **Step 3 — Open Browser**

If the browser does not open automatically, visit:

```
http://localhost:8501
```

---

## 3. Step-by-Step Demo Guide

This section explains how to explore every feature in a structured flow.

### **3.1 Generate Synthetic Data**

1. Go to the **sidebar**.
2. Set parameters:

   * Number of channels (e.g., 4)
   * Length of time series (e.g., 7000)
   * Window size (e.g., 70)
   * Window step (e.g., 2)
3. Click **"Generate synthetic data"**.
4. A preview table will appear showing the first rows of the dataset.

**Result:** Synthetic telemetry is now ready.

---

### **3.2 Inject Anomalies**

1. Choose the number of anomaly windows (default 3).
2. Set anomaly magnitude (default 2.0).
3. Click **"Inject anomalies (labels)"**.
4. The app injects spikes, drift and noise into selected windows.

**Result:** Fault‑like patterns appear in the data and labels are stored.

---

### **3.3 Prepare & Train LSTM Autoencoder**

1. Scroll to **Train LSTM Autoencoder**.
2. Click **"Prepare & Train"**.
3. App automatically performs:

   * Sliding window creation
   * Selecting "normal" windows for training
   * Standardization
   * Model building
   * LSTM Autoencoder training

**Expected Output:**

* Printed model summary
* Training complete message
* Training loss curve saved

---

## 4. Visualizations & Results

### **4.1 Reconstruction Error Plot**

Shows model reconstruction error for each window.

* Spikes indicate potential anomalies.
* Red dashed line represents threshold.

**Interpretation:** Windows above threshold = anomaly.

---

### **4.2 Anomaly Prediction Timeline**

A binary timeline showing:

* **1 = anomaly detected**
* **0 = normal**

Helps visualize coverage and detection periods.

---

### **4.3 Channel Overlay Plot**

1. Select channels in “Channels to overlay.”
2. The app overlays model-detected anomalies on raw telemetry.
3. Red shaded regions show anomaly windows.

Useful for understanding how anomalies manifest in raw signals.

---

## 5. Metrics

If anomalies exist in labels:

* Precision
* Recall
* F1 Score

These evaluate how well the model detects injected faults.

---

## 6. Downloadable Outputs

After training, the following become available:

### **6.1 Anomaly Results CSV**

Contains:

* center_idx
* reconstruction error
* binary anomaly prediction
* ground‑truth labels

### **6.2 Trained Model (.keras)**

Saved Keras model for reuse.

---

## 7. Experimentation Guide

Try variations to explore model behaviour.

### **7.1 Threshold Sensitivity Test**

* Change threshold method to `percentile` (e.g., 95%).
* Observe changes in FP/FN.

### **7.2 Window Size Experiment**

* Test window sizes: 32, 64, 128.
* Smaller = more precise but noisy.
* Larger = smoother but less localized.

### **7.3 Latent Dimension Experiment**

* Try latent dimensions: 16, 64, 128.
* Higher dimension learns complex patterns.

### **7.4 Anomaly Magnitude Variation**

* Increase anomaly strength to simulate serious faults.

Each setting shows how robust the model is.

---

## 8. Demo Checklist

Use this for live demos.

```
[ ] Run "Generate synthetic data"
[ ] Inject anomalies
[ ] Prepare & Train Autoencoder
[ ] Show reconstruction error plot
[ ] Show anomaly timeline
[ ] Show channel overlays
[ ] Show precision/recall/F1
[ ] Download CSV, model
```

---

## 9. Conclusion

This app provides a complete framework to understand how LSTM Autoencoders detect anomalies in time‑series spacecraft systems. The data pipeline ensures full demonstration even without real telemetry.

