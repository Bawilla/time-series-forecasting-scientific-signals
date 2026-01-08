
# LSTM-Based Anomaly Detection & Degradation Modeling
### NASA Turbofan Engine (CMAPSS)

This project implements **forecast-based anomaly detection and degradation modeling** on the **NASA CMAPSS turbofan engine dataset** using an **LSTM neural network**.

The model learns nominal multivariate sensor dynamics and uses **forecast error** as a proxy for system degradation.

---

##  Objectives
- Learn normal engine behavior via time-series forecasting
- Detect anomalies using forecast (reconstruction) error
- Align degradation signal with **true Remaining Useful Life (RUL)**
- Construct an interpretable **Health Index (HI)**
- Perform early **failure threshold detection**

---

##  Dataset
- **NASA CMAPSS Turbofan Engine Degradation Dataset**
- Multivariate sensor time series (21 sensors)
- Full run-to-failure trajectories
- Subset used: **FD001**

Downloaded automatically from Kaggle inside Google Colab.

---

##  Methodology
1. Normalize sensor signals
2. Create sliding-window LSTM sequences
3. Train LSTM to predict next-step sensor values
4. Compute forecast error as anomaly signal
5. Align forecast error with true RUL
6. Convert error to Health Index
7. Detect failure using statistical thresholds

---

##  Key Outputs
- Training & validation loss curves
- Forecast error (global and per-engine)
- Error vs true RUL correlation
- Health Index over engine lifetime
- Failure threshold detection
- Confidence bands for uncertainty visualization

All plots are saved to the `figures/` directory for GitHub compatibility.

---

##  How to Run (Google Colab)
1. Open the notebook in Colab
2. Upload your `kaggle.json` API key when prompted
3. Run cells top-to-bottom
4. Figures are saved automatically (no reruns needed)

---

##  Notes
- Forecast error is **not a direct RUL predictor**
- It serves as a **degradation-sensitive anomaly indicator**
- Method is generalizable to other scientific time-series signals

---

## Project Structure
