# Xgboost-Imbalanced-Time-Series-Project
This project implements a complete, production-ready workflow for building and optimizing an XGBoost-based classifier on a highly imbalanced time-series dataset. It focuses on advanced feature engineering, rigorous model evaluation, and hyperparameter optimization suitable for real-world anomaly detection or financial/sensor analytics tasks.
# Time-Series Anomaly Detection using XGBoost & Optuna

This project demonstrates a complete **end-to-end anomaly detection pipeline** for multivariate time‑series data using:

* Synthetic dataset generation
* Sliding window feature extraction
* TimeSeriesSplit cross‑validation
* Hyperparameter tuning with Optuna
* XGBoost classifier
* Evaluation using Precision, Recall, F1‑Score, AUPRC
* Visualization of Precision-Recall Curve
* Saving model & scaler

---

## 📌 Project Structure

```
├── report.md              # Full technical report
├── README.md              # Project overview & instructions
├── requirements.txt        # Dependencies
├── xgb_timeseries_model.joblib   # Saved model
├── scaler.joblib                  # Saved scaler
└── main.ipynb             # Google Colab notebook
```

---

## 🚀 Features

* Generates realistic multi-channel sensor-like time-series
* Injects rare anomalies (< 5%) as required
* Extracts rich feature vectors using sliding windows
* Performs robust time-series-aware validation using TimeSeriesSplit
* Uses Optuna to find optimal XGBoost hyperparameters
* Computes detailed evaluation metrics
* Produces plots for analysis
* Saves trained model & scaler for deployment

---

## 🧩 Dataset

The dataset is synthetic and generated using sinusoidal signals with added:

* noise
* trend
* anomaly bursts (3 to 20 timesteps)

Each window is labelled anomalous if **any timestep inside the window** is anomalous.

---

## 🛠 Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install xgboost optuna scikit-learn pandas numpy scipy matplotlib seaborn joblib
```

---

## ▶️ Run the Project (Google Colab)

Upload the notebook and run all cells.
The main execution is done using:

```python
results = run_pipeline(n_samples=10000,
                       anomaly_fraction=0.03,
                       window=50,
                       n_channels=3,
                       step=1,
                       n_trials=30,
                       n_splits=4,
                       random_state=42)
```

---

## 📊 Output

The pipeline outputs:

* Best hyperparameters found by Optuna
* Confusion matrix
* Precision, Recall, F1-score
* Area Under Precision-Recall Curve
* PR Curve plot

Model files saved:

```
xgb_timeseries_model.joblib
scaler.joblib
```

---

## 📦 Deployment

Load the saved model and scaler:

```python
import joblib
model = joblib.load('xgb_timeseries_model.joblib')
scaler = joblib.load('scaler.joblib')
```

Then use:

```python
scaled_X = scaler.transform(X_new)
pred = model.predict_proba(scaled_X)[:, 1]
```

---

## 📚 References

* XGBoost Documentation
* Optuna Hyperparameter Optimization
* Scikit-Learn TimeSeriesSplit
* Matplotlib & Seaborn for Visualization

---


