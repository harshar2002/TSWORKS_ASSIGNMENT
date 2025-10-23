# Time Series Anomaly Detection for IoT Sensors

## Project Overview

This project implements an end-to-end machine learning pipeline for detecting anomalies in multivariate time series data from IoT sensors monitoring manufacturing equipment. The goal is early identification of equipment failure or maintenance needs by recognizing unusual sensor readings.

It uses the **NASA IMS Bearing Dataset** and implements:

* **Isolation Forest** – a statistical unsupervised model
* **LSTM Autoencoder** – a deep learning model capturing temporal dependencies

---

## Features and Approach

### Exploratory Data Analysis (EDA)

* Visualizations: histograms, boxplots, and correlation heatmaps
* Statistical summaries per snapshot and channel
* Outlier detection and treatment using IQR and z-score methods
* Insights on sensor dependencies and vibration trends

### Feature Engineering

* **Time-domain features:** mean, RMS, standard deviation, skewness, kurtosis, median, rolling statistics
* **Frequency-domain features:** spectral centroid, peak frequency, spectral entropy, energy ratios
* Features scaled using robust scaling to handle noise and outliers

### Labeling

* Heuristic approach: last 10% of snapshots marked as anomalies to approximate degradation onset
* Enables supervised evaluation despite the absence of ground-truth labels

---

## Model Selection

| Model            | Type          | Key Advantage                                 |
| ---------------- | ------------- | --------------------------------------------- |
| Isolation Forest | Unsupervised  | Detects statistical outliers without labels   |
| LSTM Autoencoder | Deep Learning | Captures temporal patterns and nonlinearities |

---

## Evaluation Metrics

* Precision, Recall, F1-score, ROC-AUC using heuristic labels
* Visual validation through anomaly time series plots

**Key Observations:**

* LSTM Autoencoder outperforms Isolation Forest in most datasets
* Both models struggle on datasets with more noise or distribution shifts
* High precision but moderate recall for Isolation Forest indicates missing some anomalies

---

## Requirements

### System Requirements

* Python 3.9+
* At least 8GB RAM (more recommended for LSTM training)
* Optional: GPU for faster deep learning training

### Python Libraries

Install via pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy tensorflow keras python-docx
```

Or, if using `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Dataset

* NASA IMS Bearing Dataset
* Recommended folder structure:

```
data/
├─ 1st_test/
├─ 2nd_test/
└─ 3rd_test/
```

> **Note:** Large datasets are not included in the repo; download separately if needed.

---

## Project Structure

```
project_folder/
│
├─ data/                         # Raw dataset folders
├─ Time_Series_Anomaly.ipynb     # Single Jupyter notebook with EDA, feature engineering, modeling, evaluation
├─ Time_Series_Anomaly_Report.docx
├─ README.md
└─ requirements.txt
```

---

## How to Run

1. **Clone the repository**

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
```

2. **Set up Python environment**

```bash
pip install -r requirements.txt
```

3. **Open and run the Jupyter Notebook**

```bash
jupyter notebook Time_Series_Anomaly.ipynb
```

* The notebook includes **EDA, feature engineering, model training, and evaluation** all in one file
* Modify dataset paths if needed

---

## Notes

* **Heuristic labeling:** Last 10% of snapshots marked as anomalous
* **LSTM Autoencoder:** GPU recommended for faster training
* Ensure dataset paths match your local setup

---

## Future Improvements

* Use true labels or expert annotations for better evaluation
* Extend models with attention mechanisms or ConvLSTM
* Deploy real-time anomaly detection pipelines with alerts
* Explore multimodal sensor fusion for richer fault diagnosis

---

## Contact

Prepared by: **Harsha R**
Date: **October 23, 2025**
