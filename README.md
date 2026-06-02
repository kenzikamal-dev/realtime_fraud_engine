# Real-Time Fraud Detection Engine (FICO-style)

## Overview

This project implements a real-time fraud detection system inspired by FICO, designed for fintech transaction monitoring and risk scoring.

The system uses an XGBoost machine learning model to predict fraud probability (0–1) and classify transactions into Low, Medium, or High Risk categories.

The project demonstrates a complete Machine Learning Engineering workflow including:

* Business Understanding
* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Feature Engineering
* Model Training
* Model Evaluation
* Streamlit Deployment
* Production-Oriented Project Structure

---

## Business Problem

Financial institutions process millions of transactions daily.

Fraudulent transactions cause significant financial losses, operational costs, and customer trust issues.

This project simulates a real-world fraud detection engine that can:

* Analyze transaction characteristics
* Predict fraud probability in real time
* Flag suspicious activity
* Support fraud analysts with risk scoring

---

## Key Features

### Machine Learning

* XGBoost Fraud Detection Model
* Imbalanced Data Handling
* Stratified Train/Test Split
* Probability-Based Risk Scoring
* ROC-AUC Evaluation

### Fraud Risk Classification

| Risk Score  | Risk Level  |
| ----------- | ----------- |
| 0.00 – 0.30 | Low Risk    |
| 0.30 – 0.70 | Medium Risk |
| 0.70 – 1.00 | High Risk   |

### Streamlit Dashboard

* Real-time Fraud Prediction
* Interactive User Interface
* Probability Score Display
* Risk Category Classification
* Fintech-Style Dashboard Design

---

## Project Architecture

```text
Raw Transaction Data
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
XGBoost Fraud Model
        │
        ▼
Fraud Probability (0–1)
        │
        ▼
Risk Classification
        │
        ▼
Streamlit Dashboard
```

---

## Project Structure

```text
realtime_fraud_engine/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── models/
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   └── screenshots/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
│
├── tests/
│
├── requirements.txt
│
└── README.md
```

---

## Dataset

Dataset: Credit Card Fraud Detection Dataset

Characteristics:

* 284,807 transactions
* 492 fraud transactions
* Highly imbalanced dataset
* PCA-transformed features (V1–V28)
* Transaction Amount
* Transaction Time

Target Variable:

```text
Class
0 = Legitimate Transaction
1 = Fraudulent Transaction
```

---

## Technology Stack

### Programming Language

* Python 3.11

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-Learn
* XGBoost

### Visualization

* Matplotlib
* Seaborn

### Deployment

* Streamlit

### Model Persistence

* Joblib

---

## Model Performance

### XGBoost Results

| Metric            | Score  |
| ----------------- | ------ |
| ROC-AUC           | 0.978  |
| Precision (Fraud) | 0.88   |
| Recall (Fraud)    | 0.87   |
| Accuracy          | 99.95% |

Confusion Matrix:

```text
[[56852    12]
 [   13    85]]
```

Classification Report:

```text
Precision: 0.88
Recall:    0.87
F1-Score:  0.87
ROC-AUC:   0.978
```

---

## Screenshots

### Fraud Detection Dashboard

Add screenshot:

```text
reports/screenshots/streamlit_ui.png
```

### Model Evaluation

Add screenshot:

```text
reports/screenshots/model_metrics.png
```

---

## Installation

### Clone Repository

```bash
git clone <your-github-repository-url>
cd realtime_fraud_engine
```

### Create Environment

```bash
conda create -n fraud-ml python=3.11
conda activate fraud-ml
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

## Run Application

Verify Environment:

```bash
which python
python --version
```

Run Streamlit:

```bash
python -m streamlit run app/streamlit_app.py
```

---

## Future Improvements

* FastAPI Inference API
* Real-Time Transaction Streaming
* Model Monitoring Dashboard
* Drift Detection
* Model Versioning
* Cloud Deployment
* Database Integration
* MLOps Pipeline

---

## Resume Highlights

* Built a real-time fraud detection engine using XGBoost achieving ROC-AUC of 0.978.
* Developed a FICO-inspired risk scoring system for fintech transaction monitoring.
* Created a Streamlit dashboard for live fraud probability prediction.
* Implemented an end-to-end machine learning workflow from data preprocessing to deployment.
* Applied machine learning techniques to highly imbalanced financial transaction data.

---

## License

MIT License

---

## Author

Machine Learning Engineering Portfolio Project

Real-Time Fraud Detection Engine (FICO-Inspired)
