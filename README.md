# 💳 Credit Card Fraud Detection System

## 1. Project Explanation
An end-to-end Machine Learning system for detecting fraudulent credit card transactions in near real-time.

The system combines **synthetic transaction generation, feature engineering, machine learning, FastAPI model serving, and a real-time fraud monitoring dashboard** to identify suspicious transactions and classify them as **ALLOW, REVIEW, or BLOCK**.

---

## 🚀 Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a small percentage of all transactions.

This project builds a complete fraud detection pipeline that:

- Generates realistic synthetic transaction data
- Performs feature engineering and preprocessing
- Handles highly imbalanced fraud data
- Trains and tunes an XGBoost classification model
- Evaluates the model using PR-AUC
- Serves predictions through a FastAPI backend
- Streams transactions through the application
- Displays fraud predictions through an interactive dashboard
- Continuously updates transaction statistics and risk visualizations

---

## 🎯 Problem Statement

Financial institutions process thousands of transactions every second. Detecting fraudulent transactions quickly is critical because delayed detection can result in financial losses.

The objective of this project is to build a machine learning system that can:

1. Analyze transaction characteristics
2. Calculate the probability that a transaction is fraudulent
3. Apply a decision threshold
4. Classify transactions as:

```text

ALLOW
REVIEW
BLOCK
``` 
🏗️ System Architecture
                 ┌──────────────────────┐                               
                 │ Transaction Generator│
                 │   Synthetic Data     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Feature Engineering  │
                 │   & Preprocessing    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │     XGBoost Model    │
                 │  Fraud Classification│
                 └──────────┬───────────┘
                            │
                    Fraud Probability
                            │
                            ▼
                 ┌──────────────────────┐
                 │    Decision Layer    │
                 │ Allow / Review /     │
                 │       Block          │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │      FastAPI         │
                 │    Model Serving     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Web Dashboard      │
                 │ Transactions / Risk  │
                 │ Alerts / Analytics   │
                 └──────────────────────┘

  ---
  

🔄 Machine Learning Pipeline
Raw Transaction Data
        ↓
Data Generation / Loading
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Train / Test Split
        ↓
Preprocessing
        ↓
Baseline Models
        ↓
XGBoost Training
        ↓
Hyperparameter Optimization
        ↓
Model Evaluation
        ↓
Model Serialization
        ↓
fraud_xgb.joblib

⚡ Real-Time Prediction Pipeline

When the application is running, a transaction follows this flow:

New Transaction
      ↓
Transaction Stream
      ↓
FastAPI Backend
      ↓
Feature Processing
      ↓
XGBoost Model
      ↓
Fraud Probability
      ↓
Decision Threshold
      ↓
ALLOW / REVIEW / BLOCK
      ↓
Frontend Dashboard
      ↓
Live Charts + Transaction Ledger
📊 Dataset

The project uses synthetically generated transaction data instead of real banking data.

This avoids exposing sensitive financial and personally identifiable information.

The data generator creates:

Transaction ID
Timestamp
Transaction amount
Merchant category
Merchant ID
Card ID
City
Country
Device type
Transaction channel
Transaction hour
Day of week
Previous 24-hour transaction count
Previous 24-hour transaction amount
Previous 1-hour transaction count
Transaction velocity
International transaction indicator
Night transaction indicator
Fraud label

The current generator creates:

Total Transactions : 50,000
Fraud Transactions  : 750
Normal Transactions : 49,250
Fraud Rate          : 1.50%
🤖 Machine Learning Model

The primary fraud detection model is:

XGBoost

XGBoost is used because it performs well on structured/tabular data and can capture nonlinear relationships between transaction characteristics.

The model produces a probability:

0.00 ─────────────────────────── 1.00
Low Risk                         High Risk

A decision threshold is then applied to determine the transaction action.

Example:

Risk Score
    ↓
0.002 → ALLOW
0.15  → ALLOW
0.45  → REVIEW
0.92  → BLOCK
  
📈 Model Evaluation

Because fraud detection is a highly imbalanced classification problem, accuracy alone is not sufficient.

The project focuses on metrics such as:

Precision
Recall
PR-AUC
Confusion Matrix
False Positive Rate
Fraud Detection Rate

The current trained model achieves a PR-AUC of approximately:

PR-AUC: 0.9884

🔴 Transaction Decisions

The system categorizes transactions into three possible actions:

Decision	Meaning
🟢 ALLOW	Transaction has low fraud risk
🟡 REVIEW	Transaction requires additional investigation
🔴 BLOCK	Transaction has high fraud risk
⚡ Live Transaction Streaming

The dashboard supports continuous transaction monitoring.

When the stream is started:

Transaction Generated
        ↓
Backend Receives Transaction
        ↓
ML Model Generates Prediction
        ↓
Risk Score Calculated
        ↓
Decision Generated
        ↓
Frontend Receives Result
        ↓
Dashboard Updates

The frontend updates:

Transaction count
Risk score chart
Fraud alerts
Transaction ledger
Blocked transaction count
Financial loss prevented
🛠️ Tech Stack
Programming
Python
JavaScript / TypeScript
Data Science
Pandas
NumPy
Machine Learning
Scikit-learn
XGBoost
Optuna
Backend
FastAPI
Uvicorn
Frontend
Next.js
React
Tailwind CSS
Development
Git
GitHub
Virtual Environment
📁 Project Structure
Credit-Card-Fraud-Detection-System/
│
├── apps/
│   └── web/
│       ├── app/
│       ├── public/
│       ├── package.json
│       └── package-lock.json
│
├── data/
│   ├── generate_data.py
│   └── .gitkeep
│
├── models/
│   └── model_card.json
│
├── notebooks/
│
├── serving/
│   ├── __init__.py
│   └── app.py
│
├── src/
│   ├── features.py
│   ├── pipeline.py
│   ├── train_baselines.py
│   └── tune_optuna.py
│
├── outputs/
│
├── main.py
├── requirements.txt
├── README.md
└── .gitignore

🔮 Future Improvements
Potential future improvements include:

Apache Kafka based transaction streaming
WebSocket-based real-time communication
Model monitoring
Data drift detection
Automated model retraining
Explainable AI using SHAP
Authentication and role-based access
Cloud deployment
Docker containerization
Database integration
Production-grade monitoring

👨‍💻 Author
Darshan Chavan
GitHub:
https://github.com/Darsha0018
LinkedIn:
https://www.linkedin.com/in/darshan-chavan-24162a27a/
