💳 Credit Card Fraud Detection System
🚀 Project Overview

An end-to-end Machine Learning system for detecting fraudulent credit card transactions in near real-time.

The system combines synthetic transaction generation, feature engineering, machine learning, FastAPI model serving, and a real-time fraud monitoring dashboard to identify suspicious transactions and classify them as ALLOW, REVIEW, or BLOCK.

🎯 Problem Statement

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a small percentage of all transactions.

The objective of this project is to build a machine learning system that can:

Analyze transaction characteristics
Calculate the probability that a transaction is fraudulent
Apply a decision threshold
Classify transactions as:
🟢 ALLOW
🟡 REVIEW
🔴 BLOCK
Display the results through an interactive monitoring dashboard
🏗️ System Architecture
┌──────────────────────────┐
│  Transaction Generator   │
│    Synthetic Data        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Feature Engineering    │
│    & Preprocessing       │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│      XGBoost Model       │
│   Fraud Classification   │
└────────────┬─────────────┘
             │
       Fraud Probability
             │
             ▼
┌──────────────────────────┐
│     Decision Layer       │
│  Allow / Review / Block  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│        FastAPI           │
│      Model Serving       │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│      Web Dashboard       │
│ Transactions / Risk /    │
│ Alerts / Analytics       │
└──────────────────────────┘
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

When the application is running, each transaction follows this flow:

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

Generated Features
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
Dataset Statistics
Metric	Value
Total Transactions	50,000
Fraud Transactions	750
Normal Transactions	49,250
Fraud Rate	1.50%
🤖 Machine Learning Model
XGBoost

The primary fraud detection model is XGBoost.

XGBoost is used to model relationships between transaction characteristics and identify potentially fraudulent behavior.

The model produces a fraud probability between 0 and 1.

0.00 ─────────────────────────── 1.00
Low Risk                         High Risk

A decision threshold is then applied to determine the transaction action.

Example
Risk Score
    ↓
0.002 → ALLOW
0.15  → ALLOW
0.45  → REVIEW
0.92  → BLOCK
📈 Model Evaluation

Because fraud detection is a highly imbalanced classification problem, accuracy alone is not sufficient.

The project focuses on:

Precision
Recall
PR-AUC
Confusion Matrix
False Positive Rate
Fraud Detection Rate
Current Model Performance
PR-AUC: 0.9884
🔴 Transaction Decisions
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

The frontend continuously updates:

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
Python Virtual Environment
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
Apache Kafka-based transaction streaming
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

📄 License

This project is licensed under the MIT License.

in such a way that i just copy paste it in a file 
# 💳 Credit Card Fraud Detection System

## 🚀 Project Overview

An end-to-end **Machine Learning system for detecting fraudulent credit card transactions in near real-time**.

The system combines **synthetic transaction generation, feature engineering, machine learning, FastAPI model serving, and a real-time fraud monitoring dashboard** to identify suspicious transactions and classify them as **ALLOW, REVIEW, or BLOCK**.

---

## 🎯 Problem Statement

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions represent only a small percentage of all transactions.

The objective of this project is to build a machine learning system that can:

1. Analyze transaction characteristics
2. Calculate the probability that a transaction is fraudulent
3. Apply a decision threshold
4. Classify transactions as:
   - 🟢 **ALLOW**
   - 🟡 **REVIEW**
   - 🔴 **BLOCK**
5. Display the results through an interactive monitoring dashboard.

---

## 🏗️ System Architecture

```text
┌──────────────────────────┐
│  Transaction Generator   │
│    Synthetic Data        │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   Feature Engineering    │
│    & Preprocessing       │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│      XGBoost Model       │
│   Fraud Classification   │
└────────────┬─────────────┘
             │
       Fraud Probability
             │
             ▼
┌──────────────────────────┐
│     Decision Layer       │
│  Allow / Review / Block  │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│        FastAPI           │
│      Model Serving       │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│      Web Dashboard       │
│ Transactions / Risk /    │
│ Alerts / Analytics       │
└──────────────────────────┘
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

When the application is running, each transaction follows this flow:

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

Generated Features
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
Dataset Statistics
Metric	Value
Total Transactions	50,000
Fraud Transactions	750
Normal Transactions	49,250
Fraud Rate	1.50%
🤖 Machine Learning Model
XGBoost

The primary fraud detection model is XGBoost.

XGBoost is used to model relationships between transaction characteristics and identify potentially fraudulent behavior.

The model produces a fraud probability between 0 and 1.

0.00 ─────────────────────────── 1.00
Low Risk                         High Risk

A decision threshold is then applied to determine the transaction action.

Example
Risk Score
    ↓
0.002 → ALLOW
0.15  → ALLOW
0.45  → REVIEW
0.92  → BLOCK
📈 Model Evaluation

Because fraud detection is a highly imbalanced classification problem, accuracy alone is not sufficient.

The project focuses on:

Precision
Recall
PR-AUC
Confusion Matrix
False Positive Rate
Fraud Detection Rate
Current Model Performance
PR-AUC: 0.9884
🔴 Transaction Decisions
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

The frontend continuously updates:

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
Python Virtual Environment
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
⚙️ Installation & Setup
1. Clone the Repository
git clone https://github.com/Darsha0018/Credit-Card-Fraud-Detection-System.git
cd Credit-Card-Fraud-Detection-System
2. Create a Virtual Environment
python -m venv .venv
Windows
.venv\Scripts\activate
3. Install Python Dependencies
pip install -r requirements.txt
📊 Generate Transaction Data

Run:

python data/generate_data.py

This generates the synthetic transaction dataset.

🤖 Train the Model

Run:

python src/tune_optuna.py

The trained model is then used by the FastAPI serving application.

🚀 Start the Backend

Run:

python -m uvicorn serving.app:app --reload

The API will be available at:

http://127.0.0.1:8000
🌐 Start the Frontend

Open another terminal:

cd apps/web

Install dependencies:

npm install

Start the development server:

npm run dev

The dashboard will normally be available at:

http://localhost:3000
🔌 Backend + Frontend Communication

The machine learning backend and web dashboard communicate through the API.

                 FRONTEND
              Next.js / React
                    │
                    │ HTTP Requests
                    ▼
                 BACKEND
                  FastAPI
                    │
                    ▼
              ML Prediction
                    │
                    ▼
               XGBoost Model
                    │
                    ▼
             Prediction Result
                    │
                    ▼
                 Frontend
📸 Dashboard

The dashboard provides:

Live transaction monitoring
Fraud risk scores
Fraud alerts
Transaction ledger
Blocked transaction statistics
Financial loss prevention statistics
Model information
Continuous transaction updates

Add your dashboard screenshot here:

![Fraud Detection Dashboard](assets/dashboard.png)
✨ Key Features
✅ Synthetic transaction generation
✅ Fraud and normal transaction simulation
✅ Feature engineering
✅ Data preprocessing
✅ Imbalanced classification
✅ XGBoost fraud detection
✅ Hyperparameter optimization with Optuna
✅ PR-AUC based evaluation
✅ FastAPI model serving
✅ Real-time transaction streaming
✅ Live fraud risk monitoring
✅ Transaction ledger
✅ Fraud alerts
✅ Allow / Review / Block decisions
✅ Financial loss prevention tracking
✅ Interactive web dashboard
🔮 Future Improvements
Apache Kafka-based transaction streaming
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
