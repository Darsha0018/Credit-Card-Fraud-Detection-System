# Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Next.js](https://img.shields.io/badge/Next.js-15.0+-black)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

An end-to-end Machine Learning system that detects fraudulent credit card transactions in near real-time. It utilizes cost-sensitive learning techniques for imbalanced classification, exposes a scoring API (batch + streaming hook), and visualizes precision-recall trade-offs, fraud alerts, and feature impacts in a highly responsive Next.js dashboard.

---

## 1. Project Explanation

### What is Credit Card Fraud Detection?

Credit Card Fraud Detection is the process of identifying unauthorized, malicious transactions before they are approved. It relies on analyzing historical transaction data to recognize patterns of legitimate versus fraudulent behavior.

### Why is it Important & What Problems Does it Solve?

Fraud costs the global economy billions of dollars annually. This system prevents direct financial loss for banks and customers, reduces the operational costs of manual reviews, and protects the institution's reputation.

### How Banks and Fintechs Use It

Banks use low-latency machine learning pipelines to score transactions in milliseconds. If a transaction's fraud probability exceeds a dynamic threshold, the system triggers an alert to either block the transaction instantly or send it to a human analyst for review.

### The Workflow

Raw Transaction Data → Preprocessing (handling missing values, scaling) → Feature Engineering (velocity, transaction ratios) → ML Model Inference → Fraud Prediction (Probability Score) → Dashboard Alert (Allow/Review/Block).

---

## 2. Tech Stack

This project implements an **Intermediate to Advanced** tech stack, ideal for bridging Data Science and Software Engineering.

- **Data Processing:** Python, Pandas, NumPy
- **Machine Learning:** Scikit-learn, XGBoost/LightGBM, Imbalanced-learn (SMOTE)
- **Model Serving:** FastAPI (Python)
- **Frontend Dashboard:** Next.js (React), Tailwind CSS, Custom SVG Charts
- **Evaluation Metrics:** PR-AUC, Recall@FPR, Confusion Matrix

---

## 3. Project Architecture

### Data Flow

1. **Input:** Transaction data (Amount, Time, Location, Merchant Category, Card ID).
2. **Processing:** PII removal, chronological splitting, scaling, and engineering velocity features.
3. **Model:** Cost-sensitive classification model (e.g., XGBoost) optimized for PR-AUC.
4. **Output:** A continuous probability score [0, 1] evaluated against a dynamic threshold to output a decision.

### Text-Based Diagram

```text
[ Raw Transactions (CSV/Parquet) ]
       ↓
[ Feature Engineering Pipeline (Velocity, Aggregates) ]
       ↓
[ ML Model (XGBoost / Random Forest) ] ← Trained with SMOTE / Class Weights
       ↓
[ FastAPI Scoring Endpoint (/score) ]
       ↓
[ Next.js Fraud Ops Dashboard (Live Visualization & Thresholding) ]
```

---

## 4. Implementation Plan

- **Phase 1 (Setup):** Initialize Python environment and Next.js frontend.
- **Phase 2 (Data Loading):** Ingest public/synthetic credit card data safely.
- **Phase 3 (Data Cleaning):** Handle missing values and format data types.
- **Phase 4 (EDA):** Analyze class imbalance and plot distribution metrics.
- **Phase 5 (Feature Engineering):** Create time-based velocity and categorical frequency features.
- **Phase 6 (Model Training):** Train baselines and tune XGBoost with `scale_pos_weight`.
- **Phase 7 (Evaluation):** Measure PR-AUC and plot Precision-Recall curves.
- **Phase 8 (Prediction API):** Wrap the model in a FastAPI endpoint.
- **Phase 9 (Visualization):** Connect the Next.js dashboard to visualize metrics.
- **Phase 10 (Deployment):** Prepare the repository for GitHub and portfolio presentation.

---

## 5. Folder Structure

```text
Credit-Card-Fraud-Detection/
│
├── data/                    # Synthetic or public transaction datasets
├── notebooks/               # Jupyter notebooks for EDA and model experimentation
├── src/                     # Python source code (features, pipeline, train)
├── models/                  # Serialized ML models (.joblib / .pkl)
├── apps/web/                # Next.js Frontend Dashboard
│   ├── app/                 # React components, pages, and API routes
│   └── globals.css          # Tailwind styling
├── assets/                  # Images and screenshots for documentation
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies
```

---

## 6. Installation & How to Run

### Requirements

- Python 3.10+
- Node.js 18+

### Setup Instructions

1. **Clone the repository:**

```bash
git clone https://github.com/Sonia068/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

2. **Run the Frontend Dashboard:**

```bash
cd apps/web
npm install
npm run dev
# The dashboard will be available at http://localhost:3000
```

_(Note: Ensure you have run the data generation or connected the API route for the dashboard to populate with transaction data)._

---

## 7. Virtual Simulation

Because real banking data is highly confidential, this project uses simulated, PII-safe data.

- **Simulation Logic:** The system generates synthetic transactions. Normal transactions follow standard consumer habits. Fraudulent transactions mimic velocity attacks (multiple rapid transactions) or sudden high-value purchases at unusual hours.
- **Detection:** The ML model picks up on these velocity spikes and contextual anomalies, outputting a high risk score.
- **Alerts:** The dashboard visually flags any transaction crossing the analyst-defined threshold (e.g., > 70% risk) in bright red, moving it to the blocked queue.

---

## 8. Outputs & Screenshots

### Main Dashboard Overview

![Main Dashboard](assets/dashboard_overview.png)

### Deep Analytics View

![Analytics View](assets/analytics_view.png)

### Transaction Ledger

![Transaction Ledger](assets/transaction_ledger.png)

### Demo Video

[![Watch Demo](https://img.youtube.com/vi/HQBtJNfBPzk/0.jpg)](https://youtu.be/HQBtJNfBPzk)

---

## 9. Proof Strategy & Key Learnings

**Day-wise Execution Strategy:**

- **Day 1:** Project setup, requirement gathering, and virtual environment creation.
- **Day 2:** Dataset ingestion, schema enforcement, and chronological splitting.
- **Day 3:** Exploratory Data Analysis (EDA) and feature engineering (velocity ratios).
- **Day 4:** Model training, handling class imbalance, and threshold optimization.
- **Day 5:** FastAPI integration, Next.js Dashboard development, and results documentation.

**Key Learnings:**

- Managing extreme class imbalance in real-world scenarios.
- The importance of PR-AUC over standard accuracy metrics.
- Building a full-stack ML product, from Python backend to React frontend.

---

## Author

Built by **[Sonia Thakur]**

- GitHub: [Sonia068](https://github.com/Sonia068)
- LinkedIn: [Sonia Thakur](https://www.linkedin.com/in/sonia-thakur-6ab93b349/)

---

## License

MIT License
