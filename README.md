# Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Next.js](https://img.shields.io/badge/Next.js-15.0+-black)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

An end-to-end Machine Learning system that detects fraudulent credit card transactions in near real-time. It utilizes cost-sensitive learning techniques for imbalanced classification, exposes a scoring API (batch + streaming hook), and visualizes precision-recall trade-offs, fraud alerts, and feature impacts in a highly responsive Next.js dashboard.

---

## 1. Project Explanation


# 💳 Credit Card Fraud Detection System

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


