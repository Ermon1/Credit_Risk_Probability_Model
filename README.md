# Credit Risk Probability Model (Alternative Data)

## Overview

This project implements an end-to-end **credit risk modeling system using alternative transactional data** from an eCommerce platform. The objective is to estimate customer credit risk in the absence of traditional credit bureau information and explicit default labels.

The system transforms raw transaction-level data into behavioral risk signals, trains probabilistic risk models, and exposes predictions through a containerized API. The solution is designed with **reproducibility, interpretability, and deployment readiness** in mind, reflecting real-world financial and regulatory constraints.

---

## Business Context

Bati Bank is partnering with an eCommerce provider to offer a **Buy-Now-Pay-Later (BNPL)** service. Since customers may not have formal credit histories, lending decisions must rely on **behavioral transaction data** rather than traditional credit scores.

The model outputs:
- A **risk probability score** per customer  
- A **credit score** derived from that probability  
- Inputs for **loan approval, amount, and duration** decisions  

---

## Data

The dataset consists of transactional records including:
- Customer and account identifiers
- Transaction amounts and timestamps
- Product, channel, and provider metadata
- Fraud indicators

These variables are used to derive **customer-level behavioral features** such as transaction frequency, monetary value, and recency.

---

## System Design

High-level pipeline:
1. Raw transaction ingestion  
2. Feature engineering and aggregation at customer level  
3. Proxy target construction using behavioral patterns  
4. Supervised model training and evaluation  
5. Model tracking and versioning with MLflow  
6. API-based inference using FastAPI  
7. CI/CD enforcement and containerized deployment  

---

## Credit Scoring Business Understanding (Task 1)

### Basel II & Risk Measurement
Under Basel II, credit risk must be quantified, auditable, and interpretable. Our model outputs a probability of being high-risk, which serves as a proxy for the Probability of Default (PD) for internal risk scoring.

### Proxy Target Variable
We lack a true default label. To construct a supervised model, we define `is_high_risk` using RFM (Recency, Frequency, Monetary) metrics:
- High-risk customers: low engagement, low frequency, low total spend
- Low-risk customers: active, frequent, high spend

**Risk:** Label noise and bias may arise; assumptions are documented for audit purposes.

### Model Trade-offs
- **Interpretable models (Logistic Regression + WoE):** Easy to explain to regulators, stable, fast to train.
- **Complex models (Gradient Boosting, Random Forest):** Higher predictive power, but require explainability tools (SHAP/LIME) and stronger monitoring.

---

## Limitations

- Risk labels are proxy-based, not true defaults  
- Behavioral engagement does not perfectly represent creditworthiness  
- Model outputs require ongoing validation once real repayment data becomes available  

---

## Tech Stack

- Python, pandas, scikit-learn  
- MLflow for experiment tracking and model registry  
- FastAPI for model serving  
- Docker & Docker Compose  
- Pytest and GitHub Actions for CI/CD  

---

## Repository Structure

