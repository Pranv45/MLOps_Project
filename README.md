**IPL Win Predictor - Software Design Document**

---

### 1. Overview

This document describes the high-level and low-level design of the IPL Win Predictor application. The project predicts the probability of a batting team winning an IPL match given live match conditions. It is built using FastAPI for the backend and an HTML/CSS/JS frontend, with a machine learning model trained on IPL datasets. Apache Airflow is used to orchestrate the data pipeline.

---

### 2. Architecture Diagram

```
┌────────────────────────┐       REST API       ┌────────────────────────────┐
│  Frontend (HTML/JS)   │ <------------------> │   FastAPI Backend (app.py) │
└────────────────────────┘                      └────────────────────────────┘
                                                         │
                                                         ▼
                                               ┌───────────────────┐
                                               │ Trained ML Model  │
                                               │ (lr_model.pkl)    │
                                               └───────────────────┘
                                                         │
                                                         ▼
                                               ┌───────────────────┐
                                               │   Logging System   │
                                               │ (app_logs.log)     │
                                               └───────────────────┘
                                                         ▲
                                                         │
                                               ┌───────────────────┐
                                               │  Airflow DAG       │
                                               │ (ETL Pipeline)     │
                                               └───────────────────┘
```

---

### 3. High-Level Design (HLD)

**Modules:**

- Frontend UI: Collects match inputs and sends REST API request to backend.
- Backend API: Accepts JSON data and returns win probability or match result.
- Model Inference: Predicts win probability using a trained Logistic Regression model.
- Logging: Logs API request, prediction results, and exceptions.
- Orchestration: Apache Airflow manages the data pipeline including ingestion, preprocessing, and model training.

**Tech Stack:**

- Frontend: HTML, CSS, JS
- Backend: Python (FastAPI)
- Model: Trained with scikit-learn
- MLOps Tools: MLflow (tracking), Airflow (orchestration)

---

### 4. Low-Level Design (LLD)

#### 4.1 API Specification

**Endpoint:** `/predict`

- **Method:** POST
- **Input JSON Schema:**

```json
{
  "batting_team": "string",
  "bowling_team": "string",
  "city": "string",
  "match_type": "string",
  "toss_winner": "string",
  "total_runs_x": int,
  "required_runs": int,
  "balls_left": int,
  "wickets_left": int
}
```

- **Output (success):**

```json
{
  "message": "Win probability of <Team> is <percent>%"
}
```

- **Output (error):**

```json
{
  "error": "Validation or server-side error message."
}
```

---

### 5. Design Principles Followed

- **Loose Coupling**: Frontend and backend communicate via REST API only.
- **Separation of Concerns**: UI logic is separate from model logic.
- **Functional Design**: Backend structured around functions instead of classes.
- **Logging and Exception Handling**: Implemented in app.py using Python's `logging` module.
- **Pipeline Modularity**: Airflow DAG encapsulates ingestion, preprocessing, and training as distinct tasks.

---

### 6. Data Ingestion

- Used airflow for data ingestion and orchestration

### 7. Experiment tracking
- Used MLflow for experiment tracking and used MLproject

