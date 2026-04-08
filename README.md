# 📱 Telco Churn Prediction System

An end-to-end machine learning system that predicts which telecom customers
are likely to churn — giving retention teams time to act before customers leave.

> **Status**: Core system complete — AWS deployment in progress

---

## 🎯 What This System Does

Given a customer's account information and service usage, the system predicts
whether they will churn and assigns a risk level so retention teams know
exactly who to contact and how urgently.

**Example API response:**
```json
{
  "churn": true,
  "churn_probability": 0.7826,
  "threshold_used": 0.2,
  "risk_level": "HIGH"
}
```

---

## 💼 Problem Solved & Business Value

- **Early warning**: Identifies churners before they leave — not after
- **Prioritized action**: HIGH / MEDIUM / LOW risk levels tell teams
  exactly who needs attention first
- **Operationalized ML**: Predictions accessible via REST API — no
  notebook needed, anyone can query it
- **Traceable experiments**: MLflow logs every training run with
  parameters, metrics, and model artifacts
- **Repeatable delivery**: One command retrains everything from scratch

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Model | XGBoost Classifier |
| Recall | 97.3% |
| Precision | 38.5% |
| F1 Score | 55.1% |
| Threshold | 0.2 (tuned for maximum recall) |
| Tuning | Optuna (30 trials) |

**Why recall over accuracy?**
In churn prediction, missing a churner (false negative) costs far more
than a false alarm (false positive). A customer you incorrectly flag as
high-risk costs one retention call. A churner you miss costs their entire
lifetime revenue. The model is deliberately tuned to catch as many
churners as possible — 97.3% recall means only 2.7% of churners slip through.

---

## 🏗️ Architecture
Raw Data (CSV)
│
▼
┌─────────────────────┐
│    Data Pipeline     │  load → preprocess → build features
└─────────────────────┘
│
▼
┌─────────────────────┐
│    ML Pipeline       │  tune (Optuna) → train → evaluate
│    + MLflow          │  every run logged automatically
└─────────────────────┘
│
▼
┌─────────────────────┐
│    REST API          │  FastAPI + Pydantic validation
│    + Risk Level      │  HIGH / MEDIUM / LOW recommendation
└─────────────────────┘
│
▼
┌─────────────────────┐
│    Docker            │  containerized, runs anywhere
└─────────────────────┘
│
▼
┌─────────────────────┐
│    AWS ECS           │  live deployment (in progress)
└─────────────────────┘

---

## ⚙️ Feature Engineering

| Type | Features | Reason |
|------|----------|--------|
| Binary encoding | gender, Partner, Dependents, PhoneService, PaperlessBilling | 2-category features → 0/1 |
| One-hot encoding | Contract, InternetService, PaymentMethod + 7 more | Multi-category → dummy columns |
| Numeric | tenure, MonthlyCharges, TotalCharges | Direct model input |

**Key design decision**: Binary encoding uses deterministic sorted mapping
so inference always produces the same encoding as training — no leakage,
no surprises.

---

## 🗂️ Project Structure
telco-churn-prediction/
├── scripts/
│   └── run_pipeline.py          # one command runs everything
├── src/
│   ├── data/
│   │   ├── load_data.py         # loads and validates raw CSV
│   │   └── preprocess.py        # cleans data, fixes types
│   ├── features/
│   │   └── build_features.py    # binary + one-hot encoding
│   ├── models/
│   │   ├── train.py             # XGBoost + MLflow tracking
│   │   ├── evaluate.py          # classification report + recall
│   │   └── tune.py              # Optuna hyperparameter search
│   ├── serving/
│   │   └── inference.py         # loads artifact, makes predictions
│   └── app/
│       ├── main.py              # FastAPI endpoints + risk levels
│       └── app.py               # uvicorn entry point
├── Dockerfile
├── requirements.txt
└── .github/workflows/           # CI/CD pipeline

---

## 🚀 Run Locally

```bash
git clone https://github.com/TalhaTahir250/telco-churn-prediction.git
cd telco-churn-prediction
pip install -r requirements.txt

# Add your Telco CSV to src/data/
# Dataset: IBM Telco Customer Churn (Kaggle)

# Run full pipeline (tune + train + evaluate)
python -m scripts.run_pipeline --file "src/data/WA_Fn-UseC_-Telco-Customer-Churn.csv" --tune

# Start API
python src/app/app.py
```

Visit `http://localhost:8000/docs` for interactive API documentation.

---

## 🐳 Run with Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | / | Service info |
| GET | /health | Health check |
| POST | /predict | Get churn prediction + risk level |

**Example request:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "Total_Charges": 845.50
  }'
```

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.11 |
| ML | XGBoost, Scikit-learn, Optuna |
| Experiment Tracking | MLflow |
| API | FastAPI, Pydantic, Uvicorn |
| Data | Pandas, NumPy |
| Container | Docker |
| CI/CD | GitHub Actions |
| Deployment | AWS ECS Fargate (in progress) |
| Monitoring | AWS CloudWatch (in progress) |

---

## 🔍 Roadblocks & How I Solved Them

**Inference column mismatch**
- Cause: `build_features` uses `nunique()` to detect binary vs
  multi-category columns — on a single inference row, every column
  has exactly 1 unique value, breaking the detection logic entirely.
- Fix: At inference, skip `build_features` entirely. Instead apply
  `pd.get_dummies` directly then `reindex` to match training columns
  exactly using the saved `feature_cols` list from the artifact.
  Missing columns fill with 0, extra columns drop automatically.

**Custom threshold not applied at inference**
- Cause: Artifact saved model and threshold separately but inference
  used default 0.5 instead of the Optuna-tuned 0.2 threshold.
- Fix: Saved model, threshold, and feature_cols together in one
  artifact dictionary. Inference loads all three and applies the
  correct threshold every time.

**Class imbalance (73% non-churn, 27% churn)**
- Cause: XGBoost trained on imbalanced data predicts majority class
  by default — catches very few churners.
- Fix: `scale_pos_weight = negative_count / positive_count` passed
  to XGBoost automatically during training. Combined with threshold
  tuning to 0.2, recall jumped to 97.3%.

---

## 📈 Roadmap

- [x] Data pipeline
- [x] Feature engineering
- [x] Hyperparameter tuning with Optuna
- [x] MLflow experiment tracking
- [x] REST API with FastAPI
- [x] Docker containerization
- [x] GitHub repository
- [x] GitHub Actions CI/CD
- [ ] AWS ECS deployment
- [ ] Prediction logging to database
- [ ] Live monitoring dashboard
- [ ] Batch scoring pipeline

---

## 👤 Author

**Talha Tahir**
- GitHub: [@TalhaTahir250](https://github.com/TalhaTahir250)
- LinkedIn: [muhammad-talha-tahir](https://linkedin.com/in/muhammad-talha-tahir)

---

## 📄 Dataset

IBM Telco Customer Churn Dataset — 7,043 customers with 20 features
including demographics, account information, and services subscribed.
Available on Kaggle.