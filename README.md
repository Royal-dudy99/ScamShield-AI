# ScamShield AI

## AI-Based Detection of Scam and Phishing Messages Using Machine Learning

ScamShield AI is a full-stack web application that classifies SMS, WhatsApp, and email messages as **Legitimate**, **Spam**, or **Potential Scam/Phishing** using a trained machine learning model. Built as a B.Tech college project, it demonstrates practical NLP and ML techniques for cybersecurity applications.

---

## Problem Statement

With the rise of digital communication, users are increasingly targeted by scam and phishing messages through SMS, WhatsApp, and email. These messages attempt to steal personal information, financial data, or install malware. Existing spam filters are often insufficient against sophisticated phishing attacks. There is a need for an intelligent system that can detect and classify such messages to alert users before they become victims.

## Proposed Solution

ScamShield AI uses a **Character-level TF-IDF + Logistic Regression** pipeline to classify messages into three categories:

| Class | Label | Risk Level |
|-------|-------|------------|
| `ham` | Legitimate Message | Low |
| `spam` | Spam Message | Medium |
| `smishing` | Potential Scam / Phishing | High |

The system provides real-time classification through a web interface, along with model probabilities and rule-based signal detection.

---

## Features

- **Real-time Message Classification** — Paste any message and get instant prediction
- **Three-class Classification** — Ham (Legitimate), Spam, Smishing (Phishing)
- **Model Probability Display** — Shows classifier confidence distribution across all classes
- **Rule-based Signal Detection** — Identifies suspicious patterns (URLs, urgency language, KYC terms, etc.)
- **Safety Advice** — Provides actionable safety tips for flagged messages
- **Responsive Design** — Works on desktop, tablet, and mobile
- **Privacy-first** — No messages are stored or sent to external services
- **REST API** — Clean FastAPI backend with health checks and model info endpoints

---

## Architecture

```
USER
  ↓
FRONTEND (HTML/CSS/JavaScript)
  ↓
HTTP Request (POST /predict)
  ↓
FASTAPI BACKEND (Python)
  ↓
Saved TF-IDF Vectorizer (character-level)
  ↓
Saved Logistic Regression Model
  ↓
Prediction Result (ham / spam / smishing)
  ↓
JSON Response
  ↓
FRONTEND
  ↓
Result Display (label, risk, probabilities, signals)
```

---

## ML Methodology

### Feature Extraction

**Character-level TF-IDF Vectorization**

| Parameter | Value |
|-----------|-------|
| `analyzer` | `char` |
| `ngram_range` | `(3, 5)` |
| `min_df` | `2` |
| `max_features` | `20000` |

Character-level n-grams capture sub-word patterns, misspellings, and obfuscation techniques commonly used in phishing messages, making them more robust than word-level features for this domain.

### Classifier

**Logistic Regression**

| Parameter | Value |
|-----------|-------|
| `C` | `1.0` |
| `class_weight` | `balanced` |
| `max_iter` | `1000` |
| `random_state` | `42` |

The `balanced` class weight compensates for class imbalance in the training data.

---

## Dataset Information

| Metric | Value |
|--------|-------|
| Total Messages | 5,947 |
| Ham (Legitimate) | 4,834 |
| Smishing (Phishing) | 626 |
| Spam | 487 |

The dataset combines SMS spam collection data with additional smishing samples to create a three-class classification problem.

---

## Model Comparison

During development, several algorithms and feature configurations were tested:

| # | Feature Extraction | Classifier | Accuracy | Macro F1 |
|---|-------------------|------------|----------|----------|
| 1 | **Character TF-IDF** | **Logistic Regression** | **97.14%** | **0.91** |
| 2 | Character TF-IDF | SVM | 96.72% | 0.90 |
| 3 | Character TF-IDF | SGDClassifier | 96.72% | 0.90 |
| 4 | Character TF-IDF | Random Forest | 96.39% | 0.89 |
| 5 | Character TF-IDF | Naive Bayes | 94.79% | 0.84 |
| 6 | Combined TF-IDF | Logistic Regression | 96.64% | 0.89 |
| 7 | Word TF-IDF | Logistic Regression | 95.55% | 0.87 |
| 8 | Word TF-IDF | Random Forest | 95.38% | 0.87 |
| 9 | Word TF-IDF | SVM | 95.55% | 0.86 |
| 10 | Word TF-IDF | Naive Bayes | 93.45% | 0.81 |

**Selected Model: Character TF-IDF + Logistic Regression** (highest accuracy and Macro F1)

---

## Final Model Evaluation

Evaluated on a held-out test set of **1,190 messages**.

### Overall Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 0.9714285714285714 (97.14%) |
| Macro Average F1 | 0.91 |
| Weighted Average F1 | 0.97 |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| `ham` | 0.99 | 1.00 | 0.99 | 967 |
| `smishing` | 0.93 | 0.87 | 0.90 | 125 |
| `spam` | 0.84 | 0.83 | 0.84 | 98 |
| **Macro Average** | **0.92** | **0.90** | **0.91** | **1,190** |
| **Weighted Average** | **0.97** | **0.97** | **0.97** | **1,190** |

### Confusion Matrix

|  | Predicted Ham | Predicted Smishing | Predicted Spam |
|--|--------------|-------------------|---------------|
| **Actual Ham** | 966 | 1 | 0 |
| **Actual Smishing** | 1 | 109 | 15 |
| **Actual Spam** | 10 | 7 | 81 |

---

## Project Structure

```
Scam_Detection/
├── backend/
│   ├── main.py                        # FastAPI application & ML pipeline
│   └── requirements.txt               # Backend dependencies
├── frontend/
│   ├── index.html                     # Single-page web application
│   └── config.js                      # Client configuration (API_BASE_URL)
├── models/
│   ├── scam_detector_model.pkl        # Trained Logistic Regression model
│   └── scam_detector_vectorizer.pkl   # Fitted Character TF-IDF vectorizer
├── tests/
│   └── test_api.py                    # Automated test suite (12 tests)
├── Datasets/
│   ├── Dataset_5971.csv               # Development dataset
│   └── sms+spam+collection/           # SMS Spam Collection data
├── README.md                          # Project documentation
└── .gitignore                         # Version control exclusions
```

---

## Local Development

### Prerequisites

- Python 3.9+
- pip (Python package manager)
- Modern web browser

### 1. Backend Setup & Startup

```bash
# Navigate to the project root
cd Scam_Detection

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS

# Install dependencies
pip install -r backend/requirements.txt

# Start backend development server
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at `http://127.0.0.1:8000`.

### 2. Frontend Setup & Startup

Serve the static frontend using Python's built-in HTTP server:

```bash
cd Scam_Detection/frontend
python -m http.server 5500
```

Then visit `http://localhost:5500` in your web browser.

---

## Production Deployment Architecture

```
User Web Browser
       │
       ▼
Vercel (Static Frontend: HTML / CSS / JS / config.js)
       │
       │ HTTP REST API (POST /predict, GET /health)
       ▼
Render (Web Service: FastAPI + Uvicorn + Pre-loaded ML Models)
       │
       └── GitHub Repository (Source Code + Model Artifacts)
```

### Backend Deployment (Render)

1. Create a new **Web Service** on [Render](https://render.com) and connect your GitHub repository.
2. Configure the service settings:
   - **Environment:** `Python 3`
   - **Root Directory:** (leave blank for repository root)
   - **Build Command:**
     ```bash
     pip install -r backend/requirements.txt
     ```
   - **Start Command:**
     ```bash
     uvicorn backend.main:app --host 0.0.0.0 --port $PORT
     ```
3. Set Environment Variables on Render:
   - `FRONTEND_URL`: `https://your-app-name.vercel.app` *(or comma-separated allowed domains)*
4. Set the Health Check Path:
   - **Health Check Path:** `/health`
5. Deploy. Once active, note your public service URL (e.g., `https://scamshield-backend.onrender.com`).

### Frontend Deployment (Vercel)

1. Import your GitHub repository on [Vercel](https://vercel.com).
2. Configure project settings:
   - **Framework Preset:** `Other` (Static)
   - **Root Directory:** `frontend`
3. Point the frontend to the deployed backend:
   - In `frontend/config.js`, set `API_BASE_URL` to your Render service URL:
     ```javascript
     window.APP_CONFIG = {
         API_BASE_URL: "https://your-service-name.onrender.com"
     };
     ```
4. Deploy on Vercel. The frontend will now communicate directly with your Render API over HTTPS.

---

## API Endpoints

### Health Check

```
GET /health
```

Response:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "vectorizer_loaded": true
}
```

### Predict Message

```
POST /predict
Content-Type: application/json

{
    "message": "Your account will be blocked today. Verify your KYC immediately."
}
```

Response:
```json
{
    "prediction": "smishing",
    "label": "POTENTIAL SMISHING / PHISHING",
    "risk": "High Risk",
    "risk_level": "high",
    "probabilities": {
        "ham": 0.3276,
        "smishing": 0.6000,
        "spam": 0.0724
    },
    "signals": [
        {"signal": "Urgency Language", "description": "Uses words that create a sense of urgency."},
        {"signal": "Account Verification Language", "description": "Requests to verify or confirm account details."},
        {"signal": "KYC Terminology", "description": "Mentions KYC (Know Your Customer) processes."},
        {"signal": "Banking Terminology", "description": "Mentions banking or financial terms."},
        {"signal": "Account Suspension Threat", "description": "Threatens account suspension or restriction."}
    ],
    "message_length": 65,
    "disclaimer": "Model probabilities describe the classifier's prediction distribution and are not a guarantee of real-world safety."
}
```

### Model Metadata

```
GET /model-info
```

Returns model metadata including algorithm, feature extraction specifications, dataset distributions, and benchmark accuracy.

---

## Automated Testing

An automated test suite is provided in `tests/test_api.py`. It executes 12 end-to-end tests against the live model pipeline:

```bash
python tests/test_api.py
```

### Representative Test Cases

| Test | Category | Message | Expected Prediction |
|------|----------|---------|-------------------|
| 1 | Ham | `"Hey bro, are we meeting tomorrow?"` | `ham` (Low Risk) |
| 2 | Ham | `"Can you send me the notes from today's lecture?"` | `ham` (Low Risk) |
| 3 | Smishing | `"Congratulations! You have won a cash prize. Click this link immediately to claim."` | `smishing` (High Risk) |
| 4 | Smishing | `"Your account will be blocked today. Verify your KYC immediately."` | `smishing` (High Risk) |
| 5 | Spam | `"Free entry in 2 a weekly competition to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C apply"` | `spam` (Medium/High Risk) |
| 6 | Validation | `""` or `"   "` (empty/whitespace) | HTTP 422 Error |
| 7 | Validation | Length > 5000 characters | HTTP 422 Error |
| 8 | Heuristics | `"Verify at https://secure-bank-login.phishingsite.com"` | Suspicious URL Detected |
| 9 | Heuristics | `"Call 09061701461 now or text 87121"` | Contact Number / Shortcode Detected |
| 10 | Heuristics | `"Urgent: Complete your bank account KYC verification"` | KYC & Banking Terminology |

---

## Limitations

- The model is trained on a specific dataset and may not generalize to all real-world messaging patterns.
- Real-world scam patterns continuously evolve; the model may not detect novel attack vectors.
- The dataset is primarily in English; performance on other languages or Hinglish may be limited.
- Performance metrics are based on the test set and do not guarantee real-world performance.
- False positives (legitimate messages flagged as scams) and false negatives (scams missed) are possible.
- **This application should be treated as a decision-support tool, not a guaranteed security system.**
- The rule-based signal detection is heuristic and may produce false indicators.

## Future Scope

- **Multilingual Support** — Support for Hindi, Hinglish, and other regional languages
- **Larger Datasets** — Training on more diverse and recent datasets
- **Transformer Models** — Using BERT, RoBERTa, or similar models for improved accuracy
- **URL Reputation Analysis** — Real-time verification of URLs against threat databases
- **Domain Verification** — Checking sender domains for legitimacy
- **Email Integration** — Direct integration with email clients
- **Browser Extension** — Real-time protection while browsing
- **Mobile Application** — Native Android/iOS app for on-device protection
- **Real-time Messaging Protection** — Integration with messaging apps
- **Model Explainability** — LIME/SHAP-based explanations for predictions
- **Continuous Learning** — Periodic model updates with new data
- **Adversarial Robustness** — Defending against adversarial message crafting

---

## Technologies Used

| Component | Technology |
|-----------|-----------|
| Backend | Python, FastAPI, Uvicorn |
| ML Framework | Scikit-learn |
| Feature Extraction | TF-IDF (character-level) |
| Classifier | Logistic Regression |
| Model Serialization | joblib |
| Frontend | HTML5, CSS3, JavaScript (ES6+) |
| API Format | REST (JSON) |

---

## License

This project is developed for academic and educational purposes.

---

*ScamShield AI — Detect suspicious messages before they deceive you.*
