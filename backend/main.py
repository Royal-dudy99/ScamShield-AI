import os
import re
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, List, Any
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODELS_DIR, 'scam_detector_model.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'scam_detector_vectorizer.pkl')

# App state
model = None
vectorizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vectorizer
    try:
        if not os.path.isfile(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        if not os.path.isfile(VECTORIZER_PATH):
            raise FileNotFoundError(f"Vectorizer file not found at: {VECTORIZER_PATH}")
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        model = None
        vectorizer = None
    yield
    # Cleanup if needed

app = FastAPI(
    title="ScamShield AI API",
    description="AI-Based Detection of Scam, Spam, and Phishing Messages",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
# Supports local development across common dev server ports and production URL via FRONTEND_URL env var
DEFAULT_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5500",
    "http://localhost:5173",
    "http://localhost:8080",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8080",
    "null",  # Allow file:// protocol during local testing
]

# Allow custom production URLs via FRONTEND_URL environment variable (single URL or comma-separated)
allowed_origins = list(DEFAULT_ORIGINS)
env_frontend_url = os.getenv("FRONTEND_URL", "").strip()
if env_frontend_url:
    for url in env_frontend_url.split(","):
        cleaned_url = url.strip().rstrip("/")
        if cleaned_url and cleaned_url not in allowed_origins:
            allowed_origins.append(cleaned_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

LABEL_MAP = {
    "ham": {"label": "LEGITIMATE MESSAGE", "risk": "Low Risk", "risk_level": "low"},
    "spam": {"label": "SPAM MESSAGE", "risk": "Medium Risk", "risk_level": "medium"},
    "smishing": {"label": "POTENTIAL SMISHING / PHISHING", "risk": "High Risk", "risk_level": "high"}
}

def detect_signals(message: str) -> List[Dict[str, str]]:
    signals = []
    msg_lower = message.lower()
    
    # URLs
    if re.search(r'(https?://|www\.)[^\s]+', msg_lower):
        signals.append({"signal": "Suspicious URL Detected", "description": "Contains a web link which could lead to a phishing or malicious site."})

    # Phone numbers / shortcodes
    if re.search(r'(\b(?:call|txt|text|contact|phone)\s*(?:at|on|to)?\s*[:\s]?\+?\d[\d\s-]{4,}\b|\b\d{5,6}\b|\+?\d{1,3}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b0\d{10}\b)', msg_lower):
        signals.append({"signal": "Contact Number / Shortcode Detected", "description": "Contains a phone number or SMS shortcode for direct response."})
        
    # Urgency words
    urgency_words = ['immediately', 'urgent', 'hurry', 'fast', 'quickly', 'now', 'asap', 'expire', 'deadline']
    if any(word in msg_lower for word in urgency_words):
        signals.append({"signal": "Urgency Language", "description": "Uses words that create a sense of urgency."})
        
    # Account verification
    verification_words = ['verify', 'verification', 'confirm', 'validate', 'authenticate']
    if any(word in msg_lower for word in verification_words):
        signals.append({"signal": "Account Verification Language", "description": "Requests to verify or confirm account details."})
        
    # KYC terms
    kyc_words = ['kyc', 'know your customer']
    if any(word in msg_lower for word in kyc_words):
        signals.append({"signal": "KYC Terminology", "description": "Mentions KYC (Know Your Customer) processes."})
        
    # Banking terms
    banking_words = ['bank', 'account', 'credit card', 'debit card', 'transaction', 'transfer']
    if any(word in msg_lower for word in banking_words):
        signals.append({"signal": "Banking Terminology", "description": "Mentions banking or financial terms."})
        
    # Prize/reward
    prize_words = ['congratulations', 'winner', 'won', 'prize', 'reward', 'lucky', 'jackpot', 'lottery']
    if any(word in msg_lower for word in prize_words):
        signals.append({"signal": "Prize / Reward Language", "description": "Claims you have won a prize or reward."})
        
    # Payment request
    payment_words = ['payment', 'pay', 'send money', 'transfer funds']
    if any(word in msg_lower for word in payment_words):
        signals.append({"signal": "Payment Request", "description": "Requests payment or money transfer."})
        
    # OTP/password
    otp_words = ['otp', 'password', 'pin', 'passcode', 'security code', 'cvv']
    if any(word in msg_lower for word in otp_words):
        signals.append({"signal": "OTP / Password Request", "description": "Requests sensitive security codes or passwords."})
        
    # Account threat
    threat_words = ['blocked', 'suspended', 'deactivated', 'terminated', 'locked', 'restricted']
    if any(word in msg_lower for word in threat_words):
        signals.append({"signal": "Account Suspension Threat", "description": "Threatens account suspension or restriction."})
        
    # Financial request
    financial_words = ['rupees', 'dollars', 'amount', 'credit', 'loan', 'emi', 'installment']
    if any(word in msg_lower for word in financial_words):
        signals.append({"signal": "Financial Request", "description": "Mentions money, loans, or credit."})
        
    return signals


def evaluate_hybrid_risk(
    ml_pred: str,
    probabilities: Optional[Dict[str, float]],
    signals: List[Dict[str, str]]
) -> Dict[str, str]:
    """
    Hybrid Risk Decision Layer combining ML predictions and rule-based heuristic signals.
    Returns a dict with 'label', 'risk', and 'risk_level' ('high', 'medium', 'low').
    """
    probs = probabilities or {}
    smishing_prob = probs.get("smishing", 0.0)
    spam_prob = probs.get("spam", 0.0)
    ham_prob = probs.get("ham", 0.0)

    signal_names = {s.get("signal") for s in signals}

    has_url = "Suspicious URL Detected" in signal_names
    has_contact = "Contact Number / Shortcode Detected" in signal_names
    has_urgency = "Urgency Language" in signal_names
    has_verification = "Account Verification Language" in signal_names
    has_kyc = "KYC Terminology" in signal_names
    has_banking = "Banking Terminology" in signal_names
    has_prize = "Prize / Reward Language" in signal_names
    has_payment = "Payment Request" in signal_names
    has_otp = "OTP / Password Request" in signal_names
    has_threat = "Account Suspension Threat" in signal_names
    has_financial = "Financial Request" in signal_names

    # 1. HIGH RISK (POTENTIAL SMISHING / PHISHING)
    # Triggered by high ML smishing confidence OR strong phishing indicator combinations
    is_strong_phishing_combo = (
        (has_url and has_prize) or
        (has_url and (has_threat or has_banking or has_kyc or (has_verification and has_urgency))) or
        (has_url and has_otp) or
        (has_otp and (has_threat or has_banking or has_kyc or has_verification)) or
        (has_url and (has_payment or has_financial) and (has_urgency or has_threat or has_prize or smishing_prob >= 0.40))
    )

    if (
        smishing_prob >= 0.70 or
        (ml_pred == "smishing" and smishing_prob >= 0.50) or
        is_strong_phishing_combo
    ):
        return {
            "label": "POTENTIAL SMISHING / PHISHING",
            "risk": "High Risk",
            "risk_level": "high"
        }

    # 2. SPAM PRESERVATION
    # Maintain spam classification when predicted by ML without strong phishing indicators
    if ml_pred == "spam":
        if spam_prob >= 0.85:
            return {"label": "SPAM MESSAGE", "risk": "High Risk", "risk_level": "high"}
        else:
            return {"label": "SPAM MESSAGE", "risk": "Medium Risk", "risk_level": "medium"}

    # 3. MEDIUM RISK (SUSPICIOUS MESSAGE)
    # ML predicted ham, but suspicious signals exist or elevated threat probability
    has_meaningful_signals = (
        has_url or
        has_financial or
        has_payment or
        has_banking or
        has_kyc or
        has_verification or
        has_threat or
        has_prize or
        has_otp or
        (has_contact and (has_urgency or has_financial or has_prize)) or
        smishing_prob >= 0.30 or
        spam_prob >= 0.45
    )

    if has_meaningful_signals:
        return {
            "label": "SUSPICIOUS MESSAGE",
            "risk": "Medium Risk",
            "risk_level": "medium"
        }

    # 4. LOW RISK (LEGITIMATE MESSAGE)
    # Legitimate ham message without suspicious signals
    return {
        "label": "LEGITIMATE MESSAGE",
        "risk": "Low Risk",
        "risk_level": "low"
    }


class MessageRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, description="Message text to analyze")

    @field_validator('message')
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message cannot be empty or contain only whitespace.")
        return v

class PredictionResponse(BaseModel):
    prediction: str
    label: str
    risk: str
    risk_level: str
    probabilities: Optional[Dict[str, float]] = None
    signals: List[Dict[str, str]] = []
    message_length: int
    disclaimer: str = "Model probabilities describe the classifier's prediction distribution and are not a guarantee of real-world safety."

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    clean_errors = []
    for err in exc.errors():
        clean_errors.append({
            "type": err.get("type"),
            "loc": [str(x) for x in err.get("loc", [])],
            "msg": str(err.get("msg")),
        })
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Invalid input data: Message cannot be empty, whitespace-only, or malformed.", "errors": clean_errors}
    )

@app.exception_handler(404)
async def custom_404_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_404_NOT_FOUND,
        content={"detail": "Resource not found"}
    )

@app.get("/health")
def health_check():
    is_ready = (model is not None and vectorizer is not None)
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: MessageRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Machine learning models are not loaded on server.")
        
    msg = request.message.strip()
    if not msg:
        raise HTTPException(status_code=422, detail="Message cannot be empty or contain only whitespace.")
        
    try:
        vec_msg = vectorizer.transform([msg])
        pred = model.predict(vec_msg)[0]
        
        probabilities = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(vec_msg)[0]
            # Safely map to class labels from model.classes_
            probabilities = {str(cls): round(float(prob), 4) for cls, prob in zip(model.classes_, proba)}
            
        signals = detect_signals(msg)
        decision = evaluate_hybrid_risk(pred, probabilities, signals)
        
        return PredictionResponse(
            prediction=pred,
            label=decision["label"],
            risk=decision["risk"],
            risk_level=decision["risk_level"],
            probabilities=probabilities,
            signals=signals,
            message_length=len(msg)
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return JSONResponse(status_code=500, content={"detail": "Internal server error during prediction"})

@app.get("/model-info")
def model_info():
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model or vectorizer not loaded.")
        
    classes_list = [str(c) for c in model.classes_] if hasattr(model, 'classes_') else ["ham", "smishing", "spam"]
    return {
        "model_type": "Logistic Regression",
        "feature_extraction": "Character-level TF-IDF",
        "classes": classes_list,
        "accuracy": 0.9714285714285714,
        "macro_f1": 0.91,
        "test_set_size": 1190,
        "dataset_size": 5947,
        "class_distribution": {"ham": 4834, "smishing": 626, "spam": 487}
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)

