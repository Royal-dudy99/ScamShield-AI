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
            
        mapped = LABEL_MAP.get(pred, LABEL_MAP["ham"]).copy()
        
        # Dynamic risk assessment for spam based on probability
        if pred == "spam" and probabilities:
            spam_prob = probabilities.get("spam", 0.0)
            if spam_prob >= 0.85:
                mapped["risk"] = "High Risk"
                mapped["risk_level"] = "high"
            else:
                mapped["risk"] = "Medium Risk"
                mapped["risk_level"] = "medium"

        signals = detect_signals(msg)
        
        return PredictionResponse(
            prediction=pred,
            label=mapped["label"],
            risk=mapped["risk"],
            risk_level=mapped["risk_level"],
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

