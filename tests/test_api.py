"""
Automated Test Suite for ScamShield AI API
Tests the FastAPI backend against the actual saved ML model and TF-IDF vectorizer.
"""

import sys
import os
import unittest
from fastapi.testclient import TestClient

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from backend.main import app


class TestScamShieldAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Initialize FastAPI TestClient with application lifespan."""
        cls.client_cm = TestClient(app)
        cls.client = cls.client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        """Clean up TestClient context."""
        cls.client_cm.__exit__(None, None, None)

    def test_01_health_endpoint(self):
        """Test GET /health verifies the system is healthy and models are loaded."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertTrue(data.get("model_loaded"))
        self.assertTrue(data.get("vectorizer_loaded"))

    def test_02_model_info_endpoint(self):
        """Test GET /model-info returns accurate model and dataset metadata."""
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("model_type"), "Logistic Regression")
        self.assertEqual(data.get("feature_extraction"), "Character-level TF-IDF")
        self.assertIn("ham", data.get("classes"))
        self.assertIn("smishing", data.get("classes"))
        self.assertIn("spam", data.get("classes"))
        self.assertAlmostEqual(data.get("accuracy"), 0.9714, places=3)
        self.assertEqual(data.get("macro_f1"), 0.91)
        self.assertEqual(data.get("dataset_size"), 5947)

    def test_03_normal_ham_message(self):
        """Test prediction for typical legitimate messages."""
        msg = "Hey bro, are we meeting tomorrow?"
        response = self.client.post("/predict", json={"message": msg})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("prediction"), "ham")
        self.assertEqual(data.get("label"), "LEGITIMATE MESSAGE")
        self.assertEqual(data.get("risk"), "Low Risk")
        self.assertGreater(data.get("probabilities", {}).get("ham", 0.0), 0.5)

        # Secondary ham sample
        msg2 = "Can you send me the notes from today's lecture?"
        response2 = self.client.post("/predict", json={"message": msg2})
        self.assertEqual(response2.status_code, 200)
        self.assertEqual(response2.json().get("prediction"), "ham")

    def test_04_clear_smishing_message(self):
        """Test prediction for typical smishing / phishing scam messages."""
        msg = "Congratulations! You have won a cash prize. Click this link immediately to claim."
        response = self.client.post("/predict", json={"message": msg})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("prediction"), "smishing")
        self.assertEqual(data.get("label"), "POTENTIAL SMISHING / PHISHING")
        self.assertEqual(data.get("risk"), "High Risk")
        self.assertGreater(data.get("probabilities", {}).get("smishing", 0.0), 0.5)

        # KYC urgency threat sample
        msg2 = "Your account will be blocked today. Verify your KYC immediately."
        response2 = self.client.post("/predict", json={"message": msg2})
        self.assertEqual(response2.status_code, 200)
        data2 = response2.json()
        self.assertEqual(data2.get("prediction"), "smishing")
        self.assertEqual(data2.get("risk"), "High Risk")

    def test_05_clear_spam_message(self):
        """Test prediction for typical unsolicited commercial spam messages."""
        msg = "Free entry in 2 a weekly competition to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C apply"
        response = self.client.post("/predict", json={"message": msg})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("prediction"), "spam")
        self.assertEqual(data.get("label"), "SPAM MESSAGE")
        self.assertIn("Risk", data.get("risk"))
        self.assertGreater(data.get("probabilities", {}).get("spam", 0.0), 0.5)

    def test_06_empty_and_whitespace_message(self):
        """Test API rejection of empty or whitespace-only messages."""
        # Empty string
        res_empty = self.client.post("/predict", json={"message": ""})
        self.assertEqual(res_empty.status_code, 422)

        # Whitespace-only string
        res_spaces = self.client.post("/predict", json={"message": "    \t\n   "})
        self.assertEqual(res_spaces.status_code, 422)

    def test_07_very_long_message(self):
        """Test valid long message within limit and rejection of message exceeding 5000 characters."""
        # Valid message with ~3000 characters
        long_valid_msg = "Hello friend, how are you doing today? " * 75
        self.assertLess(len(long_valid_msg), 5000)
        res_valid = self.client.post("/predict", json={"message": long_valid_msg})
        self.assertEqual(res_valid.status_code, 200)
        self.assertIn("prediction", res_valid.json())

        # Exceeding 5000 characters
        oversized_msg = "A" * 5001
        res_oversized = self.client.post("/predict", json={"message": oversized_msg})
        self.assertEqual(res_oversized.status_code, 422)

    def test_08_message_containing_url(self):
        """Test rule-based heuristic detects suspicious URLs."""
        msg = "Verify your account at https://secure-bank-login.phishingsite.com immediately."
        response = self.client.post("/predict", json={"message": msg})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        signal_names = [s["signal"] for s in data.get("signals", [])]
        self.assertIn("Suspicious URL Detected", signal_names)

    def test_09_message_containing_phone_number(self):
        """Test rule-based heuristic detects contact numbers or SMS shortcodes."""
        msg = "Call 09061701461 now to claim your voucher or text to 87121."
        response = self.client.post("/predict", json={"message": msg})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        signal_names = [s["signal"] for s in data.get("signals", [])]
        self.assertIn("Contact Number / Shortcode Detected", signal_names)

    def test_10_message_containing_kyc_and_banking_terms(self):
        """Test rule-based heuristic detects KYC, banking, and account verification terms."""
        msg = "Urgent: Complete your bank account KYC verification immediately."
        response = self.client.post("/predict", json={"message": msg})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        signal_names = [s["signal"] for s in data.get("signals", [])]
        self.assertIn("KYC Terminology", signal_names)
        self.assertIn("Banking Terminology", signal_names)
        self.assertIn("Account Verification Language", signal_names)

    def test_11_probability_distribution_integrity(self):
        """Test probability distribution structure and normalization."""
        msg = "Urgent update on your parcel delivery"
        response = self.client.post("/predict", json={"message": msg})
        self.assertEqual(response.status_code, 200)
        probs = response.json().get("probabilities", {})
        self.assertIn("ham", probs)
        self.assertIn("spam", probs)
        self.assertIn("smishing", probs)
        total_prob = sum(probs.values())
        self.assertAlmostEqual(total_prob, 1.0, delta=0.01)

    def test_12_invalid_request_body(self):
        """Test API rejection of malformed or missing body parameters."""
        # Missing message parameter
        res_missing = self.client.post("/predict", json={"text": "hello"})
        self.assertEqual(res_missing.status_code, 422)

        # Empty body
        res_empty = self.client.post("/predict", json={})
        self.assertEqual(res_empty.status_code, 422)

    def test_13_hybrid_risk_case1_legitimate(self):
        """Case 1: Standard legitimate conversation message -> Low Risk / LEGITIMATE MESSAGE."""
        msg = "Hey, are we meeting at 5 pm today?"
        response = self.client.post("/predict", json={"message": msg})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("prediction"), "ham")
        self.assertEqual(data.get("label"), "LEGITIMATE MESSAGE")
        self.assertEqual(data.get("risk"), "Low Risk")
        self.assertEqual(data.get("risk_level"), "low")

    def test_14_hybrid_risk_case2_promo_without_false_smishing(self):
        """Case 2: Promotional text without URLs/threats -> Preserves ham/spam behavior without false smishing."""
        msg = "Get 50% off on all products today. Visit our store now!"
        response = self.client.post("/predict", json={"message": msg})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertNotEqual(data.get("label"), "POTENTIAL SMISHING / PHISHING")
        self.assertNotEqual(data.get("risk_level"), "high")

    def test_15_hybrid_risk_case3_phishing_prize_url(self):
        """Case 3: Obvious prize + URL phishing -> High Risk / POTENTIAL SMISHING / PHISHING."""
        msg = "Congratulations! You have won 10 lakh dollars. Click this link to claim your prize: https://xxxxx"
        response = self.client.post("/predict", json={"message": msg})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("label"), "POTENTIAL SMISHING / PHISHING")
        self.assertEqual(data.get("risk"), "High Risk")
        self.assertEqual(data.get("risk_level"), "high")

    def test_16_hybrid_risk_case4_phishing_bank_threat_url(self):
        """Case 4: Account suspension threat + banking + URL -> High Risk / POTENTIAL SMISHING / PHISHING."""
        msg = "Your bank account has been suspended. Verify your account immediately: https://xxxxx"
        response = self.client.post("/predict", json={"message": msg})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("label"), "POTENTIAL SMISHING / PHISHING")
        self.assertEqual(data.get("risk"), "High Risk")
        self.assertEqual(data.get("risk_level"), "high")

    def test_17_hybrid_risk_case5_phishing_otp_url(self):
        """Case 5: OTP request + URL -> High Risk / POTENTIAL SMISHING / PHISHING."""
        msg = "Your OTP is required to verify your account. Send the code immediately: https://xxxxx"
        response = self.client.post("/predict", json={"message": msg})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("label"), "POTENTIAL SMISHING / PHISHING")
        self.assertEqual(data.get("risk"), "High Risk")
        self.assertEqual(data.get("risk_level"), "high")

    def test_18_hybrid_risk_case6_suspicious_ham_url_financial(self):
        """Case 6: ML predicts ham, but suspicious URL + financial signals exist -> Medium Risk / SUSPICIOUS MESSAGE."""
        msg = "Wanna 10 lakh dollars? Click on this link https://xxxxx"
        response = self.client.post("/predict", json={"message": msg})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("prediction"), "ham")
        self.assertEqual(data.get("label"), "SUSPICIOUS MESSAGE")
        self.assertEqual(data.get("risk"), "Medium Risk")
        self.assertEqual(data.get("risk_level"), "medium")


if __name__ == "__main__":
    unittest.main(verbosity=2)

