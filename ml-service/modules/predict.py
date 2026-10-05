"""
predict.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Prediction logic module.
Loads the saved Random Forest model and exposes a predict() function
that is called by the FastAPI endpoint.

Keeps prediction logic separate from the HTTP layer (main.py).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Optional

from modules.emotion_placeholder import encode_emotion

# ── Model Path ────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "learning_state_model.pkl"

# ── Lazy-load model (loaded once at startup) ──────────────────────────────────
_model_payload: Optional[dict] = None


def _load_model() -> dict:
    """Load model from disk (called once on first prediction)."""
    global _model_payload
    if _model_payload is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model file not found at {MODEL_PATH}. "
                "Run: python modules/train.py"
            )
        _model_payload = joblib.load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    return _model_payload


def predict_learning_state(
    correctness: int,
    response_time: float,
    answer_changes: int,
    quiz_level: int,
    face_expression: str,
) -> dict:
    """
    Predict the student's learning state from behavioral features.

    Parameters
    ----------
    correctness     : 0 (wrong) or 1 (correct)
    response_time   : seconds taken to answer
    answer_changes  : number of answer changes before submitting
    quiz_level      : 1, 2, or 3
    face_expression : emotion string (e.g. "neutral", "confused")

    Returns
    -------
    dict with keys:
        learningState : str  — predicted class label
        confidence    : float — probability of predicted class (0.0–1.0)
        allProbabilities : dict — probability per class
        inputFeatures : dict — the encoded feature vector used
    """
    payload = _load_model()
    model   = payload["model"]
    le      = payload["label_encoder"]

    # Encode faceExpression string to integer
    expression_encoded = encode_emotion(face_expression)

    # Build feature vector (order must match training)
    feature_vector = np.array([[
        int(correctness),
        float(response_time),
        int(answer_changes),
        int(quiz_level),
        int(expression_encoded),
    ]])

    # Run prediction
    predicted_index     = model.predict(feature_vector)[0]
    predicted_label     = le.inverse_transform([predicted_index])[0]
    class_probabilities = model.predict_proba(feature_vector)[0]

    # Build probability map per class
    all_probs = {
        le.inverse_transform([i])[0]: round(float(p), 4)
        for i, p in enumerate(class_probabilities)
    }
    confidence = round(float(class_probabilities[predicted_index]), 4)

    return {
        "learningState": predicted_label,
        "confidence": confidence,
        "allProbabilities": all_probs,
        "inputFeatures": {
            "correctness": correctness,
            "responseTime": response_time,
            "answerChanges": answer_changes,
            "quizLevel": quiz_level,
            "faceExpression": face_expression,
            "faceExpressionEncoded": expression_encoded,
        },
    }
