"""
main.py — FastAPI ML Service Entry Point
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Behavior-Aware Multi-Level Assessment System
ML Service: Learning-State Prediction via Random Forest

Endpoints:
    GET  /              — Health check
    POST /predict       — Predict learning state from behavioral features
    POST /detect-emotion — Simulate/detect facial expression from a frame

Start server (omit --reload until T5 Hugging Face files are cached, or downloads may fail):

    uvicorn main:app --host 127.0.0.1 --port 8000

Use the same interpreter as `backend\.venv` if `ml-service\venv` is broken.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os

# Reduce tokenizer multiprocessing quirks on Windows during HF loads (esp. uvicorn subprocesses).
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import tempfile
import threading
import traceback
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from modules.predict import predict_learning_state
from modules.emotion_placeholder import detect_emotion, SUPPORTED_EMOTIONS

log_ml = logging.getLogger("ml_service")

_qgen = None  # QuestionGenerator, loaded lazily on first /generate-questions call
_qgen_lock = threading.Lock()


def get_qgen():
    """Thread-safe lazy init — avoids concurrent HF downloads sharing/closing httpx prematurely."""
    global _qgen
    if _qgen is not None:
        return _qgen
    with _qgen_lock:
        if _qgen is None:
            print("Loading T5 question-generation model (first run ~30s)...")
            from modules.question_generator import QuestionGenerator

            _qgen = QuestionGenerator()
            print("T5 model ready.")
    return _qgen


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-download + load T5 before serving traffic (avoids raced first-request HF httpx failures)."""
    skip = os.getenv("ML_SKIP_T5_PRELOAD", "").strip().lower() in ("1", "true", "yes", "on")
    if not skip:
        try:
            await asyncio.to_thread(get_qgen)
        except Exception:
            log_ml.exception("T5 preload failed; generation will retry when /generate-questions is called")
    yield


# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="STEM Assessment ML Service",
    description="Learning-state prediction and emotion detection for the Behavior-Aware Assessment System",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from the React frontend and Express backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3001", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Schemas ────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """Input features for learning-state prediction."""
    correctness: int        = Field(..., ge=0, le=1, description="1=correct, 0=wrong")
    responseTime: float     = Field(..., gt=0,       description="Seconds to answer")
    answerChanges: int      = Field(0,  ge=0,        description="Times student changed answer")
    quizLevel: int          = Field(..., ge=1, le=3, description="Quiz level: 1, 2, or 3")
    detectedExpression: str = Field("neutral",       description="Emotion label from webcam")

    class Config:
        json_schema_extra = {
            "example": {
                "correctness": 0,
                "responseTime": 35.5,
                "answerChanges": 3,
                "quizLevel": 2,
                "detectedExpression": "confused",
            }
        }


class PredictResponse(BaseModel):
    """Prediction result returned to the caller."""
    learningState: str
    confidence: Optional[float]
    allProbabilities: dict
    inputFeatures: dict


class EmotionRequest(BaseModel):
    """Webcam frame for emotion detection (base64-encoded)."""
    frame: str = Field("", description="Base64-encoded JPEG frame from webcam")


class EmotionResponse(BaseModel):
    """Detected emotion label with confidence score."""
    detectedExpression: str
    confidence: float
    supportedEmotions: list


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health_check():
    """Health check — confirms ML service is running."""
    return {
        "status": "ok",
        "service": "STEM Assessment ML Service",
        "model": "Random Forest — Learning State Classifier",
        "emotions": SUPPORTED_EMOTIONS,
        "t5QuestionGeneratorLoaded": _qgen is not None,
    }


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(req: PredictRequest):
    """
    Predict the student's learning state based on behavioral features.

    - **correctness**: Whether the answer was correct (1) or wrong (0)
    - **responseTime**: Time taken in seconds
    - **answerChanges**: Number of times answer was changed before submitting
    - **quizLevel**: Assessment level (1 = Basic, 2 = Concept, 3 = Advanced)
    - **detectedExpression**: Facial expression detected by webcam module
    """
    try:
        result = predict_learning_state(
            correctness=req.correctness,
            response_time=req.responseTime,
            answer_changes=req.answerChanges,
            quiz_level=req.quizLevel,
            face_expression=req.detectedExpression,
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Model not trained yet. Run: python modules/train.py — {str(e)}",
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/generate-questions", tags=["Question Generation"])
async def generate_questions(
    pdf: UploadFile = File(..., description="PDF file to generate questions from"),
    numQuestions: int  = Form(9,         description="Total questions to generate (default 9)"),
    conceptTag:   str  = Form("General", description="Concept tag for all questions"),
    chapterTitle: str  = Form("",        description="If the PDF is a whole textbook, use only this chapter"),
    nextChapterTitle: str = Form("",     description="Title of the following chapter (marks where it ends)"),
    prevChapterTitle: str = Form("",     description="Title of the previous chapter (search starts after it)"),
):
    """
    Upload a PDF and receive auto-generated MCQ questions grouped by difficulty level.

    Returns { level1: [...], level2: [...], level3: [...] } where each question has:
    questionId, questionText, options, correctAnswer, hint, shortTheoryExplanation,
    conceptTag, quizLevel. With ``chapterTitle``, a whole textbook is narrowed to that
    chapter's pages (``chapterPages`` = [first, last], 1-based); if the chapter isn't
    found (e.g. the PDF is already just that chapter) the whole PDF is used.
    """
    if not (pdf.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")

    tmp_path = None
    try:
        content = await pdf.read()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        pages = None
        if chapterTitle.strip():
            from modules.question_generator import find_chapter_pages

            pages = await asyncio.to_thread(
                find_chapter_pages, tmp_path, chapterTitle, nextChapterTitle, prevChapterTitle
            )
        questions = await asyncio.to_thread(
            get_qgen().generate_from_pdf, tmp_path, numQuestions, conceptTag, pages
        )
        total = sum(len(v) for v in questions.values())
        return {
            "success": True,
            "total": total,
            "questions": questions,
            "chapterPages": [pages[0] + 1, pages[-1] + 1] if pages else None,
        }

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Question generation error: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/detect-emotion", response_model=EmotionResponse, tags=["Emotion"])
def detect_emotion_endpoint(req: EmotionRequest):
    """
    Detect the student's facial expression from a webcam frame.

    Currently uses a placeholder (simulated emotion). Replace the body of
    `modules/emotion_placeholder.py::detect_emotion()` with real CNN/MediaPipe
    inference to go live. The API contract stays the same.
    """
    try:
        emotion, confidence = detect_emotion(req.frame)
        return {"detectedExpression": emotion, "confidence": confidence, "supportedEmotions": SUPPORTED_EMOTIONS}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Emotion detection error: {str(e)}")
