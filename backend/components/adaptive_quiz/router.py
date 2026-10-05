"""Adaptive Quiz component API.

All routes are mounted by ``backend/main.py`` under ``/api``, so the public paths
are ``/api/adaptive-quiz/...``. Response envelopes match the original standalone
project (``{"success": true, "data": ...}``, ``_id`` strings) so ported clients
work unchanged.
"""
from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.common.config import settings
from backend.components.adaptive_quiz import ml_predict
from backend.components.adaptive_quiz.db import quiz_db_error, quiz_db_ready
from backend.components.adaptive_quiz.deps import StudentDep, TeacherDep, ensure_owner
from backend.components.adaptive_quiz.documents import (
    AssessmentReport,
    Lesson,
    LoginBody,
    Question,
    RegisterBody,
    Student,
    StudentResponse,
    StudyPlan,
)
from backend.components.adaptive_quiz.emotion import (
    SUPPORTED_EMOTIONS,
    analyze_frame,
    emotion_executor,
    emotion_model_status,
)
from backend.components.adaptive_quiz.quiz_logic import (
    QUESTIONS_PER_LEVEL,
    pick_random,
    rule_based_fallback,
)
from backend.components.adaptive_quiz.reporting import mode as _mode
from backend.components.adaptive_quiz.reporting import summary as _summary
from backend.components.adaptive_quiz.security import (
    document_legacy,
    generate_token,
    hash_password,
    verify_password,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/adaptive-quiz", tags=["Adaptive Quiz"])


def require_quiz_db() -> None:
    if not quiz_db_ready():
        reason = quiz_db_error() or "Set MONGO_URI in .env and restart."
        raise HTTPException(
            status_code=503,
            detail={
                "success": False,
                "message": f"Adaptive Quiz database is not connected. {reason}",
            },
        )


_DB = [Depends(require_quiz_db)]


# ── Status ────────────────────────────────────────────────────────────────────
@router.get("/status")
def status() -> dict:
    return {
        "status": "ok" if quiz_db_ready() else "degraded",
        "module": "Adaptive Quiz",
        "db": "connected" if quiz_db_ready() else "unavailable",
        "emotionModel": emotion_model_status(),
        "learningStateModel": ml_predict.model_status(),
    }


# ── Auth ──────────────────────────────────────────────────────────────────────
auth = APIRouter(prefix="/auth", dependencies=_DB)


@auth.post("/register")
async def register_student(body: RegisterBody):
    if not body.studentId or not body.name or not body.email or not body.password:
        return JSONResponse(status_code=400, content={"success": False, "message": "All fields are required"})

    exists = await Student.find_one(
        {"$or": [{"email": body.email.lower().strip()}, {"studentId": body.studentId}]}
    )
    if exists:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": "Student ID or Email already registered"},
        )

    student = Student(
        studentId=body.studentId,
        name=body.name,
        email=body.email.lower().strip(),
        password=hash_password(body.password),
        gradeLevel=body.gradeLevel or "Year 1",
    )
    await student.insert()

    return JSONResponse(
        status_code=201,
        content={
            "success": True,
            "message": "Registration successful",
            "data": {
                "_id": str(student.id),
                "studentId": student.studentId,
                "name": student.name,
                "email": student.email,
                "gradeLevel": student.gradeLevel,
                "token": generate_token(str(student.id)),
            },
        },
    )


@auth.post("/login")
async def login_student(body: LoginBody):
    if not body.email or not body.password:
        return JSONResponse(
            status_code=400, content={"success": False, "message": "Email and password are required"}
        )

    student = await Student.find_one(Student.email == body.email.lower().strip())
    if not student or not verify_password(body.password, student.password):
        return JSONResponse(
            status_code=401, content={"success": False, "message": "Invalid email or password"}
        )

    return {
        "success": True,
        "message": "Login successful",
        "data": {
            "_id": str(student.id),
            "studentId": student.studentId,
            "name": student.name,
            "email": student.email,
            "gradeLevel": student.gradeLevel,
            "token": generate_token(str(student.id)),
        },
    }


@auth.get("/profile")
async def get_profile(student: StudentDep):
    s = student.model_dump(mode="python")
    s["_id"] = str(student.id)
    s.pop("password", None)
    s.pop("id", None)
    return {"success": True, "data": s}


# ── Lessons ───────────────────────────────────────────────────────────────────
lessons = APIRouter(prefix="/lessons", dependencies=_DB)


@lessons.get("")
async def get_all_lessons():
    rows = await Lesson.find(Lesson.isActive == True).sort(-Lesson.id).to_list()  # noqa: E712
    return {"success": True, "count": len(rows), "data": [document_legacy(x) for x in rows]}


@lessons.get("/{lesson_id}")
async def get_lesson_by_id(lesson_id: str):
    lesson = await Lesson.find_one(Lesson.lessonId == lesson_id)
    if not lesson:
        return JSONResponse(status_code=404, content={"success": False, "message": "Lesson not found"})
    return {"success": True, "data": document_legacy(lesson)}


# ── Assessments ───────────────────────────────────────────────────────────────
assessments = APIRouter(prefix="/assessments", dependencies=_DB)


def _levels_payload(session: dict[int, list]) -> dict:
    return {
        "level1": {"label": "Basic", "questions": session[1]},
        "level2": {"label": "Concept Understanding", "questions": session[2]},
        "level3": {"label": "Advanced / Application", "questions": session[3]},
    }


@assessments.get("/{lesson_id}/adaptive/{student_id}")
async def get_adaptive_assessment_questions(lesson_id: str, student_id: str, _student: StudentDep):
    ensure_owner(student_id, _student)
    questions = (
        await Question.find(Question.lessonId == lesson_id)
        .sort(+Question.quizLevel, +Question.order)
        .to_list()
    )
    if not questions:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": f"No questions found for lesson: {lesson_id}"},
        )

    last_rows = (
        await AssessmentReport.find(
            AssessmentReport.studentId == student_id,
            AssessmentReport.lessonId == lesson_id,
        )
        .sort(-AssessmentReport.id)
        .limit(1)
        .to_list()
    )
    last_report = last_rows[0] if last_rows else None
    previous_weak_areas = list(last_report.weakAreas) if last_report and last_report.weakAreas else []
    previous_score = last_report.totalScore if last_report else None
    previous_session_id = last_report.sessionId if last_report else None

    bank_objs: dict[int, list] = {1: [], 2: [], 3: []}
    for q in questions:
        bank_objs[q.quizLevel].append(q)

    if previous_weak_areas:
        for lvl in (1, 2, 3):
            weak_ones = [q for q in bank_objs[lvl] if q.conceptTag in previous_weak_areas]
            other_ones = [q for q in bank_objs[lvl] if q.conceptTag not in previous_weak_areas]
            # Prioritise the student's weak concepts, then fill the level with random
            # other questions, then top up from anything left so every level always
            # has QUESTIONS_PER_LEVEL — a returning student still gets the full set.
            picked = weak_ones[:QUESTIONS_PER_LEVEL]
            picked += pick_random(other_ones, QUESTIONS_PER_LEVEL - len(picked))
            if len(picked) < QUESTIONS_PER_LEVEL:
                leftover = [q for q in bank_objs[lvl] if q not in picked]
                picked += pick_random(leftover, QUESTIONS_PER_LEVEL - len(picked))
            bank_objs[lvl] = picked

    session_objs = {
        1: bank_objs[1][:QUESTIONS_PER_LEVEL] if bank_objs[1] else [],
        2: bank_objs[2][:QUESTIONS_PER_LEVEL] if bank_objs[2] else [],
        3: bank_objs[3][:QUESTIONS_PER_LEVEL] if bank_objs[3] else [],
    }

    if not previous_weak_areas:
        for lvl in (1, 2, 3):
            session_objs[lvl] = pick_random(bank_objs[lvl], QUESTIONS_PER_LEVEL)

    session = {k: [document_legacy(q) for q in v] for k, v in session_objs.items()}
    total = sum(len(session[i]) for i in (1, 2, 3))

    return {
        "success": True,
        "lessonId": lesson_id,
        "isAdaptive": len(previous_weak_areas) > 0,
        "previousWeakAreas": previous_weak_areas,
        "previousScore": previous_score,
        "previousSessionId": previous_session_id,
        "bankSize": len(questions),
        "totalQuestions": total,
        "levels": _levels_payload(session),
    }


@assessments.get("/{lesson_id}/level/{level}")
async def get_questions_by_level(lesson_id: str, level: str, _student: StudentDep):
    try:
        quiz_level = int(level, 10)
    except ValueError:
        return JSONResponse(status_code=400, content={"success": False, "message": "Level must be 1, 2, or 3"})
    if quiz_level not in (1, 2, 3):
        return JSONResponse(status_code=400, content={"success": False, "message": "Level must be 1, 2, or 3"})

    rows = (
        await Question.find(Question.lessonId == lesson_id, Question.quizLevel == quiz_level)
        .sort(+Question.order)
        .to_list()
    )
    payload = [document_legacy(q) for q in rows]
    return {"success": True, "count": len(payload), "data": payload}


@assessments.get("/{lesson_id}")
async def get_assessment_questions(lesson_id: str, _student: StudentDep):
    questions = (
        await Question.find(Question.lessonId == lesson_id)
        .sort(+Question.quizLevel, +Question.order)
        .to_list()
    )
    if not questions:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": f"No questions found for lesson: {lesson_id}"},
        )

    bank: dict[int, list] = {1: [], 2: [], 3: []}
    for q in questions:
        bank[q.quizLevel].append(document_legacy(q))

    session = {lvl: pick_random(bank[lvl], QUESTIONS_PER_LEVEL) for lvl in (1, 2, 3)}
    total = sum(len(session[i]) for i in (1, 2, 3))
    return {
        "success": True,
        "lessonId": lesson_id,
        "bankSize": len(questions),
        "totalQuestions": total,
        "levels": _levels_payload(session),
    }


# ── Responses ─────────────────────────────────────────────────────────────────
responses = APIRouter(prefix="/responses", dependencies=_DB)


def _build_response(raw: dict[str, Any]) -> StudentResponse:
    return StudentResponse(
        studentId=str(raw.get("studentId")),
        lessonId=str(raw.get("lessonId")),
        questionId=str(raw.get("questionId")),
        quizLevel=int(raw.get("quizLevel", 1)),
        conceptTag=str(raw.get("conceptTag", "")),
        selectedAnswer=str(raw.get("selectedAnswer", "")),
        correctness=int(raw.get("correctness", 0)),
        responseTime=float(raw.get("responseTime", 0)),
        answerChanges=float(raw.get("answerChanges", 0) or 0),
        attempts=float(raw.get("attempts", 1) or 1),
        detectedExpression=str(raw.get("detectedExpression", "neutral")),
        learningState=str(raw.get("learningState", "partial_understanding")),
        cognitiveLoad=float(raw.get("cognitiveLoad", 0) or 0),
        adaptiveLevel=int(raw.get("adaptiveLevel", 0) or 0),
        hintUsed=bool(raw.get("hintUsed", False)),
        sessionId=str(raw.get("sessionId")),
    )


async def _upsert_responses(rows: list[StudentResponse]) -> None:
    """Idempotent write: one row per (sessionId, questionId).

    The frontend saves each answer as it happens *and* bulk-saves the whole set
    when the assessment ends, so the same answer can arrive twice. Replacing any
    existing (sessionId, questionId) row keeps exactly one, so the report never
    double-counts responses, hint usage or time.
    """
    for doc in rows:
        await StudentResponse.find(
            StudentResponse.sessionId == doc.sessionId,
            StudentResponse.questionId == doc.questionId,
        ).delete()
    await StudentResponse.insert_many(rows)


@responses.post("")
async def save_response(_student: StudentDep, body: dict[str, Any]):
    required = ("studentId", "lessonId", "questionId", "sessionId")
    if not all(body.get(k) for k in required):
        return JSONResponse(status_code=400, content={"success": False, "message": "Missing required fields"})
    ensure_owner(str(body.get("studentId")), _student)
    resp = _build_response(body)
    await _upsert_responses([resp])
    return JSONResponse(status_code=201, content={"success": True, "data": document_legacy(resp)})


@responses.post("/bulk")
async def save_bulk_responses(_student: StudentDep, body: dict[str, Any]):
    items = body.get("responses")
    if not isinstance(items, list) or not items:
        return JSONResponse(
            status_code=400, content={"success": False, "message": "responses array is required"}
        )
    for r in items:
        ensure_owner(str(r.get("studentId")), _student)
    docs = [_build_response(r) for r in items]
    await _upsert_responses(docs)
    payload = [document_legacy(r) for r in docs]
    return JSONResponse(
        status_code=201, content={"success": True, "count": len(payload), "data": payload}
    )


@responses.get("/{session_id}")
async def get_session_responses(session_id: str, _student: StudentDep):
    rows = (
        await StudentResponse.find(
            StudentResponse.sessionId == session_id,
            StudentResponse.studentId == _student.studentId,
        )
        .sort(+StudentResponse.id)
        .to_list()
    )
    payload = [document_legacy(r) for r in rows]
    return {"success": True, "count": len(payload), "data": payload}


# ── Learning-state prediction ─────────────────────────────────────────────────
predict = APIRouter(prefix="/predict-learning-state", dependencies=_DB)


@predict.post("")
async def predict_learning_state_endpoint(_student: StudentDep, body: dict[str, Any]):
    correctness = body.get("correctness")
    response_time = body.get("responseTime")
    answer_changes = body.get("answerChanges") or 0
    quiz_level = body.get("quizLevel")
    detected_expression = body.get("detectedExpression") or "neutral"

    if correctness is None or not response_time or quiz_level is None:
        return JSONResponse(
            status_code=400, content={"success": False, "message": "Missing prediction input features"}
        )

    features = {
        "correctness": int(correctness),
        "responseTime": float(response_time),
        "answerChanges": int(answer_changes),
        "quizLevel": int(quiz_level),
        "detectedExpression": str(detected_expression),
    }

    # 1) ML micro-service
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{settings.ml_service_url.rstrip('/')}/predict", json=features)
            r.raise_for_status()
            data = r.json()
        return {
            "success": True,
            "learningState": data.get("learningState"),
            "confidence": data.get("confidence"),
            "inputFeatures": data.get("inputFeatures"),
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("ML service /predict unavailable (%s); trying in-process model", e)

    # 2) In-process RandomForest
    try:
        result = await asyncio.to_thread(
            ml_predict.predict_learning_state,
            int(correctness), float(response_time), int(answer_changes),
            int(quiz_level), str(detected_expression),
        )
        return {
            "success": True,
            "learningState": result["learningState"],
            "confidence": result["confidence"],
            "inputFeatures": result["inputFeatures"],
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("In-process model unavailable (%s); using rule-based fallback", e)

    # 3) Rule-based fallback
    return {
        "success": True,
        "learningState": rule_based_fallback(
            correctness, response_time, answer_changes, str(detected_expression)
        ),
        "confidence": None,
        "fallback": True,
        "message": "ML unavailable; rule-based fallback used",
    }


# ── Reports ───────────────────────────────────────────────────────────────────
reports = APIRouter(prefix="/reports", dependencies=_DB)


@reports.get("/all")
async def get_all_reports(_teacher: TeacherDep, lessonId: str | None = None):
    if lessonId:
        reps = (
            await AssessmentReport.find(AssessmentReport.lessonId == lessonId)
            .sort(-AssessmentReport.id)
            .to_list()
        )
    else:
        reps = await AssessmentReport.find_all().sort(-AssessmentReport.id).to_list()
    payload = [document_legacy(r) for r in reps]
    return {"success": True, "count": len(payload), "data": payload}


@reports.get("/history/{student_id}")
async def get_student_history(student_id: str, _student: StudentDep):
    ensure_owner(student_id, _student)
    reps = (
        await AssessmentReport.find(AssessmentReport.studentId == student_id)
        .sort(-AssessmentReport.id)
        .to_list()
    )
    payload = [document_legacy(r) for r in reps]
    return {"success": True, "count": len(payload), "data": payload}


@reports.get("/progress/{student_id}/{lesson_id}")
async def get_progress_over_time(student_id: str, lesson_id: str, _student: StudentDep):
    ensure_owner(student_id, _student)
    reps = (
        await AssessmentReport.find(
            AssessmentReport.studentId == student_id,
            AssessmentReport.lessonId == lesson_id,
        )
        .sort(+AssessmentReport.id)
        .to_list()
    )
    if not reps:
        return {"success": True, "count": 0, "data": []}

    timeline: list[dict] = []
    for idx, r in enumerate(reps):
        prev = reps[idx - 1] if idx > 0 else None
        score_delta = round(float(r.totalScore) - float(prev.totalScore), 1) if prev else None

        improved_concepts: list[str] = []
        persistent_weak: list[str] = []
        if prev and prev.weakAreas:
            for area in prev.weakAreas:
                current_score = (
                    r.conceptPerformance.get(area) if isinstance(r.conceptPerformance, dict) else None
                )
                if current_score is not None and current_score >= 60:
                    improved_concepts.append(area)
                else:
                    persistent_weak.append(area)

        timeline.append(
            {
                "sessionIndex": idx + 1,
                "sessionId": r.sessionId,
                "date": r.model_dump(mode="python").get("createdAt"),
                "totalScore": r.totalScore,
                "scoreDelta": score_delta,
                "weakAreas": list(r.weakAreas or []),
                "improvedConcepts": improved_concepts,
                "persistentWeakAreas": persistent_weak,
                "hintUsageCount": r.hintUsageCount,
                "totalTime": r.totalTime,
            }
        )
    return {"success": True, "count": len(timeline), "data": timeline}


@reports.get("/{student_id}/{lesson_id}")
async def get_report(student_id: str, lesson_id: str, _student: StudentDep, sessionId: str | None = None):
    ensure_owner(student_id, _student)
    q: dict = {"studentId": student_id, "lessonId": lesson_id}
    if sessionId:
        q["sessionId"] = sessionId

    resp_rows = await StudentResponse.find(q).sort(+StudentResponse.id).to_list()
    if not resp_rows:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "No responses found for this student and lesson"},
        )

    total_correct = sum(1 for r in resp_rows if r.correctness == 1)
    total_score = (total_correct / len(resp_rows)) * 100

    level_groups: dict[int, list] = {1: [], 2: [], 3: []}
    for r in resp_rows:
        level_groups[r.quizLevel].append(r)

    level_scores: dict[str, float] = {}
    for lvl in (1, 2, 3):
        group = level_groups[lvl]
        if group:
            corr = sum(1 for x in group if x.correctness == 1)
            level_scores[str(lvl)] = round((corr / len(group)) * 100, 1)

    concept_groups: dict[str, list] = {}
    for r in resp_rows:
        concept_groups.setdefault(r.conceptTag, []).append(r)

    concept_performance: dict[str, float] = {}
    for tag, group in concept_groups.items():
        corr = sum(1 for x in group if x.correctness == 1)
        concept_performance[tag] = round((corr / len(group)) * 100, 1)

    all_exprs = [r.detectedExpression for r in resp_rows]
    expression_frequency: dict[str, int] = {}
    for e in all_exprs:
        expression_frequency[e] = expression_frequency.get(e, 0) + 1
    most_common_expression = _mode(all_exprs)

    weak_areas = [tag for tag, score in concept_performance.items() if score < 60]
    hint_usage_count = sum(1 for r in resp_rows if r.hintUsed)
    total_time = round(sum(float(r.responseTime or 0) for r in resp_rows), 1)

    learning_state_distribution: dict[str, int] = {}
    for r in resp_rows:
        st = r.learningState or "partial_understanding"
        learning_state_distribution[st] = learning_state_distribution.get(st, 0) + 1

    progress_summary = _summary(total_score, weak_areas, hint_usage_count)
    resolved_session = sessionId or resp_rows[-1].sessionId

    rep = await AssessmentReport.find_one(AssessmentReport.sessionId == resolved_session)
    report_kwargs = dict(
        studentId=student_id,
        lessonId=lesson_id,
        sessionId=resolved_session,
        totalScore=round(total_score, 1),
        levelScores=level_scores,
        conceptPerformance=concept_performance,
        mostCommonExpression=most_common_expression,
        expressionFrequency=expression_frequency,
        weakAreas=weak_areas,
        progressSummary=progress_summary,
        hintUsageCount=hint_usage_count,
        learningStateDistribution=learning_state_distribution,
        totalTime=total_time,
    )

    if rep:
        for k, v in report_kwargs.items():
            setattr(rep, k, v)
        now = datetime.now(UTC)
        # Rows written by an older schema (before these fields existed) load as None
        # and would fail re-validation on save.
        if getattr(rep, "createdAt", None) is None:
            rep.createdAt = now
        rep.updatedAt = now
        await rep.save()
    else:
        rep = AssessmentReport(**report_kwargs)
        await rep.insert()

    merged = document_legacy(rep)
    merged["responses"] = [document_legacy(x) for x in resp_rows]
    return {"success": True, "data": merged}


# ── Emotion detection ─────────────────────────────────────────────────────────
class EmotionRequest(BaseModel):
    frame: str = Field("", description="Base64-encoded JPEG frame (no data: URL prefix)")


class EmotionResponse(BaseModel):
    detectedExpression: str
    confidence: float
    supportedEmotions: list[str]


_emotion_busy = False
_last_emotion: tuple[str, float] = ("neutral", 0.0)


@router.post("/detect-emotion", response_model=EmotionResponse)
async def detect_emotion(req: EmotionRequest) -> EmotionResponse:
    """Public (no JWT) — analyse one webcam frame with ``backend/model/best.pt``.

    Single-flight: if an inference is already running, drop this frame and return
    the last result immediately. Keeps a burst of frames from queueing up.
    """
    global _emotion_busy, _last_emotion
    if not _emotion_busy:
        _emotion_busy = True
        try:
            loop = asyncio.get_running_loop()
            _last_emotion = await loop.run_in_executor(emotion_executor, analyze_frame, req.frame)
        finally:
            _emotion_busy = False
    expression, confidence = _last_emotion
    return EmotionResponse(
        detectedExpression=expression,
        confidence=confidence,
        supportedEmotions=list(SUPPORTED_EMOTIONS),
    )


# ── Study plan (persisted weekly timetable) ───────────────────────────────────
study_plan = APIRouter(prefix="/study-plan", dependencies=_DB)


class StudyPlanBody(BaseModel):
    schedule: list[dict[str, Any]]
    intensity: str = "medium"
    weeklyMinutes: float = 0
    weekRangeLabel: str = ""


@study_plan.get("/{lesson_id}")
async def get_study_plan(lesson_id: str, _student: StudentDep):
    plan = await StudyPlan.find_one(
        StudyPlan.studentId == _student.studentId, StudyPlan.lessonId == lesson_id
    )
    if not plan:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "No saved study plan for this lesson"},
        )
    return {"success": True, "data": document_legacy(plan)}


@study_plan.put("/{lesson_id}")
async def save_study_plan(lesson_id: str, body: StudyPlanBody, _student: StudentDep):
    plan = await StudyPlan.find_one(
        StudyPlan.studentId == _student.studentId, StudyPlan.lessonId == lesson_id
    )
    if plan:
        plan.schedule = body.schedule
        plan.intensity = body.intensity
        plan.weeklyMinutes = body.weeklyMinutes
        plan.weekRangeLabel = body.weekRangeLabel
        plan.updatedAt = datetime.now(UTC)
        await plan.save()
    else:
        plan = StudyPlan(
            studentId=_student.studentId,
            lessonId=lesson_id,
            schedule=body.schedule,
            intensity=body.intensity,
            weeklyMinutes=body.weeklyMinutes,
            weekRangeLabel=body.weekRangeLabel,
        )
        await plan.insert()
    return {"success": True, "data": document_legacy(plan)}


@study_plan.delete("/{lesson_id}")
async def reset_study_plan(lesson_id: str, _student: StudentDep):
    plan = await StudyPlan.find_one(
        StudyPlan.studentId == _student.studentId, StudyPlan.lessonId == lesson_id
    )
    if plan:
        await plan.delete()
    return {"success": True, "message": "Study plan reset to auto-generated"}


# ── PDF → questions (T5, via ML micro-service) ────────────────────────────────
questions = APIRouter(prefix="/questions", dependencies=_DB)

_MAX_PDF = 20 * 1024 * 1024
_GEN_TIMEOUT = httpx.Timeout(600.0, connect=120.0, pool=None)


async def _ml_generate(raw: bytes, filename: str, mime: str, concept_tag: str, num_questions: str) -> dict:
    files = {"pdf": (filename, raw, mime or "application/pdf")}
    data = {"conceptTag": concept_tag, "numQuestions": num_questions}
    async with httpx.AsyncClient(timeout=_GEN_TIMEOUT) as client:
        r = await client.post(
            f"{settings.ml_service_url.rstrip('/')}/generate-questions", files=files, data=data
        )
        r.raise_for_status()
        return r.json()


def _ml_error(exc: Exception) -> JSONResponse:
    if isinstance(exc, httpx.HTTPStatusError):
        detail = ""
        try:
            detail = str(exc.response.json().get("detail") or exc.response.text)
        except Exception:  # noqa: BLE001
            detail = exc.response.text or str(exc)
        return JSONResponse(
            status_code=502,
            content={"success": False, "message": f"PDF question generator failed ({exc.response.status_code}): {detail}"},
        )
    return JSONResponse(
        status_code=502,
        content={
            "success": False,
            "message": (
                f"Cannot reach the PDF/ML service at {settings.ml_service_url}: {exc}. "
                "Start it from the ml-service/ folder: "
                "uvicorn main:app --host 127.0.0.1 --port 8001"
            ),
        },
    )


@questions.post("/generate")
async def generate_questions(
    _teacher: TeacherDep,
    pdf: UploadFile = File(...),
    lessonId: str = Form("auto"),
    conceptTag: str = Form("General"),
    numQuestions: str = Form("9"),
):
    if (pdf.content_type or "") != "application/pdf":
        return JSONResponse(status_code=400, content={"success": False, "message": "Only PDF files are allowed."})
    raw = await pdf.read()
    if len(raw) > _MAX_PDF:
        return JSONResponse(status_code=400, content={"success": False, "message": "PDF exceeds 20 MB limit"})

    try:
        ml_data = await _ml_generate(raw, pdf.filename or "upload.pdf", pdf.content_type or "", conceptTag, numQuestions)
    except Exception as e:  # noqa: BLE001
        return _ml_error(e)

    bucket: dict[str, Any] = ml_data.get("questions") or {}
    saved: dict[str, list] = {"level1": [], "level2": [], "level3": []}
    errors: list[str] = []
    for level_key, qlist in bucket.items():
        if not isinstance(qlist, list):
            continue
        for q in qlist:
            try:
                doc = Question(
                    questionId=q["questionId"],
                    lessonId=lessonId,
                    questionText=q["questionText"],
                    options=q["options"],
                    correctAnswer=q["correctAnswer"],
                    hint=q["hint"],
                    shortTheoryExplanation=q["shortTheoryExplanation"],
                    conceptTag=q.get("conceptTag") or conceptTag,
                    quizLevel=int(q["quizLevel"]),
                    source="pdf_generated",
                )
                await doc.insert()
                saved.setdefault(level_key, []).append(document_legacy(doc))
            except Exception as ex:  # noqa: BLE001
                errors.append(str(ex))

    total = sum(len(v) for v in saved.values())
    out: dict[str, Any] = {"success": True, "message": f"Generated {total} questions from PDF", "questions": saved}
    if errors:
        out["warnings"] = errors
    return JSONResponse(status_code=201, content=out)


@questions.post("/practice-generate")
async def generate_practice_questions(
    _student: StudentDep,
    pdf: UploadFile = File(...),
    conceptTag: str = Form("General"),
    numQuestions: str = Form("9"),
):
    """Student self-study: PDF → MCQs returned only, never saved or scored."""
    if (pdf.content_type or "") != "application/pdf":
        return JSONResponse(status_code=400, content={"success": False, "message": "Only PDF files are allowed."})
    raw = await pdf.read()
    if len(raw) > _MAX_PDF:
        return JSONResponse(status_code=400, content={"success": False, "message": "PDF exceeds 20 MB limit"})

    try:
        ml_data = await _ml_generate(raw, pdf.filename or "upload.pdf", pdf.content_type or "", conceptTag, numQuestions)
    except Exception as e:  # noqa: BLE001
        return _ml_error(e)

    bucket: dict[str, Any] = ml_data.get("questions") or {}
    total = sum(len(v) for v in bucket.values() if isinstance(v, list))
    return {"success": True, "message": f"Generated {total} practice questions from PDF", "questions": bucket}


for _sub in (auth, lessons, assessments, responses, predict, reports, study_plan, questions):
    router.include_router(_sub)
