"""Seed the quiz database with a demo lesson + 9 questions.

Run from the project root:  python -m backend.components.adaptive_quiz.seed
"""
from __future__ import annotations

import asyncio

from backend.components.adaptive_quiz.db import close_quiz_db, init_quiz_db
from backend.components.adaptive_quiz.documents import Lesson, Question

LESSON_ID = "human-heart-101"

LESSON_DATA = {
    "lessonId": LESSON_ID,
    "title": "The Human Heart",
    "subject": "Biology",
    "gradeLevel": "Grade 10",
    "grade": 10,
    "part": 1,
    "arModuleId": "ar-heart-v1",
    "description": (
        "Explore the structure and function of the human heart. Learn about the four chambers, "
        "blood vessels, valves, and the two circulatory loops."
    ),
    "estimatedDuration": 25,
    "isActive": True,
}

QUESTIONS_DATA: list[dict] = [
    {
        "lessonId": LESSON_ID, "questionId": "hh-q1", "quizLevel": 1, "conceptTag": "Heart Function", "order": 1,
        "questionText": "What is the primary function of the human heart?",
        "options": [
            "To filter blood and remove waste",
            "To pump blood throughout the body",
            "To produce red blood cells",
            "To store oxygen for muscles",
        ],
        "correctAnswer": "To pump blood throughout the body",
        "shortTheoryExplanation": (
            "The heart is a muscular organ that acts as a pump. It continuously contracts and relaxes to "
            "circulate blood through the pulmonary (lungs) and systemic (body) circuits."
        ),
        "hint": "Think about what the heart does physically — it is a muscle that contracts rhythmically.",
    },
    {
        "lessonId": LESSON_ID, "questionId": "hh-q2", "quizLevel": 1, "conceptTag": "Heart Structure", "order": 2,
        "questionText": "How many chambers does the human heart have?",
        "options": ["2", "3", "4", "6"],
        "correctAnswer": "4",
        "shortTheoryExplanation": (
            "The human heart has four chambers: right atrium, right ventricle, left atrium, and left ventricle. "
            "The atria receive blood; the ventricles pump blood out."
        ),
        "hint": "Remember: there are two atria (upper) and two ventricles (lower).",
    },
    {
        "lessonId": LESSON_ID, "questionId": "hh-q3", "quizLevel": 1, "conceptTag": "Blood Vessels", "order": 3,
        "questionText": "Which blood vessel carries oxygenated blood away from the heart to the body?",
        "options": ["Pulmonary vein", "Vena cava", "Aorta", "Pulmonary artery"],
        "correctAnswer": "Aorta",
        "shortTheoryExplanation": (
            "The aorta is the largest artery in the body. It originates from the left ventricle and carries "
            "oxygen-rich blood to the rest of the body."
        ),
        "hint": 'The biggest artery leaving the heart starts with the letter "A".',
    },
    {
        "lessonId": LESSON_ID, "questionId": "hh-q4", "quizLevel": 2, "conceptTag": "Heart Chambers", "order": 1,
        "questionText": "What is the specific role of the left ventricle in cardiac function?",
        "options": [
            "Receives deoxygenated blood from the body",
            "Pumps oxygenated blood to the lungs",
            "Pumps oxygenated blood to the entire body via the aorta",
            "Receives oxygenated blood from the lungs",
        ],
        "correctAnswer": "Pumps oxygenated blood to the entire body via the aorta",
        "shortTheoryExplanation": (
            "The left ventricle is the most muscular chamber because it must generate enough force to push "
            "blood through the aorta into the systemic circulation."
        ),
        "hint": "The left ventricle is the most powerful pump — which circuit needs more pressure?",
    },
    {
        "lessonId": LESSON_ID, "questionId": "hh-q5", "quizLevel": 2, "conceptTag": "Circulation", "order": 2,
        "questionText": "What is the key difference between pulmonary and systemic circulation?",
        "options": [
            "Pulmonary circulation carries blood to the body; systemic goes to lungs",
            "Pulmonary circulation carries blood to the lungs; systemic carries blood to the body",
            "Both carry oxygenated blood but to different organs",
            "Systemic circulation occurs only in the heart",
        ],
        "correctAnswer": "Pulmonary circulation carries blood to the lungs; systemic carries blood to the body",
        "shortTheoryExplanation": (
            "Pulmonary circulation (right side) sends deoxygenated blood to the lungs. Systemic circulation "
            "(left side) delivers oxygenated blood to all body tissues."
        ),
        "hint": '"Pulmonary" comes from the Latin word for lungs.',
    },
    {
        "lessonId": LESSON_ID, "questionId": "hh-q6", "quizLevel": 2, "conceptTag": "Heart Sounds", "order": 3,
        "questionText": 'What causes the "lub-dub" sound heard when listening to a heartbeat?',
        "options": [
            "Blood rushing through the aorta",
            "The heart muscle contracting",
            "The closing of heart valves during the cardiac cycle",
            "Electrical signals firing in the sinoatrial node",
        ],
        "correctAnswer": "The closing of heart valves during the cardiac cycle",
        "shortTheoryExplanation": (
            '"Lub" (S1) is the mitral and tricuspid valves closing; "Dub" (S2) is the aortic and pulmonary '
            "valves closing."
        ),
        "hint": "Heart sounds are caused by something closing, not something opening.",
    },
    {
        "lessonId": LESSON_ID, "questionId": "hh-q7", "quizLevel": 3, "conceptTag": "Cardiac Pathology", "order": 1,
        "questionText": (
            "A patient has a 90% blockage in the left anterior descending (LAD) coronary artery. "
            "Which cardiac process is most critically threatened?"
        ),
        "options": [
            "Electrical conduction through the AV node",
            "Oxygen supply to the left ventricular myocardium",
            "Blood return from systemic veins to the right atrium",
            "Valve movement during diastole",
        ],
        "correctAnswer": "Oxygen supply to the left ventricular myocardium",
        "shortTheoryExplanation": (
            "Coronary arteries supply the myocardium itself. The LAD perfuses much of the left ventricle; a "
            "90% blockage causes ischemia and risks myocardial infarction."
        ),
        "hint": "Coronary arteries feed the heart muscle itself.",
    },
    {
        "lessonId": LESSON_ID, "questionId": "hh-q8", "quizLevel": 3, "conceptTag": "Cardiac Physiology", "order": 2,
        "questionText": (
            "During intense exercise, heart rate increases significantly. At the cellular level, which "
            "mechanism primarily drives this increase?"
        ),
        "options": [
            "Increased parasympathetic (vagal) nerve signals slow the SA node",
            "Sympathetic nervous system releases norepinephrine, accelerating SA node firing rate",
            "The myocardium directly detects low oxygen and self-accelerates",
            "Increased CO2 in blood activates the AV node directly",
        ],
        "correctAnswer": "Sympathetic nervous system releases norepinephrine, accelerating SA node firing rate",
        "shortTheoryExplanation": (
            "During exercise the sympathetic nervous system releases norepinephrine, which binds beta-1 "
            "receptors on the SA node and increases its depolarisation rate."
        ),
        "hint": "Which branch of the autonomic nervous system is activated during physical stress?",
    },
    {
        "lessonId": LESSON_ID, "questionId": "hh-q9", "quizLevel": 3, "conceptTag": "Heart Structure", "order": 3,
        "questionText": "Why does the left ventricle have significantly thicker muscular walls than the right ventricle?",
        "options": [
            "It holds a larger volume of blood at rest",
            "It must generate higher pressure to pump blood through the systemic circulation",
            "It beats at a faster rate than the right ventricle",
            "It receives blood from both atria simultaneously",
        ],
        "correctAnswer": "It must generate higher pressure to pump blood through the systemic circulation",
        "shortTheoryExplanation": (
            "Systemic circulation has high vascular resistance (~120 mmHg) while pulmonary circulation is "
            "lower (~25 mmHg), so the left ventricle needs far more muscle mass."
        ),
        "hint": "Compare the distances each ventricle must push blood.",
    },
]


async def seed() -> None:
    await init_quiz_db()
    await Lesson.find(Lesson.lessonId == LESSON_ID).delete()
    await Question.find(Question.lessonId == LESSON_ID).delete()
    print("Cleared existing demo lesson data")

    await Lesson(**LESSON_DATA).insert()
    print(f'Lesson created: "{LESSON_DATA["title"]}" ({LESSON_ID})')

    await Question.insert_many([Question(**q) for q in QUESTIONS_DATA])
    print(f"{len(QUESTIONS_DATA)} questions inserted")


async def _main() -> None:
    try:
        await seed()
        print("Seed complete.")
    finally:
        await close_quiz_db()


if __name__ == "__main__":
    asyncio.run(_main())
