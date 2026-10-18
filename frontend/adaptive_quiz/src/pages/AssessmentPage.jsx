/**
 * pages/AssessmentPage.jsx
 * The core assessment flow page.
 *
 * Responsibilities:
 *  - Background webcam runs continuously via useWebcam hook
 *  - Per-question timer via useTimer hook
 *  - Tracks answerChanges, attempts
 *  - On submit: calls predict-learning-state, applies support logic,
 *    saves response, shows hint/theory, then advances
 *  - After all 9 questions: saves bulk responses + navigates to ReportPage
 */
import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';

import { useAssessment }     from '../context/AssessmentContext';
import useTimer              from '../hooks/useTimer';
import useWebcam             from '../hooks/useWebcam';

import LevelProgressBar      from '../components/assessment/LevelProgressBar';
import Timer                 from '../components/assessment/Timer';
import AnswerOptions         from '../components/assessment/AnswerOptions';
import HintBox               from '../components/assessment/HintBox';
import TheoryExplanation     from '../components/assessment/TheoryExplanation';
import WebcamCapture         from '../components/emotion/WebcamCapture';
import { useVoice } from '../context/VoiceContext';
import BrainBreakPuzzle from '../components/research/BrainBreakPuzzle';
import MemoryMatchPuzzle from '../components/research/MemoryMatchPuzzle';
import MathBlitzPuzzle from '../components/research/MathBlitzPuzzle';
import CircuitLogicPuzzle from '../components/research/CircuitLogicPuzzle';
import ExpressionQuest from '../components/research/ExpressionQuest';
import ExpressionRocket from '../components/research/ExpressionRocket';

import { predictLearningState, saveResponse, saveBulkResponses } from '../api/responseApi';
import { 
  evaluateResponse, evaluateProactiveSupport,
  getLevelLabel, getLearningStateColor,
  calculateCognitiveLoad, getAdaptiveNextLevel
}  from '../utils/quizLogic';

import './AssessmentPage.css';

const AssessmentPage = () => {
  const navigate = useNavigate();
  const {
    student, lesson, currentQuestion, currentLevel,
    currentQuestionIdx, currentQuestions, answeredCount, totalQuestions,
    sessionId, responses, isAssessmentDone,
    saveResponse: ctxSaveResponse, nextQuestion, setExpression,
  } = useAssessment();

  const { registerCommands, speak } = useVoice();

  // Hooks
  const timer  = useTimer();
  const webcam = useWebcam(true); // webcam always running

  // Question-level state
  const [selectedAnswer,  setSelectedAnswer]  = useState('');
  const [answerChanges,   setAnswerChanges]   = useState(0);
  const [submitted,       setSubmitted]       = useState(false);
  const [submitting,      setSubmitting]      = useState(false);

  // Feedback state
  const [showHint,         setShowHint]         = useState(false);
  const [showExplanation,  setShowExplanation]  = useState(false);
  const [learningState,    setLearningState]    = useState('');
  const [proactiveMessage, setProactiveMessage] = useState(null);
  const [nextTargetLevel,  setNextTargetLevel]  = useState(null);
  const [cognitiveLoad,    setCognitiveLoad]    = useState(0);
  const [showPuzzle,       setShowPuzzle]       = useState(false);
  const [puzzleType,       setPuzzleType]       = useState('rocket'); // 'rocket', 'quest', 'circuit', etc.
  const [hintRequested,    setHintRequested]    = useState(false); // student asked for the hint during the attempt (before submit)

  // Puzzle Handlers
  // Handle Brain Break Trigger
  const triggerBrainBreak = useCallback(() => {
    if (isAssessmentDone || showPuzzle) return;
    setPuzzleType('rocket'); // Use the high-action Expression Rocket game
    setShowPuzzle(true);
  }, [isAssessmentDone, showPuzzle]);

  const handlePuzzleWin = () => {
    setShowPuzzle(false);
    speak("Great job! You're ready to focus again.");
  };

  const handlePuzzleFail = () => {
    setShowPuzzle(false);
    speak("No worries, take a deep breath. Let's try to finish this question.");
  };

  // ── Handle answer selection ────────────────────────────────────────────────
  const handleSelect = useCallback((opt) => {
    if (submitted) return;
    if (selectedAnswer && selectedAnswer !== opt) {
      setAnswerChanges((c) => c + 1);
    }
    setSelectedAnswer(opt);
    speak(`Selected ${opt}`);
  }, [submitted, selectedAnswer, speak]);

  // ── Handle submit ─────────────────────────────────────────────────────────
  const handleSubmit = async () => {
    if (!selectedAnswer || submitted || submitting) return;
    setSubmitting(true);
    timer.stop();
    try {

    const isCorrect     = selectedAnswer === currentQuestion.correctAnswer ? 1 : 0;
    const responseTime  = timer.elapsed;
    const expression    = webcam.currentExpression;

    if (isCorrect) {
      speak("Correct! Well done.");
    } else {
      speak("Not quite right. Let's look at the hint.");
    }

    // 1. Predict learning state
    let predictedState = 'partial_understanding';
    try {
      const prediction = await predictLearningState({
        correctness:        isCorrect,
        responseTime,
        answerChanges,
        quizLevel:          currentQuestion.quizLevel,
        detectedExpression: expression,
      });
      predictedState = prediction.learningState;
    } catch {
      // Fallback
    }
    setLearningState(predictedState);

    // 2. Calculate Cognitive Load & the adaptive difficulty *signal*.
    // NOTE: this only labels the response for the report/research layer — it does
    // NOT reroute the quiz. Every student answers all 9 questions (3 per level,
    // Level 1 → 2 → 3) so the assessment is complete and comparable across students.
    const load = calculateCognitiveLoad({ responseTime, answerChanges, expression });
    const isLastInLevel = currentQuestionIdx === currentQuestions.length - 1;
    const target = isLastInLevel
      ? getAdaptiveNextLevel(currentLevel, isCorrect === 1, load)
      : currentLevel;
    setCognitiveLoad(load);
    setNextTargetLevel(target);

    // 3. Evaluate support logic
    const { showHint: hint, showExplanation: explain } = evaluateResponse({
      isCorrect:          isCorrect === 1,
      learningState:      predictedState,
      detectedExpression: expression,
      responseTime,
    });
    setShowHint(hint);
    setShowExplanation(explain);
    const hintWasUsed = hint || hintRequested;

    // 3. Build response object
    const responseData = {
      studentId:          student.studentId,
      lessonId:           lesson.lessonId,
      questionId:         currentQuestion.questionId,
      quizLevel:          currentQuestion.quizLevel,
      conceptTag:         currentQuestion.conceptTag,
      selectedAnswer,
      correctness:        isCorrect,
      responseTime,
      answerChanges,
      detectedExpression: expression,
      learningState:      predictedState,
      cognitiveLoad:      load,
      adaptiveLevel:      target,
      hintUsed:           hintWasUsed,
      sessionId,
    };

    ctxSaveResponse(responseData);
    try { await saveResponse(responseData); } catch { /* non-blocking */ }

    setSubmitted(true);
    } catch (err) {
      console.error('submit failed', err);
      // Never trap the student on "Analysing…" — let them see feedback and move on
      setShowExplanation(true);
      setSubmitted(true);
      setNextTargetLevel(prev => prev ?? currentLevel);
    } finally {
      setSubmitting(false);
    }
  };

  // ── Advance to next question ───────────────────────────────────────────────
  // Always sequential: Level 1 (3 Qs) → Level 2 (3 Qs) → Level 3 (3 Qs) = 9 total.
  // The adaptive difficulty stays a recorded signal (nextTargetLevel) but never
  // skips or repeats questions, so every student sees the full 9-question set.
  const handleNext = () => {
    nextQuestion(currentLevel);
  };

  // ── Handle voice commands (REGISTERED GLOBALLY) ───────────────────────────
  useEffect(() => {
    if (!currentQuestion) return;

    const readOptions = () => {
      const text = currentQuestion.options
        .map((opt, i) => `${String.fromCharCode(65 + i)}: ${opt}`)
        .join(". ");
      speak(`The options are: ${text}`);
    };

    const optionCommands = {};
    currentQuestion.options.forEach((opt, idx) => {
      const label = String.fromCharCode(97 + idx); // a, b, c, d
      optionCommands[`option ${label}`] = () => handleSelect(opt);
      optionCommands[`select ${label}`] = () => handleSelect(opt);
      optionCommands[label] = () => handleSelect(opt);
      // Also add the full text of the option as a command for "research level" natural interaction
      optionCommands[opt.toLowerCase()] = () => handleSelect(opt);
    });

    const unregister = registerCommands('assessment', {
      ...optionCommands,
      'read options': readOptions,
      'what are my options': readOptions,
      'choices': readOptions,
      'options': readOptions,
      'submit':   () => { if (!submitted && selectedAnswer) handleSubmit(); },
      'answer':   () => { if (!submitted && selectedAnswer) handleSubmit(); },
      'next':     () => { if (submitted) handleNext(); },
      'continue': () => { if (submitted) handleNext(); },
      'help':     () => { if (submitted && showHint) speak(currentQuestion.hint); },
      'hint':     () => { if (submitted && showHint) speak(currentQuestion.hint); },
      'break':    () => triggerBrainBreak(),
      'puzzle':   () => triggerBrainBreak(),
      'game':     () => triggerBrainBreak(),
      'recharge': () => triggerBrainBreak(),
      'rocket':   () => triggerBrainBreak()
    });

    return unregister;
  }, [currentQuestion, selectedAnswer, submitted, showHint, handleSelect, handleSubmit, handleNext, registerCommands, speak, triggerBrainBreak]);

  // Sync detected expression into context
  useEffect(() => {
    setExpression(webcam.currentExpression);
  }, [webcam.currentExpression]);

  // Auto-pop the hint the moment the webcam detects a frustrated expression during the attempt
  useEffect(() => {
    if (submitted || hintRequested) return;
    if (webcam.currentExpression === 'frustrated') {
      setHintRequested(true);
      speak("I can see you're frustrated. Here's a hint.");
    }
  }, [webcam.currentExpression, submitted, hintRequested, speak]);

  // Reset per-question state and start timer when question changes
  useEffect(() => {
    if (!currentQuestion) return;
    setSelectedAnswer('');
    setAnswerChanges(0);
    setSubmitted(false);
    setSubmitting(false);
    setShowHint(false);
    setShowExplanation(false);
    setLearningState('');
    setProactiveMessage(null);
    setNextTargetLevel(null);
    setCognitiveLoad(0);
    setHintRequested(false);
    timer.start();
  }, [currentQuestion?._id, currentQuestion?.questionId]);

  // Proactive support monitor
  useEffect(() => {
    if (!currentQuestion || submitted || proactiveMessage) return;

    const msg = evaluateProactiveSupport({
      elapsed: timer.elapsed,
      answerChanges,
      expression: webcam.currentExpression,
      hintText: currentQuestion.hint
    });

    if (msg) {
      setProactiveMessage(msg);
      speak("I notice you're thinking hard. Here's a quick tip.");
      
      // Automatic Brain Break if load is extreme
      if (cognitiveLoad > 85 && !showPuzzle) {
        setShowPuzzle(true);
      }
    }
  }, [timer.elapsed, answerChanges, webcam.currentExpression, submitted, proactiveMessage, currentQuestion, speak, cognitiveLoad, showPuzzle]);

  // Navigate to report after all questions done
  useEffect(() => {
    if (isAssessmentDone) {
      setShowPuzzle(false); // Force close any puzzles at the end
      const finishAssessment = async () => {
        try { await saveBulkResponses(responses); } catch { }
        navigate('/report');
      };
      finishAssessment();
    }
  }, [isAssessmentDone]);

  // Redirect if no active assessment
  useEffect(() => {
    if (!student || !lesson) navigate('/login');
  }, []);

  if (!currentQuestion) {
    return <div className="assessment-loading"><div className="spin-ring"/><p>Loading questions…</p></div>;
  }

  return (
    <div className="assessment-page">

      {/* ── Research & Gaming Layer ── */}
      {/* <ResearchAvatar expression={webcam.currentExpression} learningState={learningState} /> */}
      
      <ExpressionRocket
        isOpen={showPuzzle && puzzleType === 'rocket'}
        currentExpression={webcam.currentExpression}
        onWin={handlePuzzleWin}
        avatarSpeak={speak}
      />

      <ExpressionQuest
        isOpen={showPuzzle && puzzleType === 'quest'}
        currentExpression={webcam.currentExpression}
        onWin={handlePuzzleWin}
        avatarSpeak={speak}
      />

      <CircuitLogicPuzzle
        isOpen={showPuzzle && puzzleType === 'circuit'}
        onWin={handlePuzzleWin}
        onFail={handlePuzzleFail}
        avatarSpeak={speak}
      />

      <BrainBreakPuzzle 
        isOpen={showPuzzle && puzzleType === 'pattern'} 
        onWin={handlePuzzleWin} 
        onFail={handlePuzzleFail} 
        avatarSpeak={speak} 
      />

      <MemoryMatchPuzzle
        isOpen={showPuzzle && puzzleType === 'memory'}
        onWin={handlePuzzleWin}
        onFail={handlePuzzleFail}
        avatarSpeak={speak}
      />

      <MathBlitzPuzzle
        isOpen={showPuzzle && puzzleType === 'math'}
        onWin={handlePuzzleWin}
        onFail={handlePuzzleFail}
        avatarSpeak={speak}
      />
      
      {/* ── Webcam (top-right corner) ── */}
      <div className="webcam-corner">
        <WebcamCapture {...webcam} />
      </div>

      <div className="assessment-container">

        {/* ── Header ── */}
        <div className="assessment-header">
          <div className="assessment-meta">
            <span className="lesson-chip">📖 {lesson?.title}</span>
            <span className="level-chip">{getLevelLabel(currentLevel)}</span>
            {submitted && (
              <span className="stat-chip" style={{ background: 'rgba(14, 124, 102,0.05)', borderColor: 'rgba(14, 124, 102,0.2)', color: '#0E7C66' }}>
                🧠 Load: {Math.round(cognitiveLoad)}%
              </span>
            )}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Timer elapsed={timer.elapsed} />
          </div>
        </div>

        {/* ── Progress ── */}
        <LevelProgressBar
          currentLevel={currentLevel}
          answeredCount={answeredCount}
          totalQuestions={totalQuestions}
        />

        {/* ── Question card ── */}
        <div className="question-card" key={currentQuestion._id || currentQuestion.questionId}>

          {/* Question meta */}
          <div className="question-meta">
            <span className="question-num">
              Question {(currentLevel - 1) * currentQuestions.length + currentQuestionIdx + 1}
            </span>
            <span className="concept-tag">🏷 {currentQuestion.conceptTag}</span>
            {submitted && learningState && (
              <span className="state-badge" style={{ background: getLearningStateColor(learningState) + '22', color: getLearningStateColor(learningState), border: `1px solid ${getLearningStateColor(learningState)}44` }}>
                {learningState.replace(/_/g, ' ')}
              </span>
            )}
          </div>

          {/* Answer change counter */}
          <div className="answer-stats">
            <span>🔄 Changes: <strong>{answerChanges}</strong></span>
            {submitted && <span style={{ color: selectedAnswer === currentQuestion.correctAnswer ? '#22c55e' : '#ef4444' }}>
              {selectedAnswer === currentQuestion.correctAnswer ? '✓ Correct' : '✗ Incorrect'}
            </span>}
          </div>

          {/* Question text */}
          <h2 className="question-text">{currentQuestion.questionText}</h2>

          {/* Proactive Student Supporter */}
          <HintBox type="support" hint={proactiveMessage} visible={!submitted && !!proactiveMessage} />

          {/* Hint auto-pops up only when the webcam detects a frustrated expression during the attempt */}
          <HintBox hint={currentQuestion.hint} visible={!submitted && hintRequested} />

          {/* Answer options */}
          <AnswerOptions
            options={currentQuestion.options}
            selectedAnswer={selectedAnswer}
            onSelect={handleSelect}
            disabled={submitted}
            correctAnswer={currentQuestion.correctAnswer}
            showResult={submitted}
          />

          {/* Feedback: Hint */}
          <HintBox hint={currentQuestion.hint} visible={submitted && showHint} />

          {/* Feedback: Theory explanation */}
          <TheoryExplanation
            correctAnswer={currentQuestion.correctAnswer}
            explanation={currentQuestion.shortTheoryExplanation}
            visible={submitted && showExplanation}
          />

          {/* Action buttons */}
          <div className="question-actions">
            {!submitted ? (
              <button
                id="btn-submit-answer"
                className="btn-submit"
                onClick={handleSubmit}
                disabled={!selectedAnswer || submitting}
              >
                {submitting ? <><span className="spinner-sm"/>Analysing…</> : '✅ Submit Answer'}
              </button>
            ) : (
              <button id="btn-next-question" className="btn-next" onClick={handleNext}>
                {(currentLevel === 3 && currentQuestionIdx + 1 === currentQuestions.length) ? '📊 View Report' : 'Next Question →'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AssessmentPage;
